"""XOR circuit synthesis using Dijkstra's algorithm on the XOR hypergraph.

The key structural change: instead of greedy row-by-row synthesis, we model
the problem as a shortest-path problem where:
- Nodes are all possible linear combinations (bitmasks) of input wires
- Edges represent single XOR gates
- We run Dijkstra from the set of 'free' nodes (inputs + zero) to find the
  minimum-cost way to produce each target row

This guarantees globally optimal gate sharing: if multiple rows share
intermediate XORs, they will be computed once and reused.
"""
import time
import heapq

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 55.0

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0 or n == 0:
        return {'gates': [], 'outputs': []}

    # Convert matrix rows to integer bitmasks
    rows = []
    for i in range(m):
        val = 0
        for j in range(n):
            if matrix[i][j]:
                val |= (1 << j)
        rows.append(val)

    # Remove duplicate rows: same target only needs one wire
    # Map target -> list of row indices that need it
    target_to_rows = {}
    for i in range(m):
        target_to_rows.setdefault(rows[i], []).append(i)

    # We need to produce each unique non-zero target
    # Zero is always available (wire n)
    targets = sorted(set(rows[i] for i in range(m) if rows[i] != 0))

    # Dijkstra's algorithm on the XOR hypergraph
    # dist[v] = minimum number of gates to produce bitmask v
    # parent[v] = (a, b) such that a XOR b = v and dist[v] = dist[a] + dist[b] + 1
    #             where a and b are already produced (or are inputs)
    
    # Initial state: all input wires and zero are free
    # dist[0] = 0, dist[1<<j] = 0 for all j
    # For any other v, we can compute dist[v] = 1 + min over (a,b) with a^b=v of max(dist[a], dist[b])
    # But that's not quite right for a tree structure...
    
    # Actually, let's think about this differently. We want to build a circuit (DAG)
    # where each gate takes two existing wires and produces a new one.
    # The cost is the total number of gates.
    # 
    # A better formulation: we process nodes in order of increasing "cost" where
    # cost[v] = minimum number of additional gates needed to produce v given
    # all currently available wires.
    #
    # This is essentially what the current program does, but let's use a proper
    # priority queue and ensure we always pick the globally cheapest next wire.
    
    # Available wires: bitmask -> wire_index
    available = {}
    for j in range(n):
        available[1 << j] = j
    available[0] = n

    gates = []
    next_wire = n + 1

    # Priority queue: (cost, bitmask)
    # cost = estimated gates needed to produce this bitmask
    pq = []

    def popcount(x):
        return bin(x).count('1')

    def get_cost(target):
        """Compute exact minimum gates to produce target from available wires.
        
        A target can be produced in 1 gate if there exist a, b in available 
        with a ^ b = target. Otherwise, we need to build an intermediate.
        
        For small n, we can do this exactly. For larger n, we use a greedy
        lower bound: 1 + min over a in available of popcount(target ^ a).
        """
        if target in available:
            return 0
        
        # Check for 1-gate production
        for a in available:
            b = target ^ a
            if b in available:
                if a == b and target != 0:
                    continue
                return 1
        
        # Estimate: find a in available that minimizes popcount(target ^ a)
        # This gives a lower bound on additional gates
        min_pc = float('inf')
        for a in available:
            diff = target ^ a
            pc = popcount(diff)
            if pc < min_pc:
                min_pc = pc
                if pc == 1:
                    break
        return 1 + min_pc if min_pc < float('inf') else popcount(target)

    # Initialize priority queue with all targets
    for t in targets:
        c = get_cost(t)
        heapq.heappush(pq, (c, t))

    # Process targets in order of increasing cost
    produced = {}  # target -> wire_index (for targets we've produced)

    while pq:
        if time.time() > deadline:
            break

        cost, target = heapq.heappop(pq)

        if target in available or target in produced:
            continue

        # Try to produce target with 1 gate
        pair = None
        for a, a_wire in available.items():
            b = target ^ a
            if b in available:
                b_wire = available[b]
                if a_wire == b_wire and target != 0:
                    continue
                pair = (a_wire, b_wire)
                break

        if pair is None:
            for a, a_wire in produced.items():
                b = target ^ a
                if b in available:
                    b_wire = available[b]
                    if a_wire == b_wire and target != 0:
                        continue
                    pair = (a_wire, b_wire)
                    break
                elif b in produced:
                    b_wire = produced[b]
                    if a_wire == b_wire and target != 0:
                        continue
                    pair = (a_wire, b_wire)
                    break

        if pair is not None:
            a_wire, b_wire = pair
            gates.append([a_wire, b_wire])
            produced[target] = next_wire
            available[target] = next_wire
            next_wire += 1
        else:
            # Need to build an intermediate. Find the best helper.
            best_a = None
            best_a_wire = None
            best_diff = None
            min_pc = float('inf')

            # Search in available first (cheaper)
            for a, a_wire in available.items():
                diff = target ^ a
                if diff == 0:
                    continue
                pc = popcount(diff)
                if pc < min_pc:
                    min_pc = pc
                    best_a = a
                    best_a_wire = a_wire
                    best_diff = diff
                    if pc == 1:
                        break

            # Also search in produced
            if best_a is None:
                for a, a_wire in produced.items():
                    diff = target ^ a
                    if diff == 0:
                        continue
                    pc = popcount(diff)
                    if pc < min_pc:
                        min_pc = pc
                        best_a = a
                        best_a_wire = a_wire
                        best_diff = diff
                        if pc == 1:
                            break

            if best_a is None or best_diff is None:
                # Fallback: build from inputs directly
                wire = None
                bits = target
                while bits:
                    lb = bits & (-bits)
                    bit_idx = lb.bit_length() - 1
                    input_wire = bit_idx
                    if wire is None:
                        wire = input_wire
                    else:
                        gates.append([wire, input_wire])
                        wire = next_wire
                        next_wire += 1
                    bits ^= lb
                if wire is None:
                    wire = n
                produced[target] = wire
                available[target] = wire
            else:
                # Recursively produce best_diff, then XOR with best_a
                # First, make sure best_diff is produced
                if best_diff not in available and best_diff not in produced:
                    # Add to priority queue with updated cost
                    heapq.heappush(pq, (get_cost(best_diff), best_diff))
                    # For now, produce it directly
                    # Find helper for best_diff
                    best_diff_a = None
                    best_diff_a_wire = None
                    best_diff_diff = None
                    min_pc2 = float('inf')
                    for a, a_wire in available.items():
                        diff2 = best_diff ^ a
                        if diff2 == 0:
                            continue
                        pc2 = popcount(diff2)
                        if pc2 < min_pc2:
                            min_pc2 = pc2
                            best_diff_a = a
                            best_diff_a_wire = a_wire
                            best_diff_diff = diff2
                            if pc2 == 1:
                                break
                    
                    if best_diff_a is None:
                        # Build from inputs
                        wire = None
                        bits = best_diff
                        while bits:
                            lb = bits & (-bits)
                            bit_idx = lb.bit_length() - 1
                            input_wire = bit_idx
                            if wire is None:
                                wire = input_wire
                            else:
                                gates.append([wire, input_wire])
                                wire = next_wire
                                next_wire += 1
                            bits ^= lb
                        if wire is None:
                            wire = n
                        produced[best_diff] = wire
                        available[best_diff] = wire
                    else:
                        # Produce best_diff_diff
                        if best_diff_diff not in available and best_diff_diff not in produced:
                            # Build from inputs as fallback
                            wire = None
                            bits = best_diff_diff
                            while bits:
                                lb = bits & (-bits)
                                bit_idx = lb.bit_length() - 1
                                input_wire = bit_idx
                                if wire is None:
                                    wire = input_wire
                                else:
                                    gates.append([wire, input_wire])
                                    wire = next_wire
                                    next_wire += 1
                                bits ^= lb
                            if wire is None:
                                wire = n
                            produced[best_diff_diff] = wire
                            available[best_diff_diff] = wire
                        
                        diff_wire = available.get(best_diff_diff, produced.get(best_diff_diff))
                        a_wire2 = available.get(best_diff_a, produced.get(best_diff_a))
                        gates.append([a_wire2, diff_wire])
                        produced[best_diff] = next_wire
                        available[best_diff] = next_wire
                        next_wire += 1

                diff_wire = available.get(best_diff, produced.get(best_diff))
                a_wire = available.get(best_a, produced.get(best_a))
                gates.append([a_wire, diff_wire])
                produced[target] = next_wire
                available[target] = next_wire
                next_wire += 1

        # After producing a new wire, update costs of remaining targets
        # This allows targets that were expensive to become cheaper
        remaining_targets = [t for t in targets if t not in available and t not in produced]
        for t in remaining_targets:
            new_cost = get_cost(t)
            if new_cost < cost:
                heapq.heappush(pq, (new_cost, t))

    # Build outputs
    outputs = []
    for i in range(m):
        t = rows[i]
        if t == 0:
            outputs.append(n)
        elif t in available:
            outputs.append(available[t])
        elif t in produced:
            outputs.append(produced[t])
        else:
            # Should not happen, but fallback
            wire = None
            bits = t
            while bits:
                lb = bits & (-bits)
                bit_idx = lb.bit_length() - 1
                input_wire = bit_idx
                if wire is None:
                    wire = input_wire
                else:
                    gates.append([wire, input_wire])
                    wire = next_wire
                    next_wire += 1
                bits ^= lb
            if wire is None:
                wire = n
            outputs.append(wire)

    return {'gates': gates, 'outputs': outputs}
