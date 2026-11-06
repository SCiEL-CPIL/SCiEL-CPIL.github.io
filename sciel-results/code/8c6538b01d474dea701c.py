"""Synthesis of XOR circuits under bounded depth, minimizing gate count."""
import time
from collections import deque

# EVOLVE-BLOCK-START
def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # Parse matrix
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Convert each row to an integer
    targets = []
    for i in range(m):
        v = 0
        for j in range(n):
            if matrix[i][j]:
                v |= (1 << j)
        targets.append(v)

    # Trivial case: all zero
    if all(v == 0 for v in targets):
        return {'gates': [], 'outputs': [n] * m}

    # Case with only unit vectors and zeros
    has_nontrivial = any(v > 1 and v & (v - 1) for v in targets)  # weight >= 2
    if not has_nontrivial:
        gates = []
        outputs = []
        for v in targets:
            if v == 0:
                outputs.append(n)
            else:
                # unit vector: find the bit
                outputs.append(v.bit_length() - 1)
        return {'gates': gates, 'outputs': outputs}

    # BFS-based synthesis with subexpression sharing
    # State: frozenset of available intermediate values (as ints)
    # We build the circuit level by level (depth-limited)
    
    def bfs_synthesize(targets, n, depth_limit, time_limit=80):
        """
        BFS over levels of gates. At each level, we can add any XOR of two
        previously available values. We track which target values have been
        synthesized and try to minimize total gate count.
        
        Strategy: 
        - Level 0: available = input wires (0..n-1) + zero wire (n)
        - Each level: we can create new values by XOR-ing pairs of available values
        - We stop when all targets are available
        - Minimize number of gates (total across all levels)
        
        Since we want minimum gate count, we use a modified approach:
        Greedy with backtracking isn't feasible. Instead, use a heuristic:
        At each level, add the most useful new values (those that are targets
        or that help build other targets).
        """
        
        # Available values at level 0: input wires and zero
        # Wire mapping: wire i -> value (1 << i) for i in 0..n-1, wire n -> 0
        # Gate k (0-indexed) -> wire n+1+k
        
        # We track:
        # - avail: dict from value (int) to wire index
        # - gates: list of [w1, w2]
        # - gate_values: list of values produced by each gate
        
        avail = {}
        for i in range(n):
            avail[1 << i] = i
        avail[0] = n
        
        gates = []
        gate_values = []
        next_gate_id = 0
        
        # Targets still needed
        remaining = set(targets)
        
        # We'll do iterative deepening on depth, but with a time budget
        # For each depth level, we decide which new values to introduce
        
        # Heuristic: at each step, introduce values that are either:
        # 1. Directly a target
        # 2. Useful for building remaining targets (appear in many target decompositions)
        
        # Precompute: for each target, what's its "complexity" (weight)?
        # A value of weight w needs w-1 gates if built from scratch
        
        # Greedy approach: repeatedly pick the most valuable new value
        # that can be formed from current available set
        
        # To form a value v from available set, we need v = a ^ b where a, b in avail
        # But we can also form it in multiple steps
        
        # Actually, let's use a different approach:
        # For each target, compute the minimum number of gates needed
        # given current available set, then pick the target that gives
        # the best "bang for the buck"
        
        # Simpler greedy: at each step, for each pair of available values,
        # compute their XOR. If the result is a target or reduces the
        # complexity of remaining targets, add it.
        
        # Let's use a more structured approach:
        # Represent the problem as: we have a set of "free" values (inputs + zero)
        # and need to produce all targets using XOR gates.
        # Each gate produces one new value from two existing ones.
        # This is equivalent to finding a minimum XOR circuit.
        
        # Key insight: the minimum number of gates to produce a set of target
        # values from n inputs is related to the rank and structure of the
        # target matrix.
        
        # Alternative: use the fact that any linear combination of inputs
        # can be produced. The minimum number of gates to produce k linearly
        # independent targets from n inputs is at most k*n - k (each target
        # needs at most n-1 gates if built independently, but sharing helps).
        
        # Let's try a practical heuristic:
        # 1. Build a basis of the target space
        # 2. Express all targets in terms of the basis
        # 3. Synthesize the basis vectors, then combine
        
        # First, let's try a simple but effective approach:
        # Iteratively, at each step, compute all possible XORs of pairs of
        # available values. Among those not yet available, pick the one that
        # is most "useful" (i.e., appears in the decomposition of the most
        # remaining targets, or is itself a remaining target).
        
        max_steps = n * m  # upper bound on useful gates
        step = 0
        
        while remaining and step < max_steps:
            if time.time() > time_limit + start_time:
                break
            
            # Find the most useful new value
            best_val = None
            best_score = -1
            
            # Generate candidate values: XOR of pairs of available values
            # To limit the search, only consider XORs where at least one
            # operand is a "new" gate output or an input
            # Actually, just try all pairs but be smart about scoring
            
            avail_list = list(avail.keys())
            candidates = set()
            
            # For efficiency, only consider XORs that could be targets
            # or intermediate values leading to targets
            for target in remaining:
                # What values v satisfy: target = v ^ w for some w in avail?
                # i.e., v = target ^ w for w in avail
                for w in avail_list:
                    v = target ^ w
                    if v not in avail:
                        candidates.add(v)
            
            # Also add XORs of available values that might be useful
            # (but this could be expensive, so limit it)
            if len(avail_list) <= 100:
                for i in range(len(avail_list)):
                    for j in range(i + 1, len(avail_list)):
                        v = avail_list[i] ^ avail_list[j]
                        if v not in avail:
                            candidates.add(v)
            
            # Score each candidate
            for v in candidates:
                if v in avail:
                    continue
                score = 0
                # If v is a remaining target, high score
                if v in remaining:
                    score += 100
                # Count how many remaining targets can be completed with one
                # more gate if v were available
                for t in remaining:
                    diff = t ^ v
                    if diff in avail:
                        score += 10
                # Prefer lower-weight values (easier to build)
                score -= v.bit_count() * 0.5
                
                if score > best_score:
                    best_score = score
                    best_val = v
            
            if best_val is None or best_score <= 0:
                # No good candidate found; try to build a target directly
                # Pick the target with minimum weight
                min_target = min(remaining, key=lambda v: v.bit_count())
                bits = [i for i in range(n) if min_target & (1 << i)]
                if len(bits) == 0:
                    remaining.discard(0)
                    continue
                elif len(bits) == 1:
                    remaining.discard(min_target)
                    continue
                else:
                    # Build it step by step
                    # This should have been handled by the candidate search
                    # Force-create the first intermediate
                    current = bits[0]
                    for i in range(1, len(bits)):
                        new_val = current ^ (1 << bits[i])
                        if new_val not in avail:
                            # Add this gate
                            w1 = avail[current]
                            w2 = bits[i]
                            new_wire = n + 1 + next_gate_id
                            next_gate_id += 1
                            gates.append([w1, w2])
                            gate_values.append(new_val)
                            avail[new_val] = new_wire
                            step += 1
                        current = new_val
                    remaining.discard(min_target)
                    continue
            
            # Add the best candidate as a gate
            # Find w1, w2 in avail such that w1 ^ w2 == best_val
            # We need to search for this
            found = False
            for w in avail_list:
                other = best_val ^ w
                if other in avail and other != w:
                    w1 = avail[w]
                    w2 = avail[other]
                    new_wire = n + 1 + next_gate_id
                    next_gate_id += 1
                    gates.append([w1, w2])
                    gate_values.append(best_val)
                    avail[best_val] = new_wire
                    found = True
                    break
                elif other == w and w == 0:
                    # best_val = 0 ^ w = w, but w is already avail
                    pass
            
            if not found:
                # Shouldn't happen if best_val was generated correctly
                break
            
            step += 1
            remaining.discard(best_val)
        
        # Now build the outputs
        outputs = []
        for v in targets:
            if v in avail:
                outputs.append(avail[v])
            else:
                # Fallback: build from scratch (shouldn't happen)
                bits = [i for i in range(n) if v & (1 << i)]
                if not bits:
                    outputs.append(n)
                else:
                    current = bits[0]
                    for i in range(1, len(bits)):
                        new_val = current ^ (1 << bits[i])
                        if new_val not in avail:
                            w1 = avail[current]
                            w2 = bits[i]
                            new_wire = n + 1 + next_gate_id
                            next_gate_id += 1
                            gates.append([w1, w2])
                            gate_values.append(new_val)
                            avail[new_val] = new_wire
                        current = new_val
                    outputs.append(current)
        
        return gates, outputs
    
    # Run the synthesis
    gates, outputs = bfs_synthesize(targets, n, depth_limit)
    
    # Verify the result
    # Check that the outputs are correct
    wire_values = {}
    for i in range(n):
        wire_values[i] = 1 << i
    wire_values[n] = 0
    for k, (w1, w2) in enumerate(gates):
        wire = n + 1 + k
        wire_values[wire] = wire_values[w1] ^ wire_values[w2]
    
    # Verify outputs
    for idx, (out_wire, target) in enumerate(zip(outputs, targets)):
        if wire_values.get(out_wire, -1) != target:
            # Fallback to simple synthesis
            gates = []
            outputs = []
            for v in targets:
                bits = [i for i in range(n) if v & (1 << i)]
                if not bits:
                    outputs.append(n)
                else:
                    current = bits[0]
                    for i in range(1, len(bits)):
                        new_val = current ^ (1 << bits[i])
                        w1 = avail.get(current, current)
                        w2 = bits[i]
                        new_wire = n + 1 + len(gates)
                        gates.append([w1, w2])
                        current = new_val
                    outputs.append(current)
            break
    
    return {'gates': gates, 'outputs': outputs}

# EVOLVE-BLOCK-END
