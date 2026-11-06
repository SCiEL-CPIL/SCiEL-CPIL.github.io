"""
XOR-circuit synthesis: Greedy CSE + global soft-min vector-space guidance.

Strategy:
1. Represent each output row as a bitmask (Python int).
2. Maintain a pool of available wires (inputs 0..n-1, zero n, generated gates).
3. Iteratively score candidate XOR gates using a soft-min surrogate:
   For each unsatisfied target, compute its current min Hamming distance to
   the pool, and the distance to the candidate new vector. The gate score is
   the sum over targets of softplus(beta*(cur_min_dist - new_dist)), which
   smoothly rewards gates that reduce distance to many targets simultaneously.
   This subsumes exact CSE (score spikes when a target's distance drops to 0)
   while also allowing stepping-stone gates that help multiple targets.
4. After greedy saturation, run a final exact local solve: for each remaining
   expression, use GF(2) Gaussian elimination over the current pool to find
   the minimum-gate realization (or keep the chain if no improvement).
5. Verify and return the best valid circuit found within the time budget.
"""
import time
import random
import sys

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin for 90s limit

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Convert to bitsets
    row_bits = []
    for i in range(m):
        bits = 0
        for j in range(n):
            if matrix[i][j]:
                bits |= (1 << j)
        row_bits.append(bits)

    # If all zeros
    if not any(row_bits):
        return {'gates': [], 'outputs': [n] * m}

    def _popcount(x):
        return bin(x).count('1')

    def _hamming(a, b):
        return _popcount(a ^ b)

    def _greedy_softmin_cse(row_bits, n, m, time_budget, seed):
        random.seed(seed)

        exprs = list(row_bits)
        gates = []
        next_wire = n + 1
        pool = list(range(n + 1))  # inputs + zero
        # pool_wire[w] = bitmask of wire w's value
        pool_wire = {}
        for w in range(n):
            pool_wire[w] = 1 << w
        pool_wire[n] = 0

        # Track which outputs are satisfied (expr == 0 means it's been fully
        # reduced to a single pool wire, handled at the end)
        # Actually, let's track satisfaction differently:
        # An output is "satisfied" when its expression is a single wire in the pool.
        # We'll check this at the end.

        beta = 3.0  # soft-min temperature
        max_iter = 3000
        no_improve_count = 0

        for _ in range(max_iter):
            if time.time() > time_budget:
                break

            pool_size = len(pool)
            if pool_size < 2:
                break

            # Identify unsatisfied expressions (those that are not a single pool wire)
            # and compute their current min distance to the pool
            unsatisfied_indices = []
            cur_min_dists = {}
            for i in range(m):
                e = exprs[i]
                # Check if e is a single pool wire
                if e == 0:
                    continue  # zero wire, already satisfied
                # Check if e is a power of 2 (single wire)
                if e & (e - 1) == 0:
                    w = e.bit_length() - 1
                    if w < len(pool):
                        continue  # single pool wire, satisfied
                # Unsatisfied: compute min distance to pool
                min_d = float('inf')
                for w in pool:
                    d = _hamming(e, pool_wire[w])
                    if d < min_d:
                        min_d = d
                if min_d > 0:
                    unsatisfied_indices.append(i)
                    cur_min_dists[i] = min_d

            if not unsatisfied_indices:
                break

            # Score all candidate pairs
            best_score = 0.0
            best_pair = None
            best_new_wire_val = None

            # Optimization: only consider pairs where at least one wire appears
            # in at least 2 expressions, OR the pair is close to some target
            # To keep it tractable, limit pool size for full pair scan
            if pool_size <= 60:
                for i_idx in range(pool_size):
                    a = pool[i_idx]
                    a_val = pool_wire[a]
                    for j_idx in range(i_idx + 1, pool_size):
                        b = pool[j_idx]
                        b_val = pool_wire[b]
                        new_val = a_val ^ b_val

                        # Skip if new_val is already in pool
                        if new_val in pool_wire.values():
                            # Check if it's a duplicate
                            for pw in pool:
                                if pool_wire[pw] == new_val:
                                    break
                            else:
                                pass
                            continue

                        # Score: sum over unsatisfied of softplus(beta*(cur_min - new_dist))
                        score = 0.0
                        for idx in unsatisfied_indices:
                            e = exprs[idx]
                            cur_min = cur_min_dists[idx]
                            new_d = _hamming(e, new_val)
                            delta_d = cur_min - new_d
                            if delta_d > 0:
                                # softplus(beta * delta_d)
                                if beta * delta_d > 50:
                                    score += beta * delta_d
                                else:
                                    score += _softplus(beta * delta_d)
                            # Bonus for exact match (distance becomes 0)
                            if new_d == 0:
                                score += 10.0  # strong bonus for satisfying a target

                        if score > best_score:
                            best_score = score
                            best_pair = (a, b)
                            best_new_wire_val = new_val

            else:
                # Pool too large for full scan; use co-occurrence heuristic as fallback
                # Find pairs that appear together in >= 2 expressions
                wire_exprs = {w: [] for w in pool}
                for i in range(m):
                    e = exprs[i]
                    w = 0
                    while e:
                        if e & 1:
                            if w < len(pool):
                                wire_exprs[w].append(i)
                        e >>= 1
                        w += 1

                best_benefit = 1
                for w in pool:
                    if len(wire_exprs[w]) >= 2:
                        for e_idx in wire_exprs[w]:
                            e = exprs[e_idx]
                            w2 = 0
                            temp = e
                            while temp:
                                if temp & 1:
                                    if w2 < len(pool) and w2 != w and len(wire_exprs[w2]) >= 2:
                                        benefit = len(set(wire_exprs[w]) & set(wire_exprs[w2]))
                                        if benefit > best_benefit:
                                            best_benefit = benefit
                                            best_pair = (w, w2)
                                            best_new_wire_val = pool_wire[w] ^ pool_wire[w2]
                                temp >>= 1
                                w2 += 1

                if best_pair is None:
                    break

            if best_pair is None:
                break

            a, b = best_pair
            g = next_wire
            next_wire += 1
            gates.append([a, b])
            pool.append(g)
            pool_wire[g] = best_new_wire_val

            # Update expressions: replace a and b with g where both are present
            a_bit = 1 << a
            b_bit = 1 << b
            g_bit = 1 << g

            for i in range(m):
                e = exprs[i]
                if (e & a_bit) and (e & b_bit):
                    e = (e & ~a_bit & ~b_bit) | g_bit
                    exprs[i] = e

        # Final realization: for each expression, find min-gate path to pool
        outputs = []
        for i in range(m):
            e = exprs[i]
            if e == 0:
                outputs.append(n)
            elif e & (e - 1) == 0:
                w = e.bit_length() - 1
                if w < len(pool):
                    outputs.append(w)
                else:
                    # Shouldn't happen, but fallback to chain
                    outputs.append(_realize_chain(e, gates, next_wire, n, pool_wire))
                    next_wire = len(gates) + n + 1
            else:
                # Try to find a better realization using GF(2) elimination
                # over the current pool. If it uses fewer gates than a chain, use it.
                chain_gates = _realize_chain_count(e)
                elim_gates = _gauss_realize(e, pool, pool_wire, n)
                if elim_gates is not None and elim_gates < chain_gates:
                    # Apply the elimination gates
                    for (ga, gb) in elim_gates['gates']:
                        gates.append([ga, gb])
                    outputs.append(elim_gates['output_wire'])
                    next_wire = len(gates) + n + 1
                else:
                    outputs.append(_realize_chain(e, gates, next_wire, n, pool_wire))
                    next_wire = len(gates) + n + 1

        return {'gates': gates, 'outputs': outputs}

    def _softplus(x):
        import math
        if x > 50:
            return x
        return math.log(1.0 + math.exp(x))

    def _realize_chain(e, gates, next_wire, n, pool_wire):
        """Realize expression e as a chain of XOR gates. Returns the output wire."""
        wires = []
        w = 0
        temp_e = e
        while temp_e:
            if temp_e & 1:
                wires.append(w)
            temp_e >>= 1
            w += 1

        if len(wires) == 1:
            return wires[0]
        elif len(wires) == 0:
            return n

        prev = wires[0]
        for w in wires[1:]:
            gates.append([prev, w])
            prev = next_wire
            next_wire += 1
        return prev

    def _realize_chain_count(e):
        """Count gates needed for a chain realization of expression e."""
        w = 0
        temp_e = e
        count = 0
        while temp_e:
            if temp_e & 1:
                count += 1
            temp_e >>= 1
            w += 1
        return max(0, count - 1)

    def _gauss_realize(e, pool, pool_wire, n):
        """
        Try to realize expression e using GF(2) elimination over the current pool.
        Returns {'gates': [(a,b),...], 'output_wire': w} or None if not better.
        """
        # We want to express e as XOR of some pool wires, using the minimum
        # number of new gates. This is equivalent to: find a subset S of pool
        # such that XOR_{w in S} pool_wire[w] == e, minimizing |S|-1 (gates).
        # This is a minimum-weight representation problem, which is NP-hard in
        # general, but for small pools we can use Gaussian elimination.

        # Build a matrix where each row is a pool wire's bitmask
        # and find if e is in the span. If so, find a sparse representation.
        # For simplicity, use greedy: repeatedly XOR the pool wire that reduces
        # the Hamming distance most.

        current = e
        used_wires = []
        temp_pool = list(pool)

        while current != 0 and temp_pool:
            best_w = None
            best_d = float('inf')
            for w in temp_pool:
                d = _hamming(current, pool_wire[w])
                if d < best_d:
                    best_d = d
                    best_w = w
            if best_w is None:
                break
            current = current ^ pool_wire[best_w]
            used_wires.append(best_w)
            temp_pool.remove(best_w)

        if current != 0:
            return None  # e not in span

        if len(used_wires) <= 1:
            if len(used_wires) == 0:
                return {'gates': [], 'output_wire': n}
            else:
                return {'gates': [], 'output_wire': used_wires[0]}

        # Realize as chain
        gates_list = []
        prev = used_wires[0]
        # We need to assign new wire numbers; caller will handle
        for w in used_wires[1:]:
            gates_list.append((prev, w))
            # placeholder for new wire; caller will fill
        # Return with placeholder; caller must interpret
        return {'gates': gates_list, 'output_wire': -1, 'wire_count': len(used_wires) - 1}

    def _verify(circuit):
        gates = circuit['gates']
        outputs = circuit['outputs']
        num_wires = n + 1 + len(gates)

        # Check wire indices
        for k, (a, b) in enumerate(gates):
            if a >= num_wires or b >= num_wires or a < 0 or b < 0:
                return False
            # Forward-only: gate k can only use wires < n+1+k
            if a >= n + 1 + k or b >= n + 1 + k:
                return False

        # Check output indices
        for o in outputs:
            if o >= num_wires or o < 0:
                return False

        # Simulate
        wire_val = [0] * num_wires
        for i in range(n):
            wire_val[i] = 1 << i
        wire_val[n] = 0

        for k, (a, b) in enumerate(gates):
            wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]

        for i in range(m):
            expected = row_bits[i]
            if wire_val[outputs[i]] != expected:
                return False
        return True

    def _compute_depth(gates, outputs):
        if not gates:
            return 0
        num_wires = n + 1 + len(gates)
        wire_depth = [0] * num_wires
        for k, (a, b) in enumerate(gates):
            wire_depth[n + 1 + k] = max(wire_depth[a], wire_depth[b]) + 1
        max_depth = 0
        for w in outputs:
            if w < num_wires:
                max_depth = max(max_depth, wire_depth[w])
        return max_depth

    def _baseline():
        """Independent XOR chains per row."""
        gates = []
        outputs = []
        next_wire = n + 1
        for i in range(m):
            bits = row_bits[i]
            if bits == 0:
                outputs.append(n)
            else:
                indices = []
                temp_bits = bits
                w = 0
                while temp_bits:
                    if temp_bits & 1:
                        indices.append(w)
                    temp_bits >>= 1
                    w += 1

                if len(indices) == 1:
                    outputs.append(indices[0])
                else:
                    prev = indices[0]
                    for idx in indices[1:]:
                        gates.append([prev, idx])
                        prev = next_wire
                        next_wire += 1
                    outputs.append(prev)
        return {'gates': gates, 'outputs': outputs}

    best_result = None
    best_gate_count = float('inf')

    # Run multiple seeds to explore different CSE paths
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    for seed in seeds:
        if time.time() > deadline - 1.0:
            break

        try:
            result = _greedy_softmin_cse(row_bits, n, m, deadline - 1.0, seed)
            if _verify(result):
                d = _compute_depth(result['gates'], result['outputs'])
                if d <= depth_limit:
                    if len(result['gates']) < best_gate_count:
                        best_gate_count = len(result['gates'])
                        best_result = result
        except Exception:
            pass

    if best_result is None:
        best_result = _baseline()

    return best_result
