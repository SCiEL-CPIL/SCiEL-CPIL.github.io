import time
import random
import numpy as np
from collections import deque

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Convert rows to bitsets for efficient GF(2) manipulation
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

    def _verify(circuit):
        gates = circuit['gates']
        outputs = circuit['outputs']
        num_wires = n + 1 + len(gates)
        wire_val = [0] * num_wires
        for i in range(n):
            wire_val[i] = 1 << i
        wire_val[n] = 0

        for k, (a, b) in enumerate(gates):
            if a >= num_wires or b >= num_wires:
                return False
            wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]

        for i in range(m):
            expected = row_bits[i]
            if outputs[i] >= num_wires:
                return False
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

    def _baseline_circuit():
        """Independent XOR of inputs for each output."""
        gates = []
        outputs = []
        next_wire = n + 1
        for i in range(m):
            bits = row_bits[i]
            if bits == 0:
                outputs.append(n)
            else:
                indices = [j for j in range(n) if (bits >> j) & 1]
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

    def _build_circuit_from_exprs(exprs, gates_so_far, next_wire):
        """Build the final circuit from expression bitsets and existing gates."""
        outputs = []
        for i in range(m):
            e = exprs[i]
            if e == 0:
                outputs.append(n)
            else:
                wires = []
                temp = e
                w = 0
                while temp:
                    if temp & 1:
                        wires.append(w)
                    temp >>= 1
                    w += 1

                if len(wires) == 1:
                    outputs.append(wires[0])
                elif len(wires) == 0:
                    outputs.append(n)
                else:
                    # XOR all wires together
                    prev = wires[0]
                    for w in wires[1:]:
                        gates_so_far.append([prev, w])
                        prev = next_wire
                        next_wire += 1
                    outputs.append(prev)
        return {'gates': gates_so_far, 'outputs': outputs}

    def _greedy_cse(time_budget, seed):
        """
        Greedy Common Subexpression Elimination.
        Repeatedly find the pair of wires (a, b) that appear together in the most
        expressions, create a new wire g = a XOR b, and replace all occurrences
        of {a, b} with g in the expressions.
        """
        rng = random.Random(seed)
        exprs = list(row_bits)
        gates = []
        next_wire = n + 1
        pool = list(range(n + 1))  # inputs 0..n-1 and zero wire n

        max_iter = 300
        for _ in range(max_iter):
            if time.time() > time_budget:
                break

            # Count co-occurrence of each pair of wires
            co_occ = {}
            for e in exprs:
                if e == 0:
                    continue
                wires_in_e = []
                temp = e
                w = 0
                while temp:
                    if temp & 1:
                        wires_in_e.append(w)
                    temp >>= 1
                    w += 1

                for idx_a in range(len(wires_in_e)):
                    for idx_b in range(idx_a + 1, len(wires_in_e)):
                        a = wires_in_e[idx_a]
                        b = wires_in_e[idx_b]
                        key = (a, b)
                        co_occ[key] = co_occ.get(key, 0) + 1

            if not co_occ:
                break

            # Find the pair with highest co-occurrence count
            best_pair = None
            best_count = 1
            items = list(co_occ.items())
            rng.shuffle(items)
            for (a, b), cnt in items:
                if cnt > best_count:
                    best_count = cnt
                    best_pair = (a, b)

            if best_pair is None:
                break

            a, b = best_pair
            g = next_wire
            next_wire += 1
            gates.append([a, b])
            pool.append(g)

            a_bit = 1 << a
            b_bit = 1 << b
            g_bit = 1 << g

            for i in range(m):
                e = exprs[i]
                if (e & a_bit) and (e & b_bit):
                    e = (e & ~a_bit & ~b_bit) | g_bit
                    exprs[i] = e

        return _build_circuit_from_exprs(exprs, gates, next_wire)

    def _steiner_mst(time_budget, seed):
        """
        Steiner tree / MST-inspired approach.
        Greedily compute target vectors in order of cheapest cost from
        already-computed vectors, using existing intermediates when possible.
        """
        rng = random.Random(seed)

        wire_to_vec = {}
        vec_to_wire = {}

        for i in range(n):
            vec = 1 << i
            wire_to_vec[i] = vec
            vec_to_wire[vec] = i
        wire_to_vec[n] = 0
        vec_to_wire[0] = n

        next_wire = n + 1
        gates = []

        targets = list(set(row_bits))
        targets = [t for t in targets if t != 0]

        if not targets:
            return {'gates': [], 'outputs': [n] * m}

        pending = list(targets)
        pending_set = set(pending)

        max_iter = 500
        for _ in range(max_iter):
            if time.time() > time_budget:
                break
            if not pending:
                break

            best_target = None
            best_cost = float('inf')
            best_indices = None

            # Shuffle pending for randomization
            rng.shuffle(pending)

            for t in pending:
                if t in vec_to_wire:
                    best_target = t
                    best_cost = 0
                    best_indices = None
                    break

                # Try to compute t from two existing vectors
                # t = u XOR w => u = t XOR w
                found = False
                for w_vec, w_wire in wire_to_vec.items():
                    u_vec = t ^ w_vec
                    if u_vec in vec_to_wire:
                        u_wire = vec_to_wire[u_vec]
                        # Cost: 1 gate
                        cost = 1
                        if cost < best_cost:
                            best_cost = cost
                            best_target = t
                            best_indices = (u_wire, w_wire)
                            found = True
                            break

                if found:
                    continue

                # Fallback: direct computation from inputs
                pc = bin(t).count('1')
                cost_direct = max(0, pc - 1)
                if cost_direct < best_cost:
                    best_cost = cost_direct
                    best_target = t
                    best_indices = 'direct'

            if best_target is None:
                break

            t = best_target

            if t in vec_to_wire:
                pending.remove(t)
                pending_set.discard(t)
                continue

            if best_indices == 'direct':
                indices = [j for j in range(n) if (t >> j) & 1]
                if len(indices) == 0:
                    wire_id = n
                elif len(indices) == 1:
                    wire_id = indices[0]
                else:
                    prev = indices[0]
                    for idx in indices[1:]:
                        gates.append([prev, idx])
                        prev = next_wire
                        next_wire += 1
                    wire_id = prev
            else:
                u_wire, w_wire = best_indices
                gates.append([u_wire, w_wire])
                wire_id = next_wire
                next_wire += 1

            wire_to_vec[wire_id] = t
            vec_to_wire[t] = wire_id
            pending.remove(t)
            pending_set.discard(t)

        # Build outputs
        outputs = []
        for i in range(m):
            e = row_bits[i]
            if e == 0:
                outputs.append(n)
            elif e in vec_to_wire:
                outputs.append(vec_to_wire[e])
            else:
                # Fallback: direct
                indices = [j for j in range(n) if (e >> j) & 1]
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

    def _hypercube_mst(time_budget, seed):
        """
        Structural change: Represent the synthesis as a Steiner tree problem
        in the Boolean hypercube. Use Prim's algorithm over the complete graph
        of all reachable vectors (inputs, zero, and targets) to find a minimum
        spanning tree that connects all terminals. Edge weight is 1 if the two
        vectors differ by a single bit (i.e., one XOR gate), otherwise infinity.
        In practice, we use a BFS-like expansion from the zero vector and inputs,
        growing the set of known vectors by XOR-ing pairs, and always picking the
        next vector that is cheapest to produce from already-known vectors.
        """
        rng = random.Random(seed)

        # Known vectors: wire -> vector
        wire_to_vec = {}
        vec_to_wire = {}
        next_wire = n + 1
        gates = []

        for i in range(n):
            vec = 1 << i
            wire_to_vec[i] = vec
            vec_to_wire[vec] = i
        wire_to_vec[n] = 0
        vec_to_wire[0] = n

        # Targets are the unique non-zero row vectors
        targets = list(set(row_bits))
        targets = [t for t in targets if t != 0]

        if not targets:
            return {'gates': [], 'outputs': [n] * m}

        # Prim's algorithm: maintain a set of "tree" vectors (already connected)
        # and repeatedly add the cheapest vector outside the tree that can be
        # connected to the tree in one XOR gate.
        tree_vecs = set(vec_to_wire.keys())  # all currently known vectors
        tree_wires = list(vec_to_wire.values())  # all currently known wires

        # We want to eventually include all targets in the tree.
        # Start with all inputs and zero in the tree.
        # Repeatedly find a vector v not in the tree such that v = a XOR b
        # for some a, b in the tree, and v is a target or leads to a target.
        # To keep it simple, we only consider targets and their XORs with tree vectors.

        pending_targets = set(targets)
        all_targets = set(targets)

        max_iter = 800
        for _ in range(max_iter):
            if time.time() > time_budget:
                break
            if not pending_targets:
                break

            # Find the target that can be produced with the fewest gates
            # from the current tree.
            best_target = None
            best_cost = float('inf')
            best_pair = None

            # Candidate vectors to consider: all targets, and all XORs of
            # targets with tree vectors (these might be cheaper intermediates).
            candidates = set()
            for t in pending_targets:
                candidates.add(t)
                # Also consider t XOR w for w in tree (might be cheaper)
                for w_vec in tree_vecs:
                    candidates.add(t ^ w_vec)

            # For each candidate, compute the cost to produce it from the tree.
            for cand in candidates:
                if cand in tree_vecs:
                    continue
                # Cost = 1 if cand = a XOR b for some a, b in tree
                # Otherwise, cost = popcount(cand) - 1 (direct from inputs)
                found_in_tree = False
                for a_vec in tree_vecs:
                    b_vec = cand ^ a_vec
                    if b_vec in tree_vecs:
                        found_in_tree = True
                        cost = 1
                        best_pair = (vec_to_wire[a_vec], vec_to_wire[b_vec])
                        break
                if not found_in_tree:
                    # Direct cost
                    pc = bin(cand).count('1')
                    cost = max(0, pc - 1)
                    best_pair = 'direct'

                # We only care about candidates that are targets or help produce targets.
                # For simplicity, only consider targets as additions to the tree.
                if cand not in pending_targets:
                    continue

                if cost < best_cost:
                    best_cost = cost
                    best_target = cand
                    best_target_pair = best_pair

            if best_target is None:
                break

            t = best_target

            # Add t to the tree
            if best_target_pair == 'direct':
                indices = [j for j in range(n) if (t >> j) & 1]
                if len(indices) == 0:
                    wire_id = n
                elif len(indices) == 1:
                    wire_id = indices[0]
                else:
                    prev = indices[0]
                    for idx in indices[1:]:
                        gates.append([prev, idx])
                        prev = next_wire
                        next_wire += 1
                    wire_id = prev
            else:
                a_wire, b_wire = best_target_pair
                gates.append([a_wire, b_wire])
                wire_id = next_wire
                next_wire += 1

            wire_to_vec[wire_id] = t
            vec_to_wire[t] = wire_id
            tree_vecs.add(t)
            tree_wires.append(wire_id)
            pending_targets.discard(t)

        # Build outputs
        outputs = []
        for i in range(m):
            e = row_bits[i]
            if e == 0:
                outputs.append(n)
            elif e in vec_to_wire:
                outputs.append(vec_to_wire[e])
            else:
                # Fallback: direct
                indices = [j for j in range(n) if (e >> j) & 1]
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

    def _basis_reduction(time_budget, seed):
        """
        Structural change: Use Gaussian elimination to find a basis for the
        row space, then synthesize each basis vector independently, and
        express each row as a combination of basis vectors. This can reduce
        the number of gates if the rows are linearly dependent.
        """
        rng = random.Random(seed)

        # Find a basis for the row space
        basis = []
        basis_vectors = []
        for i in range(m):
            v = row_bits[i]
            if v == 0:
                continue
            # Reduce v by existing basis vectors
            for b in basis:
                if (v & b) and bin(v & b).count('1') <= bin(v).count('1'):
                    # Try to reduce
                    new_v = v ^ b
                    if bin(new_v).count('1') < bin(v).count('1'):
                        v = new_v
            if v == 0:
                continue
            # Add v to basis
            basis.append(v)
            basis_vectors.append(i)

        # Now each row is a combination of basis vectors
        # For simplicity, just synthesize each row independently
        # This is a placeholder for a more sophisticated approach
        gates = []
        next_wire = n + 1
        outputs = []

        for i in range(m):
            bits = row_bits[i]
            if bits == 0:
                outputs.append(n)
            else:
                indices = [j for j in range(n) if (bits >> j) & 1]
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

    # First, get a baseline
    baseline = _baseline_circuit()
    best_result = baseline
    best_gate_count = len(baseline['gates'])

    # Try multiple strategies with different seeds
    strategies = [
        ('greedy_cse', _greedy_cse),
        ('steiner_mst', _steiner_mst),
        ('hypercube_mst', _hypercube_mst),
    ]

    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for strategy_name, strategy_func in strategies:
        for seed in seeds:
            if time.time() > deadline - 3.0:
                break

            try:
                result = strategy_func(deadline - 3.0, seed)
                if _verify(result):
                    d = _compute_depth(result['gates'], result['outputs'])
                    if d <= depth_limit:
                        gc = len(result['gates'])
                        if gc < best_gate_count:
                            best_gate_count = gc
                            best_result = result
            except Exception:
                pass

    return best_result
