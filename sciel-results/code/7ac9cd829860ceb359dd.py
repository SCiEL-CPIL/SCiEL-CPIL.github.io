"""XOR-circuit synthesis: basis reduction + cost-aware greedy with sharing."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 85

    mat = np.array(matrix, dtype=np.int32)
    n_rows, n_cols = mat.shape
    n = n_cols
    zero_wire = n

    row_masks = []
    for i in range(n_rows):
        m = 0
        for j in range(n):
            if mat[i, j]:
                m |= (1 << j)
        row_masks.append(m)

    # Gaussian elimination to get a basis
    basis_by_pivot = {}
    for rm in row_masks:
        m = rm
        while m:
            pc = m.bit_length() - 1
            if pc in basis_by_pivot:
                m ^= basis_by_pivot[pc]
            else:
                break
        if m:
            pc = m.bit_length() - 1
            basis_by_pivot[pc] = m
    basis_masks = list(basis_by_pivot.values())

    # Reduce basis vectors to minimize popcount
    def reduce_basis(bm_list):
        changed = True
        while changed:
            changed = False
            for i in range(len(bm_list)):
                orig = bm_list[i]
                for j in range(len(bm_list)):
                    if i == j:
                        continue
                    candidate = orig ^ bm_list[j]
                    if candidate and candidate.bit_count() < orig.bit_count():
                        bm_list[i] = candidate
                        changed = True
                        break
        return bm_list

    reduced_basis = reduce_basis(list(basis_masks))

    # All masks we need to compute
    all_needed = set()
    for bm in reduced_basis:
        if bm:
            all_needed.add(bm)
    for rm in row_masks:
        if rm:
            all_needed.add(rm)

    gates = []
    state = {'next_wire': n + 1}
    wire_cache = {}
    for i in range(n):
        wire_cache[1 << i] = i
    wire_cache[0] = zero_wire

    def build_tree(mask):
        """Recursively build XOR tree for mask, sharing via wire_cache."""
        if mask in wire_cache:
            return wire_cache[mask]
        bits = [i for i in range(n) if (mask >> i) & 1]
        k = len(bits)
        if k == 1:
            wire_cache[mask] = bits[0]
            return bits[0]
        mid = k // 2
        m1 = 0
        m2 = 0
        for i in range(mid):
            m1 |= (1 << bits[i])
        for i in range(mid, k):
            m2 |= (1 << bits[i])
        w1 = build_tree(m1)
        w2 = build_tree(m2)
        if w1 == w2:
            wire_cache[mask] = zero_wire
            return zero_wire
        gates.append([w1, w2])
        wire_cache[mask] = state['next_wire']
        state['next_wire'] += 1
        return wire_cache[mask]

    # Cost-aware greedy: repeatedly pick the needed mask that can be completed with 1 gate
    # (XOR of two available wires), else pick the one with lowest estimated cost
    remaining = list(all_needed)
    remaining.sort(key=lambda m: m.bit_count())

    while remaining:
        if time.time() > DEADLINE:
            break
        best_idx = -1
        best_cost = float('inf')
        best_split = None
        for idx, m in enumerate(remaining):
            if m in wire_cache:
                continue
            bits = [i for i in range(n) if (m >> i) & 1]
            k = len(bits)
            if k <= 1:
                continue
            # Check for 1-gate completion
            if k <= 12:
                found = False
                for sub in range(1, (1 << k) - 1):
                    if sub == (1 << k) - 1:
                        continue
                    m1 = 0
                    m2 = 0
                    for i in range(k):
                        if (sub >> i) & 1:
                            m1 |= (1 << bits[i])
                        else:
                            m2 |= (1 << bits[i])
                    if m1 in wire_cache and m2 in wire_cache:
                        best_idx = idx
                        best_cost = 1
                        best_split = (m1, m2)
                        found = True
                        break
                if found:
                    break
            # Estimate cost
            est = max(0, k - 1)
            if est < best_cost:
                best_cost = est
                best_idx = idx
                best_split = None

        if best_idx == -1:
            break

        m = remaining.pop(best_idx)
        if best_split is not None:
            m1, m2 = best_split
            w1 = wire_cache[m1]
            w2 = wire_cache[m2]
            if w1 == w2:
                wire_cache[m] = zero_wire
            else:
                gates.append([w1, w2])
                wire_cache[m] = state['next_wire']
                state['next_wire'] += 1
        else:
            build_tree(m)

    # Build outputs
    outputs = []
    for rm in row_masks:
        if rm == 0:
            outputs.append(zero_wire)
        elif rm in wire_cache:
            outputs.append(wire_cache[rm])
        else:
            outputs.append(build_tree(rm))

    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
