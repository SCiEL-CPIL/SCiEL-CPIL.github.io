import time
import random

DIMENSIONS = [3, 4, 7]
_DEADLINE = 140.0

def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS
    total_terms = n * m * p  # 84

    # Generate all possible 2x2x2 Strassen blocks
    blocks = []
    for i0 in range(n):
        for i1 in range(i0 + 1, n):
            for j0 in range(m):
                for j1 in range(j0 + 1, m):
                    for k0 in range(p):
                        for k1 in range(k0 + 1, p):
                            terms = set()
                            for i in (i0, i1):
                                for j in (j0, j1):
                                    for k in (k0, k1):
                                        terms.add((i, j, k))
                            blocks.append({
                                'i0': i0, 'i1': i1,
                                'j0': j0, 'j1': j1,
                                'k0': k0, 'k1': k1,
                                'terms': terms
                            })

    nb = len(blocks)

    # Precompute conflict masks (bit i set if block i conflicts)
    # and term bitmasks for fast coverage tracking
    term_to_idx = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                term_to_idx[(i, j, k)] = len(term_to_idx)

    block_term_mask = []
    for b in blocks:
        mask = 0
        for t in b['terms']:
            mask |= (1 << term_to_idx[t])
        block_term_mask.append(mask)

    conflict_mask = []
    for i in range(nb):
        cm = 0
        for j in range(nb):
            if i != j and (block_term_mask[i] & block_term_mask[j]) != 0:
                cm |= (1 << j)
        conflict_mask.append(cm)

    # Fixed fallback configuration
    fixed_tuples = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 0, 1, 4, 5),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 2, 3, 4, 5),
    ]
    fixed_indices = []
    for ft in fixed_tuples:
        i0, i1, j0, j1, k0, k1 = ft
        for idx, b in enumerate(blocks):
            if (b['i0'] == i0 and b['i1'] == i1 and b['j0'] == j0 and
                b['j1'] == j1 and b['k0'] == k0 and b['k1'] == k1):
                fixed_indices.append(idx)
                break

    best_rank = 78
    best_config = None

    def evaluate_set(selected):
        covered = 0
        for idx in selected:
            covered |= block_term_mask[idx]
        rank = 7 * len(selected) + (total_terms - bin(covered).count('1'))
        return rank, covered

    def greedy_set_packing(random_seed=0):
        rng = random.Random(random_seed)
        remaining = set(range(nb))
        selected = []
        covered = 0

        while remaining:
            # Filter to blocks disjoint from covered
            candidates = [i for i in remaining if (block_term_mask[i] & covered) == 0]
            if not candidates:
                break

            # Heuristic: pick block that covers most new terms
            # (all candidates are disjoint from covered, so all cover 8 new terms)
            # Tie-break: pick one with fewest conflicts among remaining candidates
            best_pick = None
            best_conflicts = None
            for c in candidates:
                cm = conflict_mask[c]
                # count conflicts among candidates
                conf = 0
                for d in candidates:
                    if d != c and (cm >> d) & 1:
                        conf += 1
                if best_conflicts is None or conf < best_conflicts:
                    best_conflicts = conf
                    best_pick = c
                if conf == 0:
                    break

            selected.append(best_pick)
            covered |= block_term_mask[best_pick]
            remaining.discard(best_pick)

        return selected

    def local_search(selected, rng, max_iters=200):
        """Improve a packing via add/remove/swap moves."""
        selected = list(selected)
        rank, covered = evaluate_set(selected)
        selected_set = set(selected)

        for _ in range(max_iters):
            improved = False

            # Try adding a block
            add_candidates = [i for i in range(nb)
                              if i not in selected_set and (block_term_mask[i] & covered) == 0]
            if add_candidates:
                pick = rng.choice(add_candidates)
                new_selected = selected + [pick]
                new_rank, _ = evaluate_set(new_selected)
                if new_rank < rank:
                    selected = new_selected
                    rank = new_rank
                    covered |= block_term_mask[pick]
                    selected_set.add(pick)
                    improved = True

            if not improved:
                # Try removing a block
                if len(selected) > 1:
                    pick = rng.choice(selected)
                    new_selected = [s for s in selected if s != pick]
                    new_rank, _ = evaluate_set(new_selected)
                    if new_rank < rank:
                        selected = new_selected
                        rank = new_rank
                        selected_set.discard(pick)
                        covered = 0
                        for s in selected:
                            covered |= block_term_mask[s]
                        improved = True

            if not improved:
                # Try swap: remove one, add one
                if len(selected) >= 1:
                    pick = rng.choice(selected)
                    new_covered = 0
                    for s in selected:
                        if s != pick:
                            new_covered |= block_term_mask[s]
                    add_candidates = [i for i in range(nb)
                                      if i not in selected_set and (block_term_mask[i] & new_covered) == 0]
                    if add_candidates:
                        pick_add = rng.choice(add_candidates)
                        new_selected = [s for s in selected if s != pick] + [pick_add]
                        new_rank, _ = evaluate_set(new_selected)
                        if new_rank < rank:
                            selected = new_selected
                            rank = new_rank
                            selected_set.discard(pick)
                            selected_set.add(pick_add)
                            covered = new_covered | block_term_mask[pick_add]
                            improved = True

            if not improved:
                break

        return selected, rank

    # Fixed starting point
    fixed_rank, _ = evaluate_set(fixed_indices)
    if fixed_rank < best_rank:
        best_rank = fixed_rank
        best_config = list(fixed_indices)

    # Greedy with multiple random seeds
    num_restarts = 2000
    for r in range(num_restarts):
        if time.time() - _t0 > _DEADLINE - 15:
            break

        selected = greedy_set_packing(r)
        rank, _ = evaluate_set(selected)

        if rank < best_rank:
            best_rank = rank
            best_config = list(selected)

        # Local search on this and the best so far
        if time.time() - _t0 < _DEADLINE - 20:
            rng = random.Random(r * 1000 + 7)
            improved, improved_rank = local_search(selected, rng, max_iters=100)
            if improved_rank < best_rank:
                best_rank = improved_rank
                best_config = list(improved)

            # Also try local search on best_config
            if best_config is not None:
                rng2 = random.Random(r * 2000 + 13)
                improved2, improved_rank2 = local_search(best_config, rng2, max_iters=50)
                if improved_rank2 < best_rank:
                    best_rank = improved_rank2
                    best_config = list(improved2)

    if best_config is None:
        best_config = list(fixed_indices)
        best_rank, _ = evaluate_set(best_config)

    # Construct matrices
    rank = best_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    covered_terms = set()

    for b_idx in best_config:
        b = blocks[b_idx]
        i0, i1 = b['i0'], b['i1']
        j0, j1 = b['j0'], b['j1']
        k0, k1 = b['k0'], b['k1']

        # M1
        U[i0 * m + j0][col] = 1
        U[i1 * m + j1][col] = 1
        V[j0 * p + k0][col] = 1
        V[j1 * p + k1][col] = 1
        W[k0 * n + i0][col] = 1
        W[k1 * n + i1][col] = 1
        covered_terms.update([(i0, j0, k0), (i0, j0, k1), (i0, j1, k0), (i0, j1, k1),
                              (i1, j0, k0), (i1, j0, k1), (i1, j1, k0), (i1, j1, k1)])
        col += 1

        # M2
        U[i1 * m + j0][col] = 1
        U[i1 * m + j1][col] = 1
        V[j0 * p + k0][col] = 1
        W[k0 * n + i1][col] = 1
        W[k1 * n + i1][col] = -1
        col += 1

        # M3
        U[i0 * m + j0][col] = 1
        V[j0 * p + k1][col] = 1
        V[j1 * p + k1][col] = -1
        W[k1 * n + i0][col] = 1
        W[k1 * n + i1][col] = 1
        col += 1

        # M4
        U[i1 * m + j1][col] = 1
        V[j1 * p + k0][col] = 1
        V[j0 * p + k0][col] = -1
        W[k0 * n + i0][col] = 1
        W[k0 * n + i1][col] = 1
        col += 1

        # M5
        U[i0 * m + j0][col] = 1
        U[i0 * m + j1][col] = 1
        V[j1 * p + k1][col] = 1
        W[k0 * n + i0][col] = -1
        W[k1 * n + i0][col] = 1
        col += 1

        # M6
        U[i1 * m + j0][col] = 1
        U[i0 * m + j0][col] = -1
        V[j0 * p + k0][col] = 1
        V[j0 * p + k1][col] = 1
        W[k1 * n + i1][col] = 1
        col += 1

        # M7
        U[i0 * m + j1][col] = 1
        U[i1 * m + j1][col] = -1
        V[j1 * p + k0][col] = 1
        V[j1 * p + k1][col] = 1
        W[k0 * n + i0][col] = 1
        col += 1

    # Fill remaining terms with trivial rank-1 tensors
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered_terms:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    assert col == rank, f"Rank mismatch: {col} != {rank}"

    # Verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = 0
                for c in range(rank):
                    total += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if total != 1:
                    raise ValueError(f"Decomposition failed at ({i},{j},{k})")

    return {"U": U, "V": V, "W": W}
