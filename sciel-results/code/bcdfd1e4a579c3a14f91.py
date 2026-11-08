import time
import random

DIMENSIONS = [3, 4, 7]
_DEADLINE = 140.0


def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS

    # Precompute all possible 2x2x2 Strassen blocks
    # Each block covers 8 terms in a 2x2x2 sub-cube
    all_blocks = []
    for i0 in range(n):
        for i1 in range(i0 + 1, n):
            for j0 in range(m):
                for j1 in range(j0 + 1, m):
                    for k0 in range(p):
                        for k1 in range(k0 + 1, p):
                            terms = frozenset(
                                (i, j, k)
                                for i in (i0, i1)
                                for j in (j0, j1)
                                for k in (k0, k1)
                            )
                            all_blocks.append({
                                'i0': i0, 'i1': i1,
                                'j0': j0, 'j1': j1,
                                'k0': k0, 'k1': k1,
                                'terms': terms,
                            })

    # The known-good fixed set from the current program (rank 78)
    fixed_block_params = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 0, 1, 4, 5),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 2, 3, 4, 5),
    ]

    # Map parameter tuple to block index for quick lookup
    def block_key(b):
        return (b['i0'], b['i1'], b['j0'], b['j1'], b['k0'], b['k1'])

    key_to_block = {block_key(b): b for b in all_blocks}

    def evaluate_packing(selected_blocks):
        """Compute the rank for a set of disjoint blocks."""
        covered = set()
        for b in selected_blocks:
            covered.update(b['terms'])
        rank = 7 * len(selected_blocks) + (n * m * p - len(covered))
        return rank, covered

    def is_disjoint(b, covered):
        """Check if block b is disjoint from covered set."""
        for t in b['terms']:
            if t in covered:
                return False
        return True

    def greedy_pack(rng, initial_covered=None):
        """Greedy set packing with randomization. Returns list of selected blocks."""
        covered = set(initial_covered) if initial_covered else set()
        selected = []
        # Start with all blocks that are disjoint from initial covered
        candidates = [b for b in all_blocks if is_disjoint(b, covered)]

        while candidates:
            # Heuristic: pick a random candidate, or one with fewest conflicts
            # For diversity, pick randomly from the first ~50% of candidates
            pool = candidates[:max(1, len(candidates) // 2)]
            b = pool[rng.randrange(len(pool))]

            selected.append(b)
            covered.update(b['terms'])

            # Filter candidates to those still disjoint
            candidates = [c for c in candidates if c is not b and is_disjoint(c, covered)]

        return selected

    # Search for best packing
    best_rank = 78
    best_blocks = None

    # First, evaluate the fixed set
    fixed_blocks = [key_to_block[t] for t in fixed_block_params]
    fixed_rank, fixed_covered = evaluate_packing(fixed_blocks)
    if fixed_rank < best_rank:
        best_rank = fixed_rank
        best_blocks = fixed_blocks

    # Now run greedy search with multiple restarts
    rng = random.Random(42)
    num_restarts = 0
    while time.time() - _t0 < _DEADLINE - 20:
        num_restarts += 1
        selected = greedy_pack(rng)
        rank, covered = evaluate_packing(selected)
        if rank < best_rank:
            best_rank = rank
            best_blocks = selected

        # Occasionally try to improve: remove a random block and re-greedy
        if best_blocks and len(best_blocks) > 0 and num_restarts % 5 == 0:
            # Try removing each block one at a time and re-packing the rest
            for idx in range(len(best_blocks)):
                if time.time() - _t0 > _DEADLINE - 20:
                    break
                remaining = [b for i, b in enumerate(best_blocks) if i != idx]
                covered_rest = set()
                for b in remaining:
                    covered_rest.update(b['terms'])
                new_selected = list(remaining)
                # Re-greedy on top of remaining
                extra = greedy_pack(rng, initial_covered=covered_rest)
                new_selected.extend(extra)
                new_rank, new_covered = evaluate_packing(new_selected)
                if new_rank < best_rank:
                    best_rank = new_rank
                    best_blocks = new_selected

    # Fallback to fixed set if search didn't find anything better
    if best_blocks is None:
        best_blocks = fixed_blocks
        best_rank = fixed_rank

    # Build the factor matrices
    rank = best_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for b in best_blocks:
        i0, i1 = b['i0'], b['i1']
        j0, j1 = b['j0'], b['j1']
        k0, k1 = b['k0'], b['k1']

        # M1: (i0+j0)*(j0+k0) -> W[k0,n+i0] and (i1+j1)*(j1+k1) -> W[k1,n+i1]
        U[i0 * m + j0][col] = 1
        U[i1 * m + j1][col] = 1
        V[j0 * p + k0][col] = 1
        V[j1 * p + k1][col] = 1
        W[k0 * n + i0][col] = 1
        W[k1 * n + i1][col] = 1

        # M2: (i1+j0)*(j0+k0) -> W[k0,n+i1]; (i1+j1)*(j1+k1) -> -W[k1,n+i1]
        U[i1 * m + j0][col + 1] = 1
        U[i1 * m + j1][col + 1] = 1
        V[j0 * p + k0][col + 1] = 1
        W[k0 * n + i1][col + 1] = 1
        W[k1 * n + i1][col + 1] = -1

        # M3: (i0+j0)*(j0+k1 - j1+k1) -> W[k1,n+i0] and W[k1,n+i1]
        U[i0 * m + j0][col + 2] = 1
        V[j0 * p + k1][col + 2] = 1
        V[j1 * p + k1][col + 2] = -1
        W[k1 * n + i0][col + 2] = 1
        W[k1 * n + i1][col + 2] = 1

        # M4: (i1+j1)*(j1+k0 - j0+k0) -> W[k0,n+i0] and W[k0,n+i1]
        U[i1 * m + j1][col + 3] = 1
        V[j1 * p + k0][col + 3] = 1
        V[j0 * p + k0][col + 3] = -1
        W[k0 * n + i0][col + 3] = 1
        W[k0 * n + i1][col + 3] = 1

        # M5: (i0+j0 + i0+j1)*(j1+k1) -> -W[k0,n+i0] and W[k1,n+i0]
        U[i0 * m + j0][col + 4] = 1
        U[i0 * m + j1][col + 4] = 1
        V[j1 * p + k1][col + 4] = 1
        W[k0 * n + i0][col + 4] = -1
        W[k1 * n + i0][col + 4] = 1

        # M6: (i1+j0 - i0+j0)*(j0+k0 + j0+k1) -> W[k1,n+i1]
        U[i1 * m + j0][col + 5] = 1
        U[i0 * m + j0][col + 5] = -1
        V[j0 * p + k0][col + 5] = 1
        V[j0 * p + k1][col + 5] = 1
        W[k1 * n + i1][col + 5] = 1

        # M7: (i0+j1 - i1+j1)*(j1+k0 + j1+k1) -> W[k0,n+i0]
        U[i0 * m + j1][col + 6] = 1
        U[i1 * m + j1][col + 6] = -1
        V[j1 * p + k0][col + 6] = 1
        V[j1 * p + k1][col + 6] = 1
        W[k0 * n + i0][col + 6] = 1

        col += 7

    # Fill in uncovered terms with trivial rank-1 factors
    covered = set()
    for b in best_blocks:
        covered.update(b['terms'])

    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                if (i0, j0, k0) in covered:
                    continue
                U[i0 * m + j0][col] = 1
                V[j0 * p + k0][col] = 1
                W[k0 * n + i0][col] = 1
                col += 1

    assert col == rank, f"Column count {col} != rank {rank}"

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
