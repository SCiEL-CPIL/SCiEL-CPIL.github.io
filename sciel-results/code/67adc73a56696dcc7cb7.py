import itertools
import random
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    start = time.time()
    deadline = start + 118
    rng = random.Random(987654321)

    all_points = list(range(v))
    all_blocks = list(itertools.combinations(all_points, k))
    n_blocks = len(all_blocks)
    t_subsets = list(itertools.combinations(all_points, t))
    n_subsets = len(t_subsets)
    sub_idx = {s: i for i, s in enumerate(t_subsets)}

    # Build incidence: for each block, list of t-subsets it covers
    block_subsets = []
    for blk in all_blocks:
        subs = [sub_idx[sub] for sub in itertools.combinations(blk, t)]
        block_subsets.append(subs)

    # Also: for each t-subset, list of blocks covering it
    subset_blocks = [[] for _ in range(n_subsets)]
    for bi, subs in enumerate(block_subsets):
        for si in subs:
            subset_blocks[si].append(bi)

    def time_left():
        return deadline - time.time()

    def build_pool(size, rng):
        """Build a pool of blocks: greedy coverage + random + high-degree."""
        pool = set()
        # Greedy coverage to get a good starting set
        uncovered = set(range(n_subsets))
        greedy_set = []
        while uncovered and len(greedy_set) < size:
            best_bi = -1
            best_gain = 0
            for bi in range(n_blocks):
                if bi in pool:
                    continue
                gain = len(set(block_subsets[bi]) & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
                    if best_gain == len(uncovered):
                        break
            if best_bi < 0:
                break
            pool.add(best_bi)
            greedy_set.append(best_bi)
            uncovered -= set(block_subsets[best_bi])
        # Add random blocks
        while len(pool) < size and time_left() > 5:
            bi = rng.randrange(n_blocks)
            pool.add(bi)
        # Add high-degree blocks (cover many subsets)
        if len(pool) < size:
            degree = [len(block_subsets[bi]) for bi in range(n_blocks)]
            top_blocks = sorted(range(n_blocks), key=lambda x: -degree[x])
            for bi in top_blocks:
                if len(pool) >= size:
                    break
                pool.add(bi)
        return list(pool)

    def ilp_solve(pool_indices, time_budget):
        """Solve ILP: min sum x_i s.t. for each subset, sum_{i covers si} x_i >= 1, x_i in {0,1}"""
        m = len(pool_indices)
        if m == 0:
            return [], 0
        if time_budget < 2:
            return pool_indices, m  # fallback: use all

        # Build constraint matrix A (n_subsets x m)
        # A[si, j] = 1 if pool_indices[j] covers t-subset si
        # Use sparse-like approach with numpy
        A = np.zeros((n_subsets, m), dtype=np.int8)
        for j, bi in enumerate(pool_indices):
            for si in block_subsets[bi]:
                A[si, j] = 1

        c = np.ones(m, dtype=float)  # minimize sum x_i
        # Constraints: A @ x >= 1 for all subsets
        # milp uses: lb <= A_ub @ x <= ub
        # So: -A @ x <= -1  =>  A_ub = -A, ub = -1, lb = -inf
        # Or: A @ x >= 1  =>  use LinearConstraint with lb=1, ub=inf
        constraints = LinearConstraint(A, lb=1, ub=np.inf)
        integrality = np.ones(m, dtype=int)  # all integer
        bounds = Bounds(lb=0, ub=1)

        try:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            if res.status == 0:  # optimal
                x = res.x
                chosen = [pool_indices[j] for j in range(m) if x[j] > 0.5]
                return chosen, len(chosen)
            elif res.status == 1:  # infeasible
                return pool_indices, m
            else:
                # Try with relaxed bounds or fallback
                return pool_indices, m
        except Exception:
            return pool_indices, m

    def verify_and_fix(chosen, time_budget):
        """Ensure all subsets covered; add blocks if needed."""
        chosen_set = set(chosen)
        counts = np.zeros(n_subsets, dtype=np.int32)
        for bi in chosen:
            for si in block_subsets[bi]:
                counts[si] += 1
        uncovered = np.where(counts == 0)[0]
        while len(uncovered) > 0 and time_left() > 3:
            # Greedy add
            best_bi = -1
            best_gain = 0
            unc_set = set(uncovered.tolist())
            for bi in range(n_blocks):
                if bi in chosen_set:
                    continue
                gain = len(set(block_subsets[bi]) & unc_set)
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
            if best_bi < 0:
                break
            chosen.append(best_bi)
            chosen_set.add(best_bi)
            for si in block_subsets[best_bi]:
                counts[si] += 1
            uncovered = np.where(counts == 0)[0]
        return chosen

    def local_search(chosen, time_budget):
        """1-out/1-in and 2-out/1-in swaps using incidence."""
        chosen = list(chosen)
        chosen_set = set(chosen)
        counts = np.zeros(n_subsets, dtype=np.int32)
        for bi in chosen:
            for si in block_subsets[bi]:
                counts[si] += 1

        def can_remove(bi):
            for si in block_subsets[bi]:
                if counts[si] == 1:
                    return False
            return True

        def remove(bi):
            for si in block_subsets[bi]:
                counts[si] -= 1
            chosen_set.discard(bi)

        def add(bi):
            for si in block_subsets[bi]:
                counts[si] += 1
            chosen_set.add(bi)

        improved = True
        while improved and time_left() > 5:
            improved = False
            # Try 1-out/1-in
            for i in range(len(chosen)):
                if time_left() < 5:
                    break
                bi_out = chosen[i]
                if not can_remove(bi_out):
                    continue
                # Find best replacement
                best_bi = -1
                best_score = 0
                mask_out = set(block_subsets[bi_out])
                for bi_in in range(n_blocks):
                    if bi_in in chosen_set:
                        continue
                    # Score: how many of bi_out's subsets does bi_in cover?
                    overlap = len(set(block_subsets[bi_in]) & mask_out)
                    if overlap > best_score:
                        best_score = overlap
                        best_bi = bi_in
                if best_bi >= 0 and best_score > 0:
                    remove(bi_out)
                    chosen[i] = best_bi
                    add(best_bi)
                    improved = True
                    # Try to remove more
                    j = 0
                    while j < len(chosen) and time_left() > 5:
                        if can_remove(chosen[j]):
                            remove(chosen[j])
                            chosen.pop(j)
                            improved = True
                        else:
                            j += 1
                    break  # restart loop
            # Try 2-out/1-in
            for i in range(len(chosen)):
                if time_left() < 5:
                    break
                for j in range(i + 1, len(chosen)):
                    if time_left() < 5:
                        break
                    bi1, bi2 = chosen[i], chosen[j]
                    if not can_remove(bi1) or not can_remove(bi2):
                        continue
                    mask_both = set(block_subsets[bi1]) | set(block_subsets[bi2])
                    best_bi = -1
                    best_score = 0
                    for bi_in in range(n_blocks):
                        if bi_in in chosen_set:
                            continue
                        overlap = len(set(block_subsets[bi_in]) & mask_both)
                        if overlap > best_score:
                            best_score = overlap
                            best_bi = bi_in
                    if best_bi >= 0 and best_score >= len(mask_both):
                        remove(bi1)
                        remove(bi2)
                        chosen.pop(j)
                        chosen.pop(i)
                        add(best_bi)
                        chosen.append(best_bi)
                        improved = True
                        break
                if improved:
                    break
        # Final prune
        changed = True
        while changed and time_left() > 3:
            changed = False
            for i in range(len(chosen)):
                if can_remove(chosen[i]):
                    remove(chosen[i])
                    chosen.pop(i)
                    changed = True
                    break
        return chosen

    # Main: try multiple pool sizes and ILP
    best_blocks = None
    best_size = float('inf')

    pool_sizes = [100, 150, 200, 300, 500]
    for ps in pool_sizes:
        if time_left() < 10:
            break
        pool = build_pool(ps, rng)
        chosen, sz = ilp_solve(pool, time_budget=min(30, time_left() - 10))
        if chosen is None:
            continue
        chosen = verify_and_fix(chosen, time_budget=5)
        if time_left() > 10:
            chosen = local_search(chosen, time_budget=min(20, time_left() - 5))
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_blocks = chosen[:]

    # If ILP didn't work well, fallback to pure greedy
    if best_blocks is None or time_left() < 5:
        # Quick greedy
        uncovered = set(range(n_subsets))
        chosen = []
        chosen_set = set()
        while uncovered:
            best_bi = -1
            best_gain = 0
            for bi in range(n_blocks):
                if bi in chosen_set:
                    continue
                gain = len(set(block_subsets[bi]) & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
            if best_bi < 0:
                break
            chosen.append(best_bi)
            chosen_set.add(best_bi)
            uncovered -= set(block_subsets[best_bi])
        best_blocks = chosen

    return [list(all_blocks[bi]) for bi in best_blocks]
