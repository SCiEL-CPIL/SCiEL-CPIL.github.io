import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # ---------- Precomputation ----------
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}

    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)

    # For each block, the list of t-subset indices it covers
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)

    # For each t-subset, the list of blocks covering it (as numpy arrays for fast ops)
    ts_to_blocks_np = [np.array(b, dtype=np.int32) for b in ts_to_blocks]

    # Precompute, for every pair of blocks, the set of t-subsets covered by exactly one of them
    # (symmetric difference of their coverage). Two blocks can be merged into one iff
    # the union of their *unique* t-subsets (i.e., the symmetric difference) is a subset
    # of some single block's coverage.
    # For speed, precompute for each block a frozenset of its t-subset indices.
    block_ts_sets = [frozenset(bts) for bts in block_to_ts]

    # Precompute mergeability: mergeable[i][j] = True if blocks i and j can be replaced by one block.
    # This is expensive (num_k_blocks^2), so we compute lazily / on demand but cache.
    # Actually num_k_blocks = C(19,8) = 75582, too big for full matrix. Compute on demand with cache.
    merge_cache = {}
    def can_merge(i, j):
        key = (i, j) if i < j else (j, i)
        if key in merge_cache:
            return merge_cache[key]
        # Unique t-subsets of i (not in j) and j (not in i)
        si = block_ts_sets[i]
        sj = block_ts_sets[j]
        unique = si ^ sj  # symmetric difference = union of unique parts
        # Check if there exists a block covering all of unique
        # Intersect ts_to_blocks for all ts in unique
        if len(unique) == 0:
            res = True
        else:
            it = iter(unique)
            first = next(it)
            candidates = set(ts_to_blocks[first])
            for ts_idx in it:
                candidates.intersection_update(ts_to_blocks[ts_idx])
                if not candidates:
                    break
            res = len(candidates) > 0
        merge_cache[key] = res
        return res

    def C(n, r):
        if r > n or r < 0:
            return 0
        if r == 0 or r == n:
            return 1
        r = min(r, n - r)
        result = 1
        for i in range(1, r + 1):
            result = result * (n - i + 1) // i
        return result

    # ---------- Scarcity-weighted (dual-potential) greedy ----------
    def scarcity_greedy(seed, top_k=5, scarcity_beta=1.0):
        """Greedy where gain = static new-coverage + beta * sum of scarcity weights of new subsets.
        Scarcity weight of a t-subset = 1 / (number of blocks still able to cover it).
        We raise scarcity for the most-constrained uncovered subsets each round."""
        random.seed(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []

        # Scarcity weight: starts at 1.0, increased for constrained subsets
        scarcity = np.ones(num_t_subsets, dtype=np.float64)

        while True:
            if time.time() > deadline - 3.0:
                return None

            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break

            # Compute scarcity-weighted score for top candidates
            # To keep it fast, only compute for a subset of candidates: those with gain in top range
            # Find all with max gain
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break

            # If too many, sample
            if len(candidates) > 200:
                candidates = np.array(random.sample(list(candidates), 200), dtype=np.int64)

            # Score each candidate by sum of scarcity of its newly-covered t-subsets
            best_score = -1.0
            best_b = candidates[0]
            for b_idx in candidates:
                score = 0.0
                for ts_idx in block_to_ts[b_idx]:
                    if is_uncovered[ts_idx]:
                        score += scarcity[ts_idx]
                # Combine: primary = gain, secondary = scarcity score
                combined = gain[b_idx] * 10000.0 + score * scarcity_beta
                if combined > best_score:
                    best_score = combined
                    best_b = b_idx

            sol.append(list(all_k_blocks[best_b]))
            is_selected[best_b] = True
            gain[best_b] = -1

            # Update coverage and scarcity
            for ts_idx in block_to_ts[best_b]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1

            # Raise scarcity for the most-constrained uncovered subsets
            # Find uncovered subsets with lowest remaining degree (fewest blocks can cover them)
            # Approximate: for uncovered subsets, count how many unselected blocks cover them
            # This is expensive; do it sparsely (every 10 steps)
            if len(sol) % 10 == 0 and time.time() < deadline - 5.0:
                # Sample some uncovered subsets
                uncovered_list = np.where(is_uncovered)[0]
                if len(uncovered_list) > 0:
                    sample_size = min(100, len(uncovered_list))
                    sample = random.sample(list(uncovered_list), sample_size)
                    for ts_idx in sample:
                        # Count unselected blocks covering this ts
                        remaining = 0
                        for nb_idx in ts_to_blocks[ts_idx]:
                            if not is_selected[nb_idx]:
                                remaining += 1
                        if remaining > 0:
                            # Scarcity inversely proportional to remaining degree
                            scarcity[ts_idx] = max(scarcity[ts_idx], 1.0 / float(remaining))

        return sol

    # ---------- Iterative removal ----------
    def iterative_removal(blocks):
        if not blocks:
            return blocks
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1
        changed = True
        while changed:
            if time.time() > deadline - 2.0:
                break
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline - 2.0:
                    break
                redundant = True
                for ts in itertools.combinations(blocks[i], t):
                    ts_idx = ts_index_map[ts]
                    if cover_count[ts_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    for ts in itertools.combinations(blocks[i], t):
                        cover_count[ts_index_map[ts]] -= 1
                    blocks.pop(i)
                    changed = True
                    break
        return blocks

    # ---------- Pairwise 2-exchange local search ----------
    def pairwise_2exchange(blocks, max_iters=200):
        """Try to replace two blocks with one block (if mergeable), reducing size by 1."""
        if len(blocks) < 2:
            return blocks, False
        improved = False
        for _ in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            # Randomly pick a pair
            i, j = random.sample(range(len(blocks)), 2)
            # Map to block indices
            bi = all_k_blocks.index(blocks[i])
            bj = all_k_blocks.index(blocks[j])
            if can_merge(bi, bj):
                # Find a block that covers all unique t-subsets
                si = block_ts_sets[bi]
                sj = block_ts_sets[bj]
                unique = si ^ sj
                # Find a candidate block covering all unique
                it = iter(unique)
                first = next(it)
                candidates = list(ts_to_blocks[first])
                for ts_idx in it:
                    candidates = [c for c in candidates if ts_idx in block_ts_sets[c]]
                    if not candidates:
                        break
                if candidates:
                    # Pick one not already in solution
                    sol_tuples = set(tuple(b) for b in blocks)
                    valid = [c for c in candidates if tuple(all_k_blocks[c]) not in sol_tuples]
                    if valid:
                        new_idx = random.choice(valid)
                        new_block = list(all_k_blocks[new_idx])
                        # Replace blocks[i] with new_block, remove blocks[j]
                        blocks[i] = new_block
                        blocks.pop(j)
                        improved = True
                        break
        return blocks, improved

    # ---------- Single-swap local search (fallback) ----------
    def local_search_swap(blocks, max_iters=100):
        if not blocks:
            return blocks, False
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1
        sol_tuples = set(tuple(b) for b in blocks)
        improved = False
        for _ in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            i = random.randint(0, len(blocks) - 1)
            unique_ts = []
            for ts in itertools.combinations(blocks[i], t):
                ts_idx = ts_index_map[ts]
                if cover_count[ts_idx] == 1:
                    unique_ts.append(ts_idx)
            if not unique_ts:
                for ts in itertools.combinations(blocks[i], t):
                    cover_count[ts_index_map[ts]] -= 1
                blocks.pop(i)
                improved = True
                break
            candidates = set(ts_to_blocks[unique_ts[0]])
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(ts_to_blocks[ts_idx])
                if not candidates:
                    break
            if not candidates:
                continue
            bi = all_k_blocks.index(blocks[i])
            valid = [c for c in candidates if c != bi and tuple(all_k_blocks[c]) not in sol_tuples]
            if not valid:
                continue
            new_idx = random.choice(valid)
            new_block = list(all_k_blocks[new_idx])
            for ts in itertools.combinations(blocks[i], t):
                cover_count[ts_index_map[ts]] -= 1
            for ts in itertools.combinations(new_block, t):
                cover_count[ts_index_map[ts]] += 1
            blocks[i] = new_block
            sol_tuples.discard(tuple(blocks[i]))
            sol_tuples.add(tuple(new_block))
            improved = True
            break
        return blocks, improved

    # ---------- Main search ----------
    best_solution = None
    best_len = float('inf')

    # Phase 1: Scarcity-weighted greedy constructions
    num_constructions = 25
    for i in range(num_constructions):
        if time.time() > deadline - 15.0:
            break
        sol = scarcity_greedy(i, top_k=random.choice([5, 10]), scarcity_beta=random.choice([0.5, 1.0, 2.0]))
        if sol is not None:
            sol = iterative_removal(sol)
            if len(sol) < best_len:
                best_len = len(sol)
                best_solution = sol

    if best_solution is None:
        # Fallback: deterministic greedy
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []
        while True:
            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break
            b_idx = int(np.argmax(gain))
            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1
            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1
        best_solution = sol
        best_len = len(sol)

    # Phase 2: Pairwise 2-exchange local search
    for _ in range(30):
        if time.time() > deadline - 5.0:
            break
        best_solution, improved = pairwise_2exchange(best_solution, max_iters=100)
        if not improved:
            break
        best_solution = iterative_removal(best_solution)
        if len(best_solution) < best_len:
            best_len = len(best_solution)

    # Phase 3: Single-swap local search
    for _ in range(30):
        if time.time() > deadline - 5.0:
            break
        best_solution, improved = local_search_swap(best_solution, max_iters=100)
        if not improved:
            break
        best_solution = iterative_removal(best_solution)
        if len(best_solution) < best_len:
            best_len = len(best_solution)

    # Phase 4: More aggressive iterative removal
    for _ in range(10):
        if time.time() > deadline - 2.0:
            break
        new_sol = iterative_removal(list(best_solution))
        if len(new_sol) < len(best_solution):
            best_solution = new_sol
            best_len = len(best_solution)
        else:
            break

    return best_solution
