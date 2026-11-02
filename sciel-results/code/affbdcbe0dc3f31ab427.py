import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 142.0  # Safety margin for 150s limit

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

    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)

    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)

    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}

    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]

    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)

    # Convert to numpy arrays
    block_to_ts_np = [np.array(lst, dtype=np.int32) for lst in block_to_ts]
    ts_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in ts_to_blocks]

    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}

    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    def run_lookahead_greedy(seed, num_simulations=20, look_depth=3):
        """Random greedy with Monte Carlo lookahead scoring."""
        random.seed(seed)
        np_rng = np.random.RandomState(seed)
        
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        sol = []

        while True:
            if time.time() > deadline - 8.0:
                return None

            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break

            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break

            # If few candidates, pick the best by lookahead
            if len(candidates) <= 10:
                # Score each candidate via Monte Carlo lookahead
                best_score = -1
                best_candidate = candidates[0]
                for c in candidates:
                    score = _score_candidate(c, is_uncovered, gain, is_selected, 
                                            num_simulations, look_depth, np_rng)
                    if score > best_score:
                        best_score = score
                        best_candidate = c
                b_idx = int(best_candidate)
            else:
                # For many candidates, use a hybrid: pick top-k by gain,
                # then score those with lookahead
                # Take top 30 by gain (all have same gain, so random subset)
                if len(candidates) > 30:
                    top_cands = np_rng.choice(candidates, size=30, replace=False)
                else:
                    top_cands = candidates

                best_score = -1
                best_candidate = top_cands[0]
                for c in top_cands:
                    score = _score_candidate(c, is_uncovered, gain, is_selected,
                                            num_simulations, look_depth, np_rng)
                    if score > best_score:
                        best_score = score
                        best_candidate = c
                b_idx = int(best_candidate)

            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1

            # Update coverage
            bts = block_to_ts_np[b_idx]
            newly_covered_mask = is_uncovered[bts]
            newly_covered = bts[newly_covered_mask]

            if len(newly_covered) == 0:
                continue

            is_uncovered[newly_covered] = False

            # Vectorized gain update
            blocks_to_decrement = []
            for ts_idx in newly_covered:
                blocks_to_decrement.extend(ts_to_blocks[ts_idx])

            if blocks_to_decrement:
                counts = np.bincount(blocks_to_decrement, minlength=num_k_blocks)
                gain -= counts
                gain[is_selected] = -1000

        return sol

    def _score_candidate(block_idx, is_uncovered, gain, is_selected,
                         num_sims, look_depth, np_rng):
        """Score a candidate block by simulating its effect on future coverage."""
        bts = block_to_ts_np[block_idx]
        newly_covered_mask = is_uncovered[bts]
        num_newly_covered = int(np.sum(newly_covered_mask))

        if num_newly_covered == 0:
            return 0

        # Simulate: after selecting this block, how much more can we cover?
        # Monte Carlo: randomly select look_depth more blocks (weighted by gain),
        # count how many additional t-subsets get covered.
        
        # Work on a copy of relevant state
        sim_is_uncovered = is_uncovered.copy()
        sim_gain = gain.copy()
        
        # Apply this block's effect
        sim_is_uncovered[bts] = False
        sim_gain[block_idx] = -1
        
        # Update sim_gain for other blocks that share t-subsets
        for ts_idx in bts[newly_covered_mask]:
            for other_block in ts_to_blocks[ts_idx]:
                if other_block != block_idx and not is_selected[other_block]:
                    sim_gain[other_block] -= 1
        
        total_additional = 0
        for _ in range(num_sims):
            sim_covered = 0
            sim_uncovered = sim_is_uncovered.copy()
            sim_g = sim_gain.copy()
            for _ in range(look_depth):
                max_g = int(np.max(sim_g))
                if max_g <= 0:
                    break
                cands = np.where((sim_g == max_g) & (sim_g > 0))[0]
                if len(cands) == 0:
                    break
                chosen = cands[np_rng.randint(0, len(cands))]
                chosen_bts = block_to_ts_np[chosen]
                new_mask = sim_uncovered[chosen_bts]
                sim_covered += int(np.sum(new_mask))
                sim_uncovered[chosen_bts] = False
                sim_g[chosen] = -1
                # Quick update of gains for next step
                for ts_idx in chosen_bts[new_mask]:
                    for ob in ts_to_blocks[ts_idx]:
                        if sim_g[ob] > 0:
                            sim_g[ob] -= 1
            total_additional += sim_covered

        expected_additional = total_additional / num_sims
        # Score = immediate coverage + discounted future coverage
        return num_newly_covered + 0.3 * expected_additional

    def optimize_redundancy(blocks):
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)

        block_ts_indices = []
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                block_ts_indices.append(block_to_ts[idx])
            else:
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(ts_index_map[ts])
                block_ts_indices.append(ts_list)

        for ts_list in block_ts_indices:
            for ts_idx in ts_list:
                cover_count[ts_idx] += 1

        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 3.0:
                    return blocks

                redundant = True
                for ts_idx in block_ts_indices[i]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break

                if redundant:
                    blocks.pop(i)
                    for ts_idx in block_ts_indices[i]:
                        cover_count[ts_idx] -= 1
                    block_ts_indices.pop(i)
                    changed = True
                    break

        return blocks

    def local_search_swap(blocks):
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)

        block_ts_indices = []
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                block_ts_indices.append(block_to_ts[idx])
            else:
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(ts_index_map[ts])
                block_ts_indices.append(ts_list)

        for ts_list in block_ts_indices:
            for ts_idx in ts_list:
                cover_count[ts_idx] += 1

        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)

        improved = False
        for i in range(len(blocks)):
            if time.time() > deadline - 3.0:
                return blocks, improved

            unique_ts = []
            for ts_idx in block_ts_indices[i]:
                if cover_count[ts_idx] == 1:
                    unique_ts.append(ts_idx)

            if not unique_ts:
                continue

            candidates = set(ts_to_blocks[unique_ts[0]])
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(ts_to_blocks[ts_idx])
                if not candidates:
                    break

            if not candidates:
                continue

            valid_candidates = [c for c in candidates if c not in sol_block_indices]
            if not valid_candidates:
                continue

            new_block_idx = random.choice(valid_candidates)
            new_block = list(all_k_blocks[new_block_idx])

            old_block = blocks[i]
            old_block_idx = get_block_idx(old_block)

            blocks[i] = new_block
            sol_block_indices.discard(old_block_idx)
            sol_block_indices.add(new_block_idx)

            for ts_idx in block_ts_indices[i]:
                cover_count[ts_idx] -= 1
            for ts_idx in block_to_ts[new_block_idx]:
                cover_count[ts_idx] += 1
            block_ts_indices[i] = block_to_ts[new_block_idx]

            improved = True

        return blocks, improved

    # Run multiple lookahead greedy constructions with different seeds
    best_solution = None
    best_len = float('inf')

    # First run: deterministic-ish (seed 0)
    sol = run_lookahead_greedy(0, num_simulations=15, look_depth=3)
    if sol is not None:
        best_solution = sol
        best_len = len(sol)

    # More runs with varied parameters
    num_runs = 40
    for i in range(1, num_runs):
        if time.time() > deadline - 15.0:
            break

        # Vary simulation depth and count
        n_sims = 10 + (i % 5) * 5  # 10, 15, 20, 25, 30
        depth = 2 + (i % 3)  # 2, 3, 4

        sol = run_lookahead_greedy(i, num_simulations=n_sims, look_depth=depth)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol

    if best_solution is None:
        # Fallback: simple greedy
        best_solution = run_lookahead_greedy(0, num_simulations=1, look_depth=1)
        if best_solution is None:
            best_solution = [list(all_k_blocks[0])]

    # Local search refinement
    for _ in range(50):
        if time.time() > deadline - 5.0:
            break
        best_solution, improved = local_search_swap(best_solution)
        if improved:
            best_solution = optimize_redundancy(best_solution)
            if len(best_solution) < best_len:
                best_len = len(best_solution)
        else:
            break

    return best_solution
