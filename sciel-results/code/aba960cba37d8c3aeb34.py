import time
import random
import itertools
import numpy as np
from functools import reduce

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135.0

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

    block_to_ts_np = [np.array(lst, dtype=np.int32) for lst in block_to_ts]
    ts_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in ts_to_blocks]

    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}

    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    # Bitset representation for fast intersection
    # Each block's t-subset set as a frozenset of indices
    block_ts_sets = [frozenset(lst) for lst in block_to_ts]

    def run_greedy(seed, explore_prob=0.0):
        random.seed(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []

        while True:
            if time.time() > deadline - 5.0:
                return None

            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break

            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break

            if explore_prob > 0 and random.random() < explore_prob:
                explore_cands = np.where((gain >= max_gain - 3) & (~is_selected))[0]
                if len(explore_cands) > 0:
                    b_idx = int(explore_cands[random.randint(0, len(explore_cands) - 1)])
                else:
                    b_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            else:
                b_idx = int(candidates[random.randint(0, len(candidates) - 1)])

            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1

            bts = block_to_ts_np[b_idx]
            newly_covered_mask = is_uncovered[bts]
            newly_covered = bts[newly_covered_mask]

            if len(newly_covered) == 0:
                continue

            is_uncovered[newly_covered] = False

            blocks_to_decrement = []
            for ts_idx in newly_covered:
                blocks_to_decrement.extend(ts_to_blocks[ts_idx])

            if blocks_to_decrement:
                counts = np.bincount(blocks_to_decrement, minlength=num_k_blocks)
                gain -= counts
                gain[is_selected] = -1000

        return sol

    best_solution = run_greedy(0)
    if best_solution is None:
        best_solution = [list(all_k_blocks[0])]
    best_len = len(best_solution)

    num_runs = 100
    for i in range(1, num_runs):
        if time.time() > deadline - 40.0:
            break
        explore = 0.0 if i < 30 else (0.1 if i < 60 else 0.2)
        sol = run_greedy(i, explore_prob=explore)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol

    def build_cover_data(blocks):
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        block_ts_indices = []
        sol_block_set = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                block_ts_indices.append(block_to_ts_np[idx])
                sol_block_set.add(idx)
            else:
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(ts_index_map[ts])
                arr = np.array(ts_list, dtype=np.int32)
                block_ts_indices.append(arr)
            np.add.at(cover_count, block_ts_indices[-1], 1)
        return cover_count, block_ts_indices, sol_block_set

    def fast_local_search(blocks, max_iters=3000):
        n_blocks = len(blocks)
        if n_blocks == 0:
            return blocks, 0

        cover_count, block_ts_indices, sol_block_set = build_cover_data(blocks)
        improved_count = 0

        # Phase A: Remove redundant blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 3.0:
                    return blocks, improved_count
                ts_arr = block_ts_indices[i]
                if np.all(cover_count[ts_arr] >= 2):
                    np.subtract.at(cover_count, ts_arr, 1)
                    blocks.pop(i)
                    block_ts_indices.pop(i)
                    sol_block_set.discard(get_block_idx(blocks[i-1]) if i-1 < len(blocks) else -1)
                    improved_count += 1
                    changed = True
                    break

        # Phase B: 1-exchange
        iters = 0
        while iters < max_iters:
            iters += 1
            if time.time() > deadline - 3.0:
                break
            if len(blocks) == 0:
                break

            i = random.randint(0, len(blocks) - 1)
            ts_arr_i = block_ts_indices[i]

            unique_mask = cover_count[ts_arr_i] == 1
            unique_ts = ts_arr_i[unique_mask]

            if len(unique_ts) == 0:
                np.subtract.at(cover_count, ts_arr_i, 1)
                blocks.pop(i)
                block_ts_indices.pop(i)
                improved_count += 1
                continue

            candidates = set(ts_to_blocks[int(unique_ts[0])])
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(ts_to_blocks[int(ts_idx)])
                if not candidates:
                    break

            if not candidates:
                continue

            valid_candidates = [c for c in candidates if c not in sol_block_set]
            if not valid_candidates:
                continue

            new_block_idx = random.choice(valid_candidates)
            new_ts_arr = block_to_ts_np[new_block_idx]

            np.subtract.at(cover_count, ts_arr_i, 1)
            np.add.at(cover_count, new_ts_arr, 1)

            if np.min(cover_count) >= 1:
                blocks[i] = list(all_k_blocks[new_block_idx])
                block_ts_indices[i] = new_ts_arr
                sol_block_set.discard(get_block_idx(blocks[i]))
                sol_block_set.add(new_block_idx)
                improved_count += 1
            else:
                np.subtract.at(cover_count, new_ts_arr, 1)
                np.add.at(cover_count, ts_arr_i, 1)

        return blocks, improved_count

    def fast_local_search_2exchange(blocks, max_iters=2000):
        n_blocks = len(blocks)
        if n_blocks < 3:
            return blocks, 0

        cover_count, block_ts_indices, sol_block_set = build_cover_data(blocks)
        improved_count = 0
        iters = 0

        while iters < max_iters:
            iters += 1
            if time.time() > deadline - 5.0:
                break
            if len(blocks) < 3:
                break

            i = random.randint(0, len(blocks) - 1)
            j = random.randint(0, len(blocks) - 1)
            if i == j:
                continue

            ts_i = block_ts_indices[i]
            ts_j = block_ts_indices[j]

            unique_i_mask = (cover_count[ts_i] == 1)
            unique_i = ts_i[unique_i_mask]
            unique_j_mask = (cover_count[ts_j] == 1)
            unique_j = ts_j[unique_j_mask]

            all_unique = np.concatenate([unique_i, unique_j]) if (len(unique_i) > 0 or len(unique_j) > 0) else np.array([], dtype=np.int32)

            if len(all_unique) == 0:
                np.subtract.at(cover_count, ts_i, 1)
                blocks.pop(i)
                block_ts_indices.pop(i)
                if i < len(blocks):
                    j = j - 1 if j > i else j
                improved_count += 1
                continue

            candidates = set(ts_to_blocks[int(all_unique[0])])
            for ts_idx in all_unique[1:]:
                candidates.intersection_update(ts_to_blocks[int(ts_idx)])
                if not candidates:
                    break

            idx_i = get_block_idx(blocks[i])
            idx_j = get_block_idx(blocks[j])
            candidates.discard(idx_i)
            candidates.discard(idx_j)

            if not candidates:
                continue

            new_block_idx = random.choice(list(candidates))
            new_ts_arr = block_to_ts_np[new_block_idx]

            np.subtract.at(cover_count, ts_i, 1)
            np.subtract.at(cover_count, ts_j, 1)
            np.add.at(cover_count, new_ts_arr, 1)

            if np.min(cover_count) >= 1:
                hi = max(i, j)
                lo = min(i, j)
                blocks.pop(hi)
                block_ts_indices.pop(hi)
                blocks.pop(lo)
                block_ts_indices.pop(lo)

                blocks.append(list(all_k_blocks[new_block_idx]))
                block_ts_indices.append(new_ts_arr)
                sol_block_set.discard(idx_i)
                sol_block_set.discard(idx_j)
                sol_block_set.add(new_block_idx)
                improved_count += 1
            else:
                np.subtract.at(cover_count, new_ts_arr, 1)
                np.add.at(cover_count, ts_j, 1)
                np.add.at(cover_count, ts_i, 1)

        return blocks, improved_count

    def vnd_m_exchange(blocks, m, max_iters=1000):
        """Variable-neighborhood descent: m-for-m block swap.
        Pick m blocks, find m new blocks that collectively cover all
        unique t-subsets of the original m blocks."""
        n_blocks = len(blocks)
        if n_blocks < m + 1:
            return blocks, 0

        cover_count, block_ts_indices, sol_block_set = build_cover_data(blocks)
        improved_count = 0
        iters = 0

        while iters < max_iters:
            iters += 1
            if time.time() > deadline - 5.0:
                break
            if len(blocks) < m + 1:
                break

            # Pick m random distinct block indices
            indices = random.sample(range(len(blocks)), m)
            indices.sort()

            # Collect all unique t-subsets across these m blocks
            all_unique = []
            for idx in indices:
                ts_arr = block_ts_indices[idx]
                unique_mask = cover_count[ts_arr] == 1
                unique_ts = ts_arr[unique_mask]
                all_unique.extend(unique_ts.tolist())

            if len(all_unique) == 0:
                # All m blocks are fully redundant
                for idx in sorted(indices, reverse=True):
                    ts_arr = block_ts_indices[idx]
                    np.subtract.at(cover_count, ts_arr, 1)
                    blocks.pop(idx)
                    block_ts_indices.pop(idx)
                    improved_count += 1
                continue

            all_unique_set = set(all_unique)

            # Find candidate blocks that cover at least one unique t-subset
            # and intersect with all unique t-subsets' block lists
            # Strategy: pick the most constrained unique t-subset, get its blocks,
            # then filter for blocks that also cover other unique t-subsets
            # Sort by frequency to pick most constrained first
            unique_counts = {}
            for ts in all_unique_set:
                unique_counts[ts] = len(ts_to_blocks[ts])
            sorted_unique = sorted(all_unique_set, key=lambda x: unique_counts[x])

            # Start with blocks covering the most constrained t-subset
            candidates = set(ts_to_blocks[sorted_unique[0]])
            for ts_idx in sorted_unique[1:]:
                candidates.intersection_update(ts_to_blocks[ts_idx])
                if not candidates:
                    break

            if not candidates:
                # Try a softer approach: find blocks that cover a large fraction
                # of the unique t-subsets, then try to complete the set
                # For m=2: find one block covering most, then another for the rest
                if m == 2:
                    best_block1 = None
                    best_cover1 = -1
                    for ts_idx in sorted_unique[:5]:
                        for c in ts_to_blocks[ts_idx]:
                            if c in sol_block_set:
                                continue
                            # Count how many unique t-subsets this block covers
                            block_ts = block_ts_sets[c]
                            overlap = len(block_ts & all_unique_set)
                            if overlap > best_cover1:
                                best_cover1 = overlap
                                best_block1 = c
                    if best_block1 is not None and best_cover1 > len(all_unique_set) * 0.5:
                        remaining = all_unique_set - block_ts_sets[best_block1]
                        if remaining:
                            # Find block covering remaining
                            cands2 = set(ts_to_blocks[next(iter(remaining))])
                            for ts_idx in list(remaining)[1:]:
                                cands2.intersection_update(ts_to_blocks[ts_idx])
                                if not cands2:
                                    break
                            cands2.discard(best_block1)
                            cands2 -= sol_block_set
                            if cands2:
                                best_block2 = random.choice(list(cands2))
                                new_indices = [best_block1, best_block2]
                                new_ts_arrs = [block_to_ts_np[best_block1], block_to_ts_np[best_block2]]

                                # Check feasibility
                                np.subtract.at(cover_count, block_ts_indices[indices[0]], 1)
                                np.subtract.at(cover_count, block_ts_indices[indices[1]], 1)
                                np.add.at(cover_count, new_ts_arrs[0], 1)
                                np.add.at(cover_count, new_ts_arrs[1], 1)

                                if np.min(cover_count) >= 1:
                                    for idx in sorted(indices, reverse=True):
                                        blocks.pop(idx)
                                        block_ts_indices.pop(idx)
                                    blocks.extend([list(all_k_blocks[best_block1]), list(all_k_blocks[best_block2])])
                                    block_ts_indices.extend(new_ts_arrs)
                                    for idx in indices:
                                        sol_block_set.discard(get_block_idx(blocks[idx]) if idx < len(blocks) else -1)
                                    sol_block_set.add(best_block1)
                                    sol_block_set.add(best_block2)
                                    improved_count += 1
                                else:
                                    np.subtract.at(cover_count, new_ts_arrs[1], 1)
                                    np.subtract.at(cover_count, new_ts_arrs[0], 1)
                                    np.add.at(cover_count, block_ts_indices[indices[1]], 1)
                                    np.add.at(cover_count, block_ts_indices[indices[0]], 1)
                        else:
                            # All unique t-subsets covered by one block
                            new_block_idx = best_block1
                            new_ts_arr = block_to_ts_np[new_block_idx]
                            np.subtract.at(cover_count, block_ts_indices[indices[0]], 1)
                            np.subtract.at(cover_count, block_ts_indices[indices[1]], 1)
                            np.add.at(cover_count, new_ts_arr, 1)
                            if np.min(cover_count) >= 1:
                                for idx in sorted(indices, reverse=True):
                                    blocks.pop(idx)
                                    block_ts_indices.pop(idx)
                                blocks.append(list(all_k_blocks[new_block_idx]))
                                block_ts_indices.append(new_ts_arr)
                                sol_block_set.discard(get_block_idx(blocks[0]) if blocks else -1)
                                sol_block_set.add(new_block_idx)
                                improved_count += 1
                            else:
                                np.subtract.at(cover_count, new_ts_arr, 1)
                                np.add.at(cover_count, block_ts_indices[indices[1]], 1)
                                np.add.at(cover_count, block_ts_indices[indices[0]], 1)
                continue

            # Remove the original m blocks from candidates
            for idx in indices:
                candidates.discard(get_block_idx(blocks[idx]))
            candidates -= sol_block_set

            if not candidates:
                continue

            # For m=2: pick 2 blocks from candidates that together cover all unique t-subsets
            if m == 2:
                found = False
                cand_list = list(candidates)
                if len(cand_list) <= 200:
                    for a in range(len(cand_list)):
                        if time.time() > deadline - 3.0:
                            break
                        for b in range(a + 1, len(cand_list)):
                            ts_a = block_ts_sets[cand_list[a]]
                            ts_b = block_ts_sets[cand_list[b]]
                            if (ts_a | ts_b) >= all_unique_set:
                                # Feasibility check
                                np.subtract.at(cover_count, block_ts_indices[indices[0]], 1)
                                np.subtract.at(cover_count, block_ts_indices[indices[1]], 1)
                                np.add.at(cover_count, block_to_ts_np[cand_list[a]], 1)
                                np.add.at(cover_count, block_to_ts_np[cand_list[b]], 1)
                                if np.min(cover_count) >= 1:
                                    for idx in sorted(indices, reverse=True):
                                        blocks.pop(idx)
                                        block_ts_indices.pop(idx)
                                    blocks.extend([list(all_k_blocks[cand_list[a]]), list(all_k_blocks[cand_list[b]])])
                                    block_ts_indices.extend([block_to_ts_np[cand_list[a]], block_to_ts_np[cand_list[b]]])
                                    sol_block_set.discard(get_block_idx(blocks[-2]))
                                    sol_block_set.discard(get_block_idx(blocks[-1]))
                                    sol_block_set.add(cand_list[a])
                                    sol_block_set.add(cand_list[b])
                                    improved_count += 1
                                    found = True
                                    break
                                else:
                                    np.subtract.at(cover_count, block_to_ts_np[cand_list[b]], 1)
                                    np.subtract.at(cover_count, block_to_ts_np[cand_list[a]], 1)
                                    np.add.at(cover_count, block_ts_indices[indices[1]], 1)
                                    np.add.at(cover_count, block_ts_indices[indices[0]], 1)
                        if found:
                            break
                else:
                    # Sample
                    sample = random.sample(cand_list, min(100, len(cand_list)))
                    for a in range(len(sample)):
                        if time.time() > deadline - 3.0:
                            break
                        for b in range(a + 1, len(sample)):
                            ts_a = block_ts_sets[sample[a]]
                            ts_b = block_ts_sets[sample[b]]
                            if (ts_a | ts_b) >= all_unique_set:
                                np.subtract.at(cover_count, block_ts_indices[indices[0]], 1)
                                np.subtract.at(cover_count, block_ts_indices[indices[1]], 1)
                                np.add.at(cover_count, block_to_ts_np[sample[a]], 1)
                                np.add.at(cover_count, block_to_ts_np[sample[b]], 1)
                                if np.min(cover_count) >= 1:
                                    for idx in sorted(indices, reverse=True):
                                        blocks.pop(idx)
                                        block_ts_indices.pop(idx)
                                    blocks.extend([list(all_k_blocks[sample[a]]), list(all_k_blocks[sample[b]])])
                                    block_ts_indices.extend([block_to_ts_np[sample[a]], block_to_ts_np[sample[b]]])
                                    sol_block_set.discard(get_block_idx(blocks[-2]))
                                    sol_block_set.discard(get_block_idx(blocks[-1]))
                                    sol_block_set.add(sample[a])
                                    sol_block_set.add(sample[b])
                                    improved_count += 1
                                    found = True
                                    break
                                else:
                                    np.subtract.at(cover_count, block_to_ts_np[sample[b]], 1)
                                    np.subtract.at(cover_count, block_to_ts_np[sample[a]], 1)
                                    np.add.at(cover_count, block_ts_indices[indices[1]], 1)
                                    np.add.at(cover_count, block_ts_indices[indices[0]], 1)
                        if found:
                            break
            elif m == 3:
                # For m=3: greedy selection of 3 blocks
                cand_list = list(candidates)
                if len(cand_list) > 150:
                    cand_list = random.sample(cand_list, 150)

                # Greedily pick 3 blocks maximizing coverage of all_unique_set
                remaining = set(all_unique_set)
                selected = []
                for _ in range(3):
                    if not remaining:
                        break
                    best_block = None
                    best_cov = 0
                    for c in cand_list:
                        if c in selected:
                            continue
                        cov = len(block_ts_sets[c] & remaining)
                        if cov > best_cov:
                            best_cov = cov
                            best_block = c
                    if best_block is None or best_cov == 0:
                        break
                    selected.append(best_block)
                    remaining -= block_ts_sets[best_block]

                if len(selected) == 3 and not remaining:
                    new_ts_arrs = [block_to_ts_np[c] for c in selected]
                    for idx in sorted(indices, reverse=True):
                        np.subtract.at(cover_count, block_ts_indices[idx], 1)
                    for arr in new_ts_arrs:
                        np.add.at(cover_count, arr, 1)

                    if np.min(cover_count) >= 1:
                        for idx in sorted(indices, reverse=True):
                            blocks.pop(idx)
                            block_ts_indices.pop(idx)
                        for c in selected:
                            blocks.append(list(all_k_blocks[c]))
                            block_ts_indices.append(block_to_ts_np[c])
                            sol_block_set.add(c)
                        improved_count += 1
                    else:
                        for arr in new_ts_arrs:
                            np.subtract.at(cover_count, arr, 1)
                        for idx in indices:
                            np.add.at(cover_count, block_ts_indices[idx], 1)

        return blocks, improved_count

    # Phase 2: Iterative local search with VND
    for _ in range(120):
        if time.time() > deadline - 10.0:
            break
        best_solution, improved = fast_local_search(best_solution, max_iters=2000)
        if len(best_solution) < best_len:
            best_len = len(best_solution)

        if time.time() > deadline - 8.0:
            break
        best_solution, improved2 = fast_local_search_2exchange(best_solution, max_iters=1500)
        if len(best_solution) < best_len:
            best_len = len(best_solution)

        if time.time() > deadline - 6.0:
            break
        best_solution, improved3 = vnd_m_exchange(best_solution, 2, max_iters=800)
        if len(best_solution) < best_len:
            best_len = len(best_solution)

        if time.time() > deadline - 5.0:
            break
        best_solution, improved4 = vnd_m_exchange(best_solution, 3, max_iters=400)
        if len(best_solution) < best_len:
            best_len = len(best_solution)

        total_improved = improved + improved2 + improved3 + improved4
        if total_improved == 0 and time.time() > deadline - 20.0:
            break

    # Final polish: aggressive redundancy removal
    for _ in range(30):
        if time.time() > deadline - 2.0:
            break
        changed = False
        for i in range(len(best_solution)):
            if time.time() > deadline - 2.0:
                break
            b = best_solution[i]
            idx = get_block_idx(b)
            if idx < 0:
                continue
            ts_arr = block_to_ts_np[idx]

            temp_cover = np.zeros(num_t_subsets, dtype=np.int32)
            for j, b2 in enumerate(best_solution):
                if j == i:
                    continue
                idx2 = get_block_idx(b2)
                if idx2 >= 0:
                    np.add.at(temp_cover, block_to_ts_np[idx2], 1)

            if np.all(temp_cover[ts_arr] >= 1):
                best_solution.pop(i)
                if len(best_solution) < best_len:
                    best_len = len(best_solution)
                changed = True
                break
        if not changed:
            break

    return best_solution
