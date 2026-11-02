import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0

    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}

    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)

    def nCr(n, r):
        if r > n: return 0
        if r == 0: return 1
        res = 1
        for i in range(r):
            res *= (n - i)
            res //= (i + 1)
        return res

    t_count = nCr(k, t)

    # Precompute: for each block, which t-subsets it covers
    # Use both a dense array and a set for fast membership
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    block_triple_sets = [set() for _ in range(n_ks)]
    for i, block in enumerate(all_ks):
        subs = list(itertools.combinations(block, t))
        for j, sub in enumerate(subs):
            idx = idx_map[sub]
            block_t_indices[i, j] = idx
            block_triple_sets[i].add(idx)

    # Precompute blocks containing each triple (for replacement search)
    blocks_containing_triple = [[] for _ in range(n_ts)]
    for i in range(n_ks):
        for idx in block_triple_sets[i]:
            blocks_containing_triple[idx].append(i)

    best_solution = None
    best_score = float('inf')

    def rebuild_coverage(block_list):
        cc = np.zeros(n_ts, dtype=np.int32)
        for b in block_list:
            cc[block_t_indices[b]] += 1
        return cc

    def aggressive_deletion(current_blocks, coverage_count):
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 4.0:
                break
            improved = False
            new_blocks = []
            for b_idx in current_blocks:
                subs_in_block = block_t_indices[b_idx]
                if np.all(coverage_count[subs_in_block] > 1):
                    improved = True
                    coverage_count[subs_in_block] -= 1
                else:
                    new_blocks.append(b_idx)
            current_blocks = new_blocks
        return current_blocks

    def find_replacement_block(L_indices, in_solution_mask):
        if not L_indices:
            return -1
        candidates_lists = []
        for t_idx in L_indices:
            lst = blocks_containing_triple[t_idx]
            candidates_lists.append((len(lst), t_idx, lst))
        candidates_lists.sort(key=lambda x: x[0])
        _, start_t, start_list = candidates_lists[0]
        other_ts = [t for (_, t, _) in candidates_lists[1:]]
        for k in start_list:
            if in_solution_mask[k]:
                continue
            valid = True
            block_set = block_triple_sets[k]
            for t_idx in other_ts:
                if t_idx not in block_set:
                    valid = False
                    break
            if valid:
                return k
        return None

    def two_for_one_exchange(current_blocks, coverage_count):
        n_blocks = len(current_blocks)
        in_solution_mask = np.zeros(n_ks, dtype=bool)
        for b in current_blocks:
            in_solution_mask[b] = True

        for _ in range(n_blocks):
            if time.time() - start_time > time_limit - 5.0:
                return False
            improved = False
            for i_idx in range(len(current_blocks)):
                b_i = current_blocks[i_idx]
                for j_idx in range(i_idx + 1, len(current_blocks)):
                    b_j = current_blocks[j_idx]

                    seen = set()
                    for t_idx in block_t_indices[b_i]:
                        seen.add(t_idx)
                    for t_idx in block_t_indices[b_j]:
                        seen.add(t_idx)

                    L = []
                    for t_idx in seen:
                        c = coverage_count[t_idx]
                        if t_idx in block_triple_sets[b_i]:
                            c -= 1
                        if t_idx in block_triple_sets[b_j]:
                            c -= 1
                        if c < 1:
                            L.append(t_idx)

                    if not L:
                        for t_idx in block_triple_sets[b_i]:
                            coverage_count[t_idx] -= 1
                        for t_idx in block_triple_sets[b_j]:
                            coverage_count[t_idx] -= 1
                        current_blocks.remove(b_j)
                        current_blocks.remove(b_i)
                        in_solution_mask[b_i] = False
                        in_solution_mask[b_j] = False
                        return True

                    if len(L) > t_count:
                        continue

                    k = find_replacement_block(L, in_solution_mask)
                    if k is not None:
                        for t_idx in block_triple_sets[b_i]:
                            coverage_count[t_idx] -= 1
                        for t_idx in block_triple_sets[b_j]:
                            coverage_count[t_idx] -= 1
                        for t_idx in block_triple_sets[k]:
                            coverage_count[t_idx] += 1
                        current_blocks.remove(b_j)
                        current_blocks.remove(b_i)
                        current_blocks.append(k)
                        in_solution_mask[b_i] = False
                        in_solution_mask[b_j] = False
                        in_solution_mask[k] = True
                        return True
        return False

    def bitset_local_search(current_blocks, coverage_count):
        """
        Use bitset-based 1-swap and 2-swap local search.
        coverage_count: array of shape (n_ts,) with counts.
        Returns (improved_blocks, improved_coverage) or (current, current) if no improvement.
        """
        n_blocks = len(current_blocks)
        if n_blocks < 2:
            return current_blocks, coverage_count

        # Build a set for O(1) membership
        block_set = set(current_blocks)

        improved = True
        max_passes = 50
        for _pass in range(max_passes):
            if time.time() - start_time > time_limit - 5.0:
                break
            improved = False

            # 1-swap: remove one block, add one block
            for i in range(len(current_blocks)):
                b_out = current_blocks[i]
                # Coverage if we remove b_out
                # A block b_in is a valid replacement if it covers all triples that would become uncovered
                # Triples that would become uncovered: those where coverage_count[t] == 1 and t in b_out
                subs_out = block_t_indices[b_out]
                uncovered_if_removed = [t for t in subs_out if coverage_count[t] == 1]

                if not uncovered_if_removed:
                    # Can remove b_out entirely
                    current_blocks.remove(b_out)
                    block_set.discard(b_out)
                    for t in subs_out:
                        coverage_count[t] -= 1
                    improved = True
                    break

                # Need a replacement that covers all uncovered_if_removed
                # Find candidate blocks that cover all these triples
                # Use the most constrained triple
                uncovered_if_removed.sort(key=lambda t: len(blocks_containing_triple[t]))
                start_t = uncovered_if_removed[0]
                other_ts = uncovered_if_removed[1:]

                found = False
                for b_in in blocks_containing_triple[start_t]:
                    if b_in in block_set:
                        continue
                    # Check if b_in covers all other_ts
                    valid = True
                    bs = block_triple_sets[b_in]
                    for t in other_ts:
                        if t not in bs:
                            valid = False
                            break
                    if valid:
                        # Perform swap
                        current_blocks.remove(b_out)
                        block_set.discard(b_out)
                        for t in subs_out:
                            coverage_count[t] -= 1

                        current_blocks.append(b_in)
                        block_set.add(b_in)
                        for t in block_t_indices[b_in]:
                            coverage_count[t] += 1
                        improved = True
                        found = True
                        break

                if found:
                    break

            if improved:
                continue

            # 2-swap: remove two blocks, add one or two blocks
            n_b = len(current_blocks)
            for i in range(n_b):
                if time.time() - start_time > time_limit - 5.0:
                    break
                b1 = current_blocks[i]
                for j in range(i + 1, n_b):
                    b2 = current_blocks[j]
                    # Compute uncovered if both removed
                    seen = set()
                    for t in block_t_indices[b1]:
                        seen.add(t)
                    for t in block_t_indices[b2]:
                        seen.add(t)

                    L = []
                    for t in seen:
                        c = coverage_count[t]
                        if t in block_triple_sets[b1]:
                            c -= 1
                        if t in block_triple_sets[b2]:
                            c -= 1
                        if c < 1:
                            L.append(t)

                    if not L:
                        # Remove both
                        current_blocks.remove(b2)
                        current_blocks.remove(b1)
                        block_set.discard(b1)
                        block_set.discard(b2)
                        for t in block_triple_sets[b1]:
                            coverage_count[t] -= 1
                        for t in block_triple_sets[b2]:
                            coverage_count[t] -= 1
                        improved = True
                        break

                    if len(L) > t_count:
                        continue

                    # Find a single block that covers all L
                    L_sorted = sorted(L, key=lambda t: len(blocks_containing_triple[t]))
                    start_t = L_sorted[0]
                    other_ts = L_sorted[1:]
                    found = False
                    for b_in in blocks_containing_triple[start_t]:
                        if b_in in block_set:
                            continue
                        valid = True
                        bs = block_triple_sets[b_in]
                        for t in other_ts:
                            if t not in bs:
                                valid = False
                                break
                        if valid:
                            # Replace b1, b2 with b_in
                            current_blocks.remove(b2)
                            current_blocks.remove(b1)
                            block_set.discard(b1)
                            block_set.discard(b2)
                            for t in block_triple_sets[b1]:
                                coverage_count[t] -= 1
                            for t in block_triple_sets[b2]:
                                coverage_count[t] -= 1
                            current_blocks.append(b_in)
                            block_set.add(b_in)
                            for t in block_t_indices[b_in]:
                                coverage_count[t] += 1
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if improved:
                    break

            if not improved:
                break

        return current_blocks, coverage_count

    def run_greedy(seed, use_random_weights):
        random.seed(seed)
        np.random.seed(seed)

        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts
        cover_freq = np.zeros(n_ts, dtype=np.int32)

        if use_random_weights:
            point_weights = np.random.uniform(0.5, 2.0, size=v).astype(np.float32)
        else:
            point_weights = None

        while remaining > 0:
            if time.time() - start_time > time_limit - 6.0:
                return None, None

            covered_counts = covered[block_t_indices].sum(axis=1)
            gains = t_count - covered_counts

            max_gain = np.max(gains)
            if max_gain == 0:
                break

            max_gain_blocks = np.where(gains == max_gain)[0]

            if len(max_gain_blocks) > 40:
                candidates = max_gain_blocks[random.sample(range(len(max_gain_blocks)), min(40, len(max_gain_blocks)))]
            else:
                candidates = max_gain_blocks

            best_score_candidate = -1
            best_candidate = candidates[0]
            for cand in candidates:
                subs = block_t_indices[cand]
                newly_covered = ~covered[subs]
                if np.any(newly_covered):
                    bonus = np.sum(1.0 / (1.0 + cover_freq[subs[newly_covered]]))
                else:
                    bonus = 0.0
                score_cand = max_gain + bonus * 0.1
                if score_cand > best_score_candidate:
                    best_score_candidate = score_cand
                    best_candidate = cand

            blocks_indices.append(best_candidate)

            subs_to_cover = block_t_indices[best_candidate]
            newly_covered = ~covered[subs_to_cover]
            new_covers = np.sum(newly_covered)
            remaining -= new_covers
            cover_freq[subs_to_cover] += 1
            covered[subs_to_cover] = True

        current_blocks = list(blocks_indices)
        coverage_count = rebuild_coverage(current_blocks)

        current_blocks = aggressive_deletion(current_blocks, coverage_count)
        coverage_count = rebuild_coverage(current_blocks)

        # Bitset local search
        if time.time() - start_time < time_limit - 8.0:
            current_blocks, coverage_count = bitset_local_search(current_blocks, coverage_count)

        # 2-for-1 exchange
        if time.time() - start_time < time_limit - 5.0:
            while two_for_one_exchange(current_blocks, coverage_count):
                coverage_count = rebuild_coverage(current_blocks)
                current_blocks = aggressive_deletion(current_blocks, coverage_count)
                coverage_count = rebuild_coverage(current_blocks)

        return current_blocks, len(current_blocks)

    # Main loop
    restart_count = 0
    while time.time() - start_time < time_limit - 20.0:
        seed = restart_count
        blocks, score = run_greedy(seed, use_random_weights=(restart_count % 2 == 1))
        if blocks is None:
            break
        if score < best_score:
            best_score = score
            best_solution = blocks
        restart_count += 1

    if best_solution is None:
        return []

    # Final polish
    coverage_count = rebuild_coverage(best_solution)
    best_solution = aggressive_deletion(best_solution, coverage_count)
    coverage_count = rebuild_coverage(best_solution)

    if time.time() - start_time < time_limit - 8.0:
        best_solution, coverage_count = bitset_local_search(best_solution, coverage_count)

    while time.time() - start_time < time_limit - 5.0:
        improved = two_for_one_exchange(best_solution, coverage_count)
        if not improved:
            break
        coverage_count = rebuild_coverage(best_solution)
        best_solution = aggressive_deletion(best_solution, coverage_count)
        coverage_count = rebuild_coverage(best_solution)

    final_blocks = [list(all_ks[i]) for i in best_solution]
    return final_blocks
