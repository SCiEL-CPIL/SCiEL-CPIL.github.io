import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit

    # Precomputation
    points = list(range(v))
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}

    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)

    def nCr(n, r):
        if r > n:
            return 0
        if r == 0:
            return 1
        num = 1
        den = 1
        for i in range(r):
            num *= (n - i)
            den *= (i + 1)
        return num // den

    t_count = nCr(k, t)

    # block_t_indices[i] = array of t-subset indices contained in k-subset i
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = list(itertools.combinations(block, t))
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = idx_map[sub]

    # Precompute: for each triple index, which blocks contain it
    triple_to_blocks = defaultdict(list)
    for bi in range(n_ks):
        for ti in block_t_indices[bi]:
            triple_to_blocks[ti].append(bi)

    def greedy(seed):
        """Greedy set cover with smooth scoring and randomized tie-breaking."""
        rng = random.Random(seed)
        covered = np.zeros(n_ts, dtype=bool)
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        blocks = []
        point_load = np.zeros(v, dtype=np.int32)

        while True:
            if time.time() - start_time > time_limit - 5:
                break
            uncovered_flags = ~covered[block_t_indices]
            gains = uncovered_flags.sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break

            candidates = np.where(gains == max_gain)[0]
            if len(candidates) > 1:
                scores = np.zeros(len(candidates), dtype=np.float64)
                for ci, c_idx in enumerate(candidates):
                    block = all_ks[c_idx]
                    load = float(np.sum(point_load[list(block)]))
                    avg_cov = float(np.mean(coverage_count[block_t_indices[c_idx]]))
                    scores[ci] = -0.5 * load - 1.0 * avg_cov + rng.gauss(0, 0.2)
                best_local = np.argmax(scores)
                best_idx = candidates[best_local]
            else:
                best_idx = candidates[0]

            blocks.append(best_idx)
            block = all_ks[best_idx]
            subs_in_block = block_t_indices[best_idx]
            covered[subs_in_block] = True
            coverage_count[subs_in_block] += 1
            for p in block:
                point_load[p] += 1

            if np.all(covered):
                break

        return blocks

    def local_search_remove(blocks_list):
        """Remove redundant blocks iteratively (optimized)."""
        current = list(blocks_list)
        improved = True
        while improved:
            improved = False
            coverage_count = np.zeros(n_ts, dtype=np.int32)
            for b_idx in current:
                coverage_count[block_t_indices[b_idx]] += 1

            new_current = []
            for b_idx in current:
                subs = block_t_indices[b_idx]
                if np.all(coverage_count[subs] > 1):
                    improved = True
                    coverage_count[subs] -= 1
                else:
                    new_current.append(b_idx)
            current = new_current
        return current

    def local_search_replace(blocks_list, time_budget):
        """Try to replace pairs of blocks with a single block (2-for-1 swap)."""
        current = list(blocks_list)
        deadline = time.time() + time_budget

        while time.time() < deadline:
            coverage_count = np.zeros(n_ts, dtype=np.int32)
            for b_idx in current:
                coverage_count[block_t_indices[b_idx]] += 1

            n_blocks = len(current)
            found_swap = False

            for i in range(n_blocks):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n_blocks):
                    if time.time() > deadline:
                        break

                    b1 = current[i]
                    b2 = current[j]
                    subs1 = block_t_indices[b1]
                    subs2 = block_t_indices[b2]

                    critical_set = set()
                    for s in subs1:
                        if coverage_count[s] == 1:
                            critical_set.add(int(s))
                    for s in subs2:
                        if coverage_count[s] == 1:
                            critical_set.add(int(s))
                    set1 = set(subs1.tolist())
                    set2 = set(subs2.tolist())
                    for s in set1 & set2:
                        if coverage_count[s] == 2:
                            critical_set.add(int(s))

                    if not critical_set:
                        continue

                    all_points_needed = set()
                    for c in critical_set:
                        all_points_needed.update(all_ts[c])
                    if len(all_points_needed) > k:
                        continue

                    first_tri_idx = list(critical_set)[0]
                    candidate_blocks = triple_to_blocks[first_tri_idx]

                    for cb in candidate_blocks:
                        cb_subs = set(block_t_indices[cb].tolist())
                        if critical_set.issubset(cb_subs):
                            new_current = [b for idx, b in enumerate(current) if idx != i and idx != j]
                            if cb not in new_current:
                                new_current.append(cb)
                            current = new_current
                            found_swap = True
                            break
                    if found_swap:
                        break
                if found_swap:
                    break

            if not found_swap:
                break

        return current

    def local_search_single(blocks_list, time_budget):
        """Try to replace one block with another that covers all its unique triples."""
        current = list(blocks_list)
        deadline = time.time() + time_budget

        while time.time() < deadline:
            coverage_count = np.zeros(n_ts, dtype=np.int32)
            for b_idx in current:
                coverage_count[block_t_indices[b_idx]] += 1

            n_blocks = len(current)
            found = False

            for i in range(n_blocks):
                if time.time() > deadline:
                    break
                b = current[i]
                subs = block_t_indices[b]
                unique = [int(s) for s in subs if coverage_count[s] == 1]
                if not unique:
                    continue

                first_tri_idx = unique[0]
                candidates = triple_to_blocks[first_tri_idx]
                unique_set = set(unique)

                best_cb = None
                best_extra_gain = -1
                for cb in candidates:
                    if cb == b:
                        continue
                    cb_subs = set(block_t_indices[cb].tolist())
                    if unique_set.issubset(cb_subs):
                        extra = len(cb_subs - unique_set)
                        if extra > best_extra_gain:
                            best_extra_gain = extra
                            best_cb = cb

                if best_cb is not None:
                    new_current = current[:i] + current[i+1:]
                    if best_cb not in new_current:
                        new_current.append(best_cb)
                    current = new_current
                    found = True
                    break

            if not found:
                break

        return current

    def local_search_swap_point(blocks_list, time_budget):
        """Try to swap a point in a block to improve coverage, then remove redundant blocks."""
        current = list(blocks_list)
        deadline = time.time() + time_budget
        rng = random.Random(12345)

        while time.time() < deadline:
            coverage_count = np.zeros(n_ts, dtype=np.int32)
            for b_idx in current:
                coverage_count[block_t_indices[b_idx]] += 1

            improved = False
            n_blocks = len(current)
            if n_blocks == 0:
                break

            # Try random point swaps
            for _ in range(200):
                if time.time() > deadline:
                    break

                idx = rng.randint(0, n_blocks - 1)
                block = all_ks[current[idx]]
                old_x = rng.choice(list(block))
                new_x = rng.randint(0, v - 1)
                if new_x == old_x or new_x in block:
                    continue

                # Simulate swap
                other_points = [p for p in block if p != old_x]
                new_block = tuple(sorted([new_x] + other_points))
                new_block_set = set(new_block)

                # Find the index of new_block in all_ks if it exists
                # We need to check if new_block is a valid k-subset
                if len(new_block_set) != k:
                    continue

                # Check if new_block exists in our precomputed list
                # Since all_ks contains all C(v,k) combinations, new_block must be there
                # But we need its index. Let's compute it efficiently.
                # Actually, since we have all_ks as a list of tuples, we can't easily look up.
                # Instead, let's just check validity by coverage.
                
                old_subs = set(block_t_indices[current[idx]].tolist())
                # Compute new_subs
                new_subs_list = []
                for sub in itertools.combinations(new_block, t):
                    new_subs_list.append(idx_map[sub])
                new_subs = set(new_subs_list)
                
                # Check if swap is valid: no triple that was only covered by old block
                # and is lost in new block
                lost = old_subs - new_subs
                for s in lost:
                    if coverage_count[s] <= 1:
                        break
                else:
                    # Swap is valid
                    # Apply swap
                    for s in old_subs:
                        coverage_count[s] -= 1
                    for s in new_subs:
                        coverage_count[s] += 1
                    
                    # Find index of new_block
                    # This is expensive. Let's just track blocks as tuples instead of indices
                    # For now, let's just note that we'd need to map new_block to an index
                    # Since this is complex, let's skip this particular swap strategy
                    # and rely on the other local search methods
                    pass

            if not improved:
                break

        return current

    def local_search_3_for_1(blocks_list, time_budget):
        """Try to replace 3 blocks with 1 block."""
        current = list(blocks_list)
        deadline = time.time() + time_budget

        while time.time() < deadline:
            coverage_count = np.zeros(n_ts, dtype=np.int32)
            for b_idx in current:
                coverage_count[block_t_indices[b_idx]] += 1

            n_blocks = len(current)
            if n_blocks < 3:
                break

            found_swap = False

            # Sample a few triples of blocks
            for _ in range(50):
                if time.time() > deadline:
                    break

                i, j, m = rng_sample_3(n_blocks)
                if i is None:
                    continue

                b1, b2, b3 = current[i], current[j], current[m]
                subs1 = set(block_t_indices[b1].tolist())
                subs2 = set(block_t_indices[b2].tolist())
                subs3 = set(block_t_indices[b3].tolist())

                critical_set = set()
                for s in subs1:
                    if coverage_count[s] == 1:
                        critical_set.add(int(s))
                for s in subs2:
                    if coverage_count[s] == 1:
                        critical_set.add(int(s))
                for s in subs3:
                    if coverage_count[s] == 1:
                        critical_set.add(int(s))
                for s in (subs1 & subs2):
                    if coverage_count[s] == 2:
                        critical_set.add(int(s))
                for s in (subs1 & subs3):
                    if coverage_count[s] == 2:
                        critical_set.add(int(s))
                for s in (subs2 & subs3):
                    if coverage_count[s] == 2:
                        critical_set.add(int(s))
                for s in (subs1 & subs2 & subs3):
                    if coverage_count[s] == 3:
                        critical_set.add(int(s))

                if not critical_set:
                    continue

                all_points_needed = set()
                for c in critical_set:
                    all_points_needed.update(all_ts[c])
                if len(all_points_needed) > k:
                    continue

                first_tri_idx = list(critical_set)[0]
                candidate_blocks = triple_to_blocks[first_tri_idx]

                for cb in candidate_blocks:
                    cb_subs = set(block_t_indices[cb].tolist())
                    if critical_set.issubset(cb_subs):
                        new_current = [b for idx, b in enumerate(current) if idx not in (i, j, m)]
                        if cb not in new_current:
                            new_current.append(cb)
                        current = new_current
                        found_swap = True
                        break
                if found_swap:
                    break

            if not found_swap:
                break

        return current

    def rng_sample_3(n):
        """Sample 3 distinct indices from range(n)."""
        if n < 3:
            return None, None, None
        a = random.randint(0, n - 1)
        b = random.randint(0, n - 1)
        if b == a:
            b = (b + 1) % n
        c = random.randint(0, n - 1)
        if c == a or c == b:
            c = (c + 1) % n
            if c == a or c == b:
                c = (c + 1) % n
        return a, b, c

    # Main loop: multiple restarts with different seeds
    best_blocks = None
    best_count = float('inf')
    restart_deadline = time_limit * 0.40  # Leave more time for final local search

    seed = 42
    while time.time() - start_time < restart_deadline:
        random.seed(seed)
        np.random.seed(seed)
        greedy_blocks = greedy(seed)
        # Local search: remove then replace
        refined = local_search_remove(greedy_blocks)
        if time.time() - start_time < time_limit - 10:
            remaining = time_limit - (time.time() - start_time) - 10
            refined = local_search_replace(refined, min(1.5, max(0.1, remaining * 0.05)))
        if time.time() - start_time < time_limit - 8:
            refined = local_search_remove(refined)

        count = len(refined)
        if count < best_count:
            best_count = count
            best_blocks = refined

        seed += 1
        if best_count <= 70:  # Good enough, stop early
            break

    # Final polish on the best solution
    if best_blocks is not None and time.time() - start_time < time_limit - 5:
        polished = local_search_remove(best_blocks)
        if time.time() - start_time < time_limit - 4:
            remaining = time_limit - (time.time() - start_time) - 4
            polished = local_search_replace(polished, min(5.0, max(0.5, remaining * 0.4)))
        if time.time() - start_time < time_limit - 3:
            polished = local_search_remove(polished)
        if time.time() - start_time < time_limit - 2:
            remaining = time_limit - (time.time() - start_time) - 2
            polished = local_search_single(polished, min(3.0, max(0.3, remaining * 0.6)))
        if time.time() - start_time < time_limit - 1.5:
            remaining = time_limit - (time.time() - start_time) - 1.5
            polished = local_search_3_for_1(polished, min(2.0, max(0.2, remaining * 0.8)))
        if time.time() - start_time < time_limit - 1:
            polished = local_search_remove(polished)
        best_blocks = polished

    # Convert to final format
    final_blocks = [list(all_ks[i]) for i in best_blocks]
    return final_blocks
