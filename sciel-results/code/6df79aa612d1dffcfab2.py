import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    time_limit = 118.0  # Safety margin for 120s limit

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
        """Greedy set cover with weighted gain and randomized tie-breaking."""
        rng = random.Random(seed)
        covered = np.zeros(n_ts, dtype=bool)
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        blocks = []
        point_load = np.zeros(v, dtype=np.int32)

        # Weighted gain: prefer blocks that cover uncovered triples,
        # and among those, prefer blocks whose triples are less over-covered
        while True:
            if time.time() - start_time > time_limit - 5:
                break
            uncovered_flags = ~covered[block_t_indices]
            # Primary gain: number of newly covered triples
            gains = uncovered_flags.sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break

            candidates = np.where(gains == max_gain)[0]
            if len(candidates) > 1:
                # Secondary score: prefer blocks with lower average coverage count
                # on their triples (less redundancy), plus point load balance
                scores = np.zeros(len(candidates), dtype=np.float64)
                for ci, c_idx in enumerate(candidates):
                    block = all_ks[c_idx]
                    load = sum(point_load[p] for p in block)
                    # Average coverage of triples in this block (lower is better)
                    avg_cov = np.mean(coverage_count[block_t_indices[c_idx]])
                    scores[ci] = -load * 0.1 - avg_cov + rng.uniform(0, 0.3)
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
        """Remove redundant blocks iteratively."""
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
                if np.min(coverage_count[subs]) > 1:
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

                    # Critical triples: those that would become uncovered if we remove both b1 and b2
                    critical = set()
                    for s in subs1:
                        if coverage_count[s] == 1:
                            critical.add(s)
                    for s in subs2:
                        if coverage_count[s] == 1:
                            critical.add(s)
                    # Triples in both with cov==2
                    set1 = set(subs1.tolist())
                    set2 = set(subs2.tolist())
                    for s in set1 & set2:
                        if coverage_count[s] == 2:
                            critical.add(s)

                    if not critical:
                        continue

                    # Find a single block that covers all critical triples
                    crit_triples = [all_ts[c] for c in critical]
                    all_points_needed = set()
                    for ct in crit_triples:
                        all_points_needed.update(ct)
                    if len(all_points_needed) > k:
                        continue

                    # Use the first critical triple to narrow down candidates
                    first_tri = crit_triples[0]
                    first_tri_idx = idx_map[first_tri]
                    candidate_blocks = triple_to_blocks[first_tri_idx]

                    for cb in candidate_blocks:
                        cb_subs = set(block_t_indices[cb].tolist())
                        if all(c in cb_subs for c in critical):
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
                # Triples uniquely covered by b (coverage_count == 1)
                unique = [s for s in subs if coverage_count[s] == 1]
                if not unique:
                    continue

                # Find a block that covers all unique triples
                first_tri = all_ts[unique[0]]
                first_tri_idx = idx_map[first_tri]
                candidates = triple_to_blocks[first_tri_idx]

                best_cb = None
                best_extra_gain = -1
                for cb in candidates:
                    if cb == b:
                        continue
                    cb_subs = set(block_t_indices[cb].tolist())
                    if all(s in cb_subs for s in unique):
                        # Count how many additional triples cb covers that are already covered
                        # (i.e., cb adds no new unique coverage burden)
                        extra = sum(1 for s in cb_subs if s not in set(unique))
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

    def kick_polish(blocks_list, time_budget):
        """Kick out a random block, re-greedy the uncovered parts, then reduce."""
        current = list(blocks_list)
        deadline = time.time() + time_budget
        rng = random.Random(int(time.time() * 1000) % 10000)
        
        # Build initial coverage
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current:
            coverage_count[block_t_indices[b_idx]] += 1
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            if len(current) == 0:
                break
                
            # Pick a random block to kick
            kick_idx = rng.randint(0, len(current) - 1)
            kick_block = current[kick_idx]
            
            # Remove it
            new_blocks = [b for i, b in enumerate(current) if i != kick_idx]
            coverage_count[block_t_indices[kick_block]] -= 1
            
            # Find uncovered
            uncovered = coverage_count == 0
            
            if not np.any(uncovered):
                # We improved!
                current = new_blocks
                improved = True
                continue
            
            # Need to cover the newly uncovered triples
            # Use greedy to pick the best block
            gains = uncovered[block_t_indices].sum(axis=1)
            max_gain = np.max(gains)
            
            if max_gain == 0:
                # Can't fix, revert
                coverage_count[block_t_indices[kick_block]] += 1
                break
                
            candidates = np.where(gains == max_gain)[0]
            
            # Prefer blocks not already in new_blocks
            valid_candidates = [c for c in candidates if c not in new_blocks]
            if not valid_candidates:
                valid_candidates = list(candidates)
                
            # Pick randomly from top candidates to add diversity
            new_block = rng.choice(valid_candidates)
            
            # Check if this actually helps (reduces total uncovered or keeps it same but we added 1 block)
            # Actually, we just add it and see if we can reduce later
            new_blocks.append(new_block)
            coverage_count[block_t_indices[new_block]] += 1
            
            # Now try to reduce
            reduced = local_search_remove(new_blocks)
            
            if len(reduced) < len(current):
                current = reduced
                improved = True
                # Rebuild coverage for next iteration
                coverage_count = np.zeros(n_ts, dtype=np.int32)
                for b_idx in current:
                    coverage_count[block_t_indices[b_idx]] += 1
            else:
                # Revert
                coverage_count[block_t_indices[kick_block]] += 1
                break
                
        return current

    # Main loop: multiple restarts with different seeds
    best_blocks = None
    best_count = float('inf')
    restart_deadline = time_limit * 0.50  # Leave more time for final local search

    seed = 42
    while time.time() - start_time < restart_deadline:
        random.seed(seed)
        np.random.seed(seed)
        greedy_blocks = greedy(seed)
        # Local search: remove then replace
        refined = local_search_remove(greedy_blocks)
        if time.time() - start_time < time_limit - 10:
            remaining = time_limit - (time.time() - start_time) - 10
            refined = local_search_replace(refined, min(2.0, max(0.1, remaining * 0.08)))
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
            polished = local_search_replace(polished, min(6.0, max(0.5, remaining * 0.5)))
        if time.time() - start_time < time_limit - 3:
            polished = local_search_remove(polished)
        if time.time() - start_time < time_limit - 2:
            remaining = time.time() - start_time
            # Use kick polish in the remaining time, but be careful not to exceed limit
            kick_budget = min(4.0, max(0.3, time_limit - 1 - remaining))
            if kick_budget > 0.1:
                polished = kick_polish(polished, kick_budget)
        if time.time() - start_time < time_limit - 1:
            polished = local_search_remove(polished)
        best_blocks = polished

    # Convert to final format
    final_blocks = [list(all_ks[i]) for i in best_blocks]
    return final_blocks
