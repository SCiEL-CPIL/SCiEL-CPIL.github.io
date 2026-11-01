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
    # triple_to_blocks[tri] = list of block indices containing that triple
    triple_to_blocks = defaultdict(list)
    for bi in range(n_ks):
        for ti in block_t_indices[bi]:
            triple_to_blocks[ti].append(bi)

    def greedy(seed):
        """Greedy set cover with randomized tie-breaking."""
        rng = random.Random(seed)
        covered = np.zeros(n_ts, dtype=bool)
        blocks = []
        point_load = np.zeros(v, dtype=np.int32)

        while True:
            uncovered_flags = ~covered[block_t_indices]
            gains = uncovered_flags.sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break

            candidates = np.where(gains == max_gain)[0]
            if len(candidates) > 1:
                # Score: negative point load (prefer less loaded) + small random noise
                scores = np.zeros(len(candidates), dtype=np.float64)
                for ci, c_idx in enumerate(candidates):
                    block = all_ks[c_idx]
                    load = sum(point_load[p] for p in block)
                    scores[ci] = -load + rng.uniform(0, 0.5)
                best_local = np.argmax(scores)
                best_idx = candidates[best_local]
            else:
                best_idx = candidates[0]

            blocks.append(best_idx)
            block = all_ks[best_idx]
            covered[block_t_indices[best_idx]] = True
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
                    # A triple in b1 is critical if coverage_count == 1 (only b1 covers it)
                    # A triple in b2 is critical if coverage_count == 1 (only b2 covers it)
                    # A triple in both with coverage_count == 2 is critical
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
                    # Candidate blocks must contain all points that form the critical triples
                    # A block covers a critical triple if it contains all 3 points of that triple
                    # We need a block that contains ALL critical triples
                    # Strategy: pick the first critical triple, find blocks containing it,
                    # then check if they contain all other critical triples

                    # Get points of each critical triple
                    crit_triples = [all_ts[c] for c in critical]
                    # The replacement block must contain the union of all points in critical triples
                    # But a 7-set can only contain 7 points. If union > 7, impossible.
                    all_points_needed = set()
                    for ct in crit_triples:
                        all_points_needed.update(ct)
                    if len(all_points_needed) > k:
                        continue

                    # Find candidate blocks: must contain all points in all_points_needed
                    # and must contain each critical triple
                    # Use the first critical triple to narrow down
                    first_tri = crit_triples[0]
                    first_tri_idx = idx_map[first_tri]
                    candidate_blocks = triple_to_blocks[first_tri_idx]

                    for cb in candidate_blocks:
                        # Check if this block covers all critical triples
                        cb_subs = set(block_t_indices[cb].tolist())
                        if all(c in cb_subs for c in critical):
                            # Valid replacement!
                            # Remove b1 and b2, add cb
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

                for cb in candidates:
                    if cb == b:
                        continue
                    cb_subs = set(block_t_indices[cb].tolist())
                    if all(s in cb_subs for s in unique):
                        # Replace b with cb
                        new_current = current[:i] + current[i+1:]
                        if cb not in new_current:
                            new_current.append(cb)
                        current = new_current
                        found = True
                        break
                if found:
                    break

            if not found:
                break

        return current

    def simulated_annealing(blocks_list, time_budget, rng):
        """Simulated annealing to escape local minima.
        
        Moves: add a block, remove a block, or swap a block.
        Energy = number of uncovered triples + small penalty for extra blocks.
        """
        current = list(blocks_list)
        deadline = time.time() + time_budget
        
        # Initial energy
        coverage = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current:
            coverage[block_t_indices[b_idx]] += 1
        uncovered = np.sum(coverage == 0)
        energy = uncovered + 0.01 * len(current)
        
        best_energy = energy
        best_state = list(current)
        
        # Temperature schedule
        T_start = 5.0
        T_end = 0.01
        T = T_start
        cooling = 0.995
        
        current_covered_set = set(current)
        
        while time.time() < deadline and T > T_end:
            T *= cooling
            
            # Choose a random move
            move_type = rng.random()
            
            if move_type < 0.3 and len(current) < n_ks:
                # Add a random block
                new_block = rng.randint(0, n_ks - 1)
                if new_block in current_covered_set:
                    continue
                # Compute delta energy
                delta_uncovered = 0
                subs = block_t_indices[new_block]
                for s in subs:
                    if coverage[s] == 0:
                        delta_uncovered -= 1
                delta_energy = delta_uncovered + 0.01
                if delta_energy <= 0 or rng.random() < np.exp(-delta_energy / T):
                    current.append(new_block)
                    current_covered_set.add(new_block)
                    for s in subs:
                        coverage[s] += 1
                    energy += delta_energy
                    if energy < best_energy:
                        best_energy = energy
                        best_state = list(current)
                        
            elif move_type < 0.6 and len(current) > 1:
                # Remove a random block
                idx = rng.randint(0, len(current) - 1)
                b_idx = current[idx]
                subs = block_t_indices[b_idx]
                delta_uncovered = 0
                for s in subs:
                    if coverage[s] == 1:
                        delta_uncovered += 1
                delta_energy = delta_uncovered - 0.01
                if delta_energy <= 0 or rng.random() < np.exp(-delta_energy / T):
                    current.pop(idx)
                    current_covered_set.discard(b_idx)
                    for s in subs:
                        coverage[s] -= 1
                    energy += delta_energy
                    if energy < best_energy:
                        best_energy = energy
                        best_state = list(current)
                        
            else:
                # Swap: remove one block, add another
                if len(current) < 1:
                    continue
                idx = rng.randint(0, len(current) - 1)
                old_block = current[idx]
                new_block = rng.randint(0, n_ks - 1)
                if new_block == old_block or new_block in current_covered_set:
                    continue
                
                subs_old = block_t_indices[old_block]
                subs_new = block_t_indices[new_block]
                
                # Compute delta
                delta_uncovered = 0
                # First, what happens if we remove old_block
                for s in subs_old:
                    if coverage[s] == 1:
                        delta_uncovered += 1
                # Then, what happens if we add new_block (considering old_block removed)
                for s in subs_new:
                    if s in subs_old:
                        # This triple is in both, so coverage changes by 0 net from this pair
                        pass
                    else:
                        if coverage[s] == 0:
                            delta_uncovered -= 1
                
                delta_energy = delta_uncovered
                if delta_energy <= 0 or rng.random() < np.exp(-delta_energy / T):
                    # Perform swap
                    current[idx] = new_block
                    current_covered_set.discard(old_block)
                    current_covered_set.add(new_block)
                    for s in subs_old:
                        coverage[s] -= 1
                    for s in subs_new:
                        coverage[s] += 1
                    energy += delta_energy
                    if energy < best_energy:
                        best_energy = energy
                        best_state = list(current)
            
            # Occasionally do a greedy repair if we have many uncovered
            if len(current) > 0 and uncovered < 0:  # placeholder
                pass
        
        return best_state

    # Main loop: multiple restarts with different seeds
    best_blocks = None
    best_count = float('inf')
    restart_deadline = time_limit * 0.50  # Leave more time for final local search and SA

    seed = 42
    while time.time() - start_time < restart_deadline:
        random.seed(seed)
        np.random.seed(seed)
        greedy_blocks = greedy(seed)
        # Local search: remove then replace
        ls_time = min(2.0, max(0.1, (time_limit - (time.time() - start_time)) * 0.1))
        refined = local_search_remove(greedy_blocks)
        if time.time() - start_time < time_limit - 10:
            refined = local_search_replace(refined, min(1.5, max(0.1, (time_limit - (time.time() - start_time)) * 0.05)))
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
        polish_time = min(10.0, max(0.5, time_limit - (time.time() - start_time) - 3.0))
        polished = local_search_remove(best_blocks)
        if time.time() - start_time < time_limit - 3:
            polished = local_search_replace(polished, min(5.0, max(0.5, time_limit - (time.time() - start_time) - 3.0)))
        if time.time() - start_time < time_limit - 2:
            polished = local_search_remove(polished)
        
        # Add simulated annealing phase
        if time.time() - start_time < time_limit - 1:
            sa_time = min(8.0, max(0.5, time_limit - (time.time() - start_time) - 1.0))
            sa_rng = random.Random(12345)
            sa_result = simulated_annealing(polished, sa_time, sa_rng)
            if len(sa_result) < len(polished):
                polished = sa_result
        
        if time.time() - start_time < time_limit - 1:
            polished = local_search_remove(polished)
        if time.time() - start_time < time_limit - 0.5:
            polished = local_search_single(polished, min(2.0, max(0.2, time_limit - (time.time() - start_time) - 0.5)))
        best_blocks = polished

    # Convert to final format
    final_blocks = [list(all_ks[i]) for i in best_blocks]
    return final_blocks
