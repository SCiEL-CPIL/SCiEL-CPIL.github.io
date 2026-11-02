import numpy as np
import itertools
import time
import random
from collections import defaultdict

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
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = idx_map[sub]
            
    # Build inverted index: for each t-subset, list of blocks containing it
    ts_to_blocks = [[] for _ in range(n_ts)]
    for i in range(n_ks):
        for j in range(t_count):
            ts_idx = block_t_indices[i, j]
            ts_to_blocks[ts_idx].append(i)
    
    # Convert to numpy arrays for faster access
    ts_to_blocks_arr = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in ts_to_blocks]
    
    best_solution = None
    best_score = float('inf')
    
    def aggressive_deletion(current_blocks, coverage_count):
        """Remove redundant blocks with two passes."""
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 3.0:
                break
            improved = False
            new_blocks = []
            
            # Pass 1: Remove blocks where all subsets are covered by others
            for b_idx in current_blocks:
                subs_in_block = block_t_indices[b_idx]
                if np.all(coverage_count[subs_in_block] > 1):
                    improved = True
                    coverage_count[subs_in_block] -= 1
                else:
                    new_blocks.append(b_idx)
            current_blocks = new_blocks
            
            # Pass 2: Try removing blocks where most subsets are over-covered
            if time.time() - start_time > time_limit - 3.0:
                break
            improved = False
            new_blocks = []
            for b_idx in current_blocks:
                subs_in_block = block_t_indices[b_idx]
                other_covers = coverage_count[subs_in_block] - 1
                if np.all(other_covers > 0):
                    improved = True
                    coverage_count[subs_in_block] -= 1
                else:
                    # Check if we can remove by verifying uncovered subsets are covered elsewhere
                    uncovered_mask = other_covers <= 0
                    if np.any(uncovered_mask):
                        uncovered_subs = subs_in_block[uncovered_mask]
                        can_remove = True
                        for unc_sub in uncovered_subs:
                            found = False
                            for other_b in new_blocks:
                                if np.any(block_t_indices[other_b] == unc_sub):
                                    found = True
                                    break
                            if not found:
                                can_remove = False
                                break
                        if can_remove:
                            improved = True
                            coverage_count[subs_in_block] -= 1
                            continue
                new_blocks.append(b_idx)
            current_blocks = new_blocks
            
        return current_blocks
    
    def local_search_polish(current_blocks, coverage_count, max_iterations=300):
        """Try to reduce block count via targeted swaps using inverted index."""
        if time.time() - start_time > time_limit - 5.0:
            return current_blocks, coverage_count
        
        best_blocks = list(current_blocks)
        best_coverage = coverage_count.copy()
        best_count = len(best_blocks)
        
        for iteration in range(max_iterations):
            if time.time() - start_time > time_limit - 5.0:
                break
            
            # First try simple deletion
            removed_any = False
            for i in range(len(best_blocks)):
                b_idx = best_blocks[i]
                subs_in_block = block_t_indices[b_idx]
                if np.all(best_coverage[subs_in_block] > 1):
                    best_blocks.pop(i)
                    best_coverage[subs_in_block] -= 1
                    removed_any = True
                    best_count -= 1
                    break
            
            if removed_any:
                continue
            
            # Try swap: find block with lowest unique contribution
            if len(best_blocks) == 0:
                break
                
            contributions = []
            for b_idx in best_blocks:
                subs = block_t_indices[b_idx]
                unique_count = np.sum(best_coverage[subs] == 1)
                contributions.append(unique_count)
            
            contributions = np.array(contributions)
            min_contrib = np.min(contributions)
            low_contrib_indices = np.where(contributions == min_contrib)[0]
            
            if len(low_contrib_indices) == 0:
                break
                
            # Try removing a low-contribution block and finding a replacement
            remove_idx = low_contrib_indices[random.randint(0, len(low_contrib_indices) - 1)]
            remove_b_idx = best_blocks[remove_idx]
            remove_subs = block_t_indices[remove_b_idx]
            
            temp_coverage = best_coverage.copy()
            temp_coverage[remove_subs] -= 1
            uncovered_mask = (temp_coverage == 0)
            uncovered_indices = np.where(uncovered_mask)[0]
            
            if len(uncovered_indices) == 0:
                # Can remove without adding
                best_blocks.pop(remove_idx)
                best_coverage = temp_coverage
                best_count -= 1
                continue
            
            # Find candidate blocks that cover all uncovered subsets
            if len(uncovered_indices) == 1:
                candidate_blocks = set(ts_to_blocks[uncovered_indices[0]])
            elif len(uncovered_indices) == 2:
                candidate_blocks = set(ts_to_blocks[uncovered_indices[0]]) & set(ts_to_blocks[uncovered_indices[1]])
            else:
                # For 3+ uncovered, start with smallest set and intersect
                sorted_unc = sorted(uncovered_indices, key=lambda x: len(ts_to_blocks[x]))
                candidate_blocks = set(ts_to_blocks[sorted_unc[0]])
                for unc in sorted_unc[1:]:
                    candidate_blocks &= set(ts_to_blocks[unc])
                    if len(candidate_blocks) == 0:
                        break
            
            # Remove existing blocks from candidates
            existing_set = set(best_blocks)
            candidate_blocks -= existing_set
            
            if len(candidate_blocks) > 0:
                # Pick a random candidate
                cand = random.choice(list(candidate_blocks))
                best_blocks[remove_idx] = cand
                best_coverage = temp_coverage
                best_coverage[block_t_indices[cand]] += 1
        
        return best_blocks, best_coverage
    
    def hypergraph_peeling(seed):
        """
        Hypergraph peeling algorithm:
        1. Start with all t-subsets uncovered
        2. Build residual hypergraph of uncovered triples
        3. Iteratively remove point with minimum degree
        4. Add precomputed block that maximally covers triples incident to that point
        """
        random.seed(seed)
        np.random.seed(seed)
        
        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts
        cover_freq = np.zeros(n_ts, dtype=np.int32)
        
        # Track degree of each point in the residual hypergraph
        # degree[p] = number of uncovered t-subsets containing point p
        point_degree = np.zeros(v, dtype=np.int32)
        
        # Initialize: for each t-subset, increment degree of each point in it
        for i, ts in enumerate(all_ts):
            for p in ts:
                point_degree[p] += 1
        
        # Precompute for each point, the blocks that contain it
        point_to_blocks = [set() for _ in range(v)]
        for i, block in enumerate(all_ks):
            for p in block:
                point_to_blocks[p].add(i)
        
        # Convert to lists for faster iteration
        point_to_blocks_list = [list(s) for s in point_to_blocks]
        
        # Main peeling loop
        while remaining > 0:
            if time.time() - start_time > time_limit - 5.0:
                return None, None
            
            # Find point with minimum positive degree
            min_deg = np.min(point_degree[point_degree > 0]) if np.any(point_degree > 0) else 0
            if min_deg == 0:
                # Fallback to standard greedy
                covered_counts = covered[block_t_indices].sum(axis=1)
                gains = t_count - covered_counts
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                max_gain_blocks = np.where(gains == max_gain)[0]
                if len(max_gain_blocks) > 30:
                    candidates = max_gain_blocks[random.sample(range(len(max_gain_blocks)), min(30, len(max_gain_blocks)))]
                else:
                    candidates = max_gain_blocks
                best_candidate = candidates[0]
                for cand in candidates:
                    subs = block_t_indices[cand]
                    newly_covered = ~covered[subs]
                    if np.any(newly_covered):
                        bonus = np.sum(1.0 / (1.0 + cover_freq[subs[newly_covered]]))
                    else:
                        bonus = 0.0
                    score_cand = max_gain + bonus * 0.1
                    if score_cand > -1:
                        best_candidate = cand
            else:
                # Find point with minimum degree
                min_deg_points = np.where(point_degree == min_deg)[0]
                # Among min-degree points, prefer one with more uncovered incident triples
                # that can be covered by a single block
                best_point = min_deg_points[0]
                best_point_score = -1
                for p in min_deg_points:
                    # Score: number of uncovered triples incident to p that can be covered together
                    incident_uncovered = []
                    for ts_idx in range(n_ts):
                        if not covered[ts_idx] and p in all_ts[ts_idx]:
                            incident_uncovered.append(ts_idx)
                    # Estimate coverage potential
                    score = len(incident_uncovered)
                    if score > best_point_score:
                        best_point_score = score
                        best_point = p
                
                # Find best block containing best_point that covers most uncovered triples
                candidate_blocks = point_to_blocks_list[best_point]
                if not candidate_blocks:
                    # Fallback
                    covered_counts = covered[block_t_indices].sum(axis=1)
                    gains = t_count - covered_counts
                    max_gain = np.max(gains)
                    max_gain_blocks = np.where(gains == max_gain)[0]
                    best_candidate = max_gain_blocks[0] if len(max_gain_blocks) > 0 else 0
                else:
                    best_candidate = candidate_blocks[0]
                    best_gain = -1
                    # Sample if too many candidates
                    if len(candidate_blocks) > 50:
                        sampled = random.sample(candidate_blocks, 50)
                    else:
                        sampled = candidate_blocks
                    
                    for cand in sampled:
                        subs = block_t_indices[cand]
                        newly_covered = ~covered[subs]
                        gain = np.sum(newly_covered)
                        # Bonus for covering rare subsets
                        if gain > 0:
                            bonus = np.sum(1.0 / (1.0 + cover_freq[subs[newly_covered]]))
                        else:
                            bonus = 0.0
                        total_score = gain + bonus * 0.1
                        if total_score > best_gain:
                            best_gain = total_score
                            best_candidate = cand
            
            # Add the chosen block
            blocks_indices.append(best_candidate)
            
            # Update coverage
            subs_to_cover = block_t_indices[best_candidate]
            newly_covered = ~covered[subs_to_cover]
            new_covers = np.sum(newly_covered)
            remaining -= new_covers
            cover_freq[subs_to_cover] += 1
            covered[subs_to_cover] = True
            
            # Update point degrees
            for ts_idx in range(n_ts):
                if newly_covered[ts_idx]:
                    for p in all_ts[ts_idx]:
                        point_degree[p] -= 1
        
        current_blocks = list(blocks_indices)
        
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
        
        current_blocks = aggressive_deletion(current_blocks, coverage_count)
        
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
        
        current_blocks, coverage_count = local_search_polish(current_blocks, coverage_count, max_iterations=200)
        
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
        current_blocks = aggressive_deletion(current_blocks, coverage_count)
        
        return current_blocks, len(current_blocks)
    
    def run_greedy(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts
        cover_freq = np.zeros(n_ts, dtype=np.int32)
        
        while remaining > 0:
            if time.time() - start_time > time_limit - 5.0:
                return None, None
            
            covered_counts = covered[block_t_indices].sum(axis=1)
            gains = t_count - covered_counts
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            max_gain_blocks = np.where(gains == max_gain)[0]
            
            if len(max_gain_blocks) > 30:
                candidates = max_gain_blocks[random.sample(range(len(max_gain_blocks)), min(30, len(max_gain_blocks)))]
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
        
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
        
        current_blocks = aggressive_deletion(current_blocks, coverage_count)
        
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
        
        current_blocks, coverage_count = local_search_polish(current_blocks, coverage_count, max_iterations=200)
        
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
        current_blocks = aggressive_deletion(current_blocks, coverage_count)
        
        return current_blocks, len(current_blocks)
    
    # Phase 1: Multiple greedy restarts
    restart_count = 0
    while time.time() - start_time < time_limit - 10.0:
        seed = restart_count
        if restart_count % 2 == 0:
            blocks, score = run_greedy(seed)
        else:
            blocks, score = hypergraph_peeling(seed)
        
        if blocks is None:
            break
            
        if score < best_score:
            best_score = score
            best_solution = blocks
            
        restart_count += 1
    
    if best_solution is None:
        blocks, score = run_greedy(0)
        if blocks is not None:
            best_solution = blocks
            best_score = score
        else:
            return []
    
    # Phase 2: Additional polish on best solution
    if time.time() - start_time < time_limit - 5.0:
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in best_solution:
            coverage_count[block_t_indices[b_idx]] += 1
        
        polished_blocks, polished_coverage = local_search_polish(
            list(best_solution), coverage_count, max_iterations=500
        )
        
        if len(polished_blocks) < best_score:
            best_solution = polished_blocks
            best_score = len(polished_blocks)
        
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in best_solution:
            coverage_count[block_t_indices[b_idx]] += 1
        best_solution = aggressive_deletion(list(best_solution), coverage_count)
    
    final_blocks = [list(all_ks[i]) for i in best_solution]
    
    return final_blocks
