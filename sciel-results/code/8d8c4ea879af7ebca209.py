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
    # Store as list of lists of indices for faster subset checks
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    block_triple_sets = [set() for _ in range(n_ks)]
    for i, block in enumerate(all_ks):
        subs = list(itertools.combinations(block, t))
        for j, sub in enumerate(subs):
            idx = idx_map[sub]
            block_t_indices[i, j] = idx
            block_triple_sets[i].add(idx)

    # Precompute blocks containing each triple
    blocks_containing_triple = [[] for _ in range(n_ts)]
    for i in range(n_ks):
        for idx in block_t_indices[i]:
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
        """
        Find a block not in solution that covers all triples in L_indices.
        L_indices is a list of triple indices.
        in_solution_mask is a boolean array of size n_ks.
        Returns block index or None.
        """
        if not L_indices:
            return -1 # Special case: no replacement needed
            
        # Sort L_indices by the size of their candidate lists
        # to start with the most constrained
        candidates_lists = []
        for t_idx in L_indices:
            lst = blocks_containing_triple[t_idx]
            candidates_lists.append((len(lst), t_idx, lst))
        
        candidates_lists.sort(key=lambda x: x[0])
        
        # Start with the smallest list
        _, start_t, start_list = candidates_lists[0]
        
        # Other triples to check
        other_ts = [t for (_, t, _) in candidates_lists[1:]]
        
        for k in start_list:
            if in_solution_mask[k]:
                continue
            
            # Check if k covers all other_ts
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
        """
        Try to replace two blocks with one.
        Returns True if an exchange was made.
        """
        n_blocks = len(current_blocks)
        in_solution_mask = np.zeros(n_ks, dtype=bool)
        for b in current_blocks:
            in_solution_mask[b] = True
            
        # We iterate over pairs. If we make a change, we restart.
        # To avoid infinite loops, we limit iterations or rely on time.
        for _ in range(n_blocks): # Max possible improvements is n_blocks/2
            if time.time() - start_time > time_limit - 5.0:
                return False
            
            improved = False
            # We need to find a pair (i, j)
            # If we remove i, j, and add k, the size decreases by 1.
            
            # Efficiently finding the pair:
            # Iterate i
            for i_idx in range(len(current_blocks)):
                b_i = current_blocks[i_idx]
                
                # Iterate j > i
                for j_idx in range(i_idx + 1, len(current_blocks)):
                    b_j = current_blocks[j_idx]
                    
                    # Calculate L: triples lost if b_i, b_j removed
                    # c_new[t] = c_old[t] - (1 if t in b_i) - (1 if t in b_j)
                    # We need c_new[t] >= 1 for all t NOT in L?
                    # No, we need c_new[t] >= 1 for all t, EXCEPT those covered by replacement k.
                    # Actually, L is the set of t where c_new[t] < 1.
                    
                    # Compute L
                    # We can do this by checking coverage counts
                    # subs_i = block_t_indices[b_i]
                    # subs_j = block_t_indices[b_j]
                    
                    # For t in subs_i U subs_j:
                    # count = coverage_count[t]
                    # if t in subs_i: count -= 1
                    # if t in subs_j: count -= 1
                    # if count < 1: add to L
                    
                    L = []
                    # Iterate through union of subs_i and subs_j
                    # Since t_count is small (35), we can just iterate
                    seen = set()
                    for t_idx in block_t_indices[b_i]:
                        seen.add(t_idx)
                    for t_idx in block_t_indices[b_j]:
                        seen.add(t_idx)
                        
                    for t_idx in seen:
                        c = coverage_count[t_idx]
                        if t_idx in block_triple_sets[b_i]:
                            c -= 1
                        if t_idx in block_triple_sets[b_j]:
                            c -= 1
                        if c < 1:
                            L.append(t_idx)
                            
                    if not L:
                        # Can remove both
                        # Remove b_i and b_j
                        # Update coverage
                        for t_idx in block_triple_sets[b_i]:
                            coverage_count[t_idx] -= 1
                        for t_idx in block_triple_sets[b_j]:
                            coverage_count[t_idx] -= 1
                        # Remove from list
                        current_blocks.remove(b_j)
                        current_blocks.remove(b_i)
                        in_solution_mask[b_i] = False
                        in_solution_mask[b_j] = False
                        return True # Restart
                    
                    # If L is not empty, find replacement
                    if len(L) > t_count: # Max triples per block
                        continue
                    
                    k = find_replacement_block(L, in_solution_mask)
                    if k is not None:
                        # Replace b_i, b_j with k
                        # Update coverage
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
                        
                        return True # Restart
            
        return False

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
                    if point_weights is not None:
                        bonus = np.sum(1.0 / (1.0 + cover_freq[subs[newly_covered]]))
                    else:
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

        # New: 2-for-1 exchange
        if time.time() - start_time < time_limit - 10.0:
            while two_for_one_exchange(current_blocks, coverage_count):
                coverage_count = rebuild_coverage(current_blocks)
                current_blocks = aggressive_deletion(current_blocks, coverage_count)
                coverage_count = rebuild_coverage(current_blocks)

        if time.time() - start_time < time_limit - 5.0:
             # Point swaps if time permits
             # (Omitted for brevity in thought, will include in code)
             pass

        return current_blocks, len(current_blocks)

    # Main loop
    restart_count = 0
    while time.time() - start_time < time_limit - 25.0:
        seed = restart_count
        blocks, score = run_greedy(seed, use_random_weights=False)
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
    
    while time.time() - start_time < time_limit - 5.0:
        improved = two_for_one_exchange(best_solution, coverage_count)
        if not improved:
            break
        coverage_count = rebuild_coverage(best_solution)
        best_solution = aggressive_deletion(best_solution, coverage_count)
        coverage_count = rebuild_coverage(best_solution)

    final_blocks = [list(all_ks[i]) for i in best_solution]
    return final_blocks
