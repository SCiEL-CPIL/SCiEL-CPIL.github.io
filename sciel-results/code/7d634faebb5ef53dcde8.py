import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Generate all words in the space
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    total_words = len(all_words)
    
    # For the specific case q=3, n=7, radius=1, we can use the full space as candidates
    # because 2187 is small enough to handle the coverage matrix and local search efficiently.
    candidates = all_words
    num_candidates = len(candidates)
    
    # Precompute coverage matrix
    # coverage[i, j] = True if candidates[i] covers all_words[j]
    # dist <= radius
    # Using broadcasting for vectorized distance calculation
    # diff shape: (num_candidates, total_words, n)
    # This might be memory intensive for larger n, but for n=7, it's 2187*2187*7 ~ 33M bytes. Acceptable.
    
    # To save memory and time, we can compute in chunks if needed, but for 2187x2187 it's fine.
    # Let's verify memory: 2187 * 2187 * 7 * 1 byte (if int8) = ~33 MB. 
    # But we need boolean or int for sum. Let's use int8 for diff and sum to int16.
    
    # Actually, simpler:
    # dists[i, j] = sum(all_words[j] != candidates[i])
    
    # We will compute the coverage matrix in chunks to avoid creating a huge intermediate array if necessary,
    # but 33MB is fine.
    
    coverage = np.zeros((num_candidates, total_words), dtype=bool)
    
    # Chunked computation to be safe
    chunk_size = 256
    for i_start in range(0, num_candidates, chunk_size):
        i_end = min(i_start + chunk_size, num_candidates)
        # candidates[i_start:i_end, None, :] shape (C, 1, n)
        # all_words[None, :, :] shape (1, N, n)
        # diff shape (C, N, n)
        diff = candidates[i_start:i_end, None, :].astype(np.int8) != all_words[None, :, :].astype(np.int8)
        dists = np.sum(diff, axis=2, dtype=np.int16)
        coverage[i_start:i_end, :] = dists <= radius
        
    # Greedy Set Cover to get initial solution
    remaining = np.ones(total_words, dtype=np.int8)
    remaining_count = total_words
    selected_indices = []
    
    while remaining_count > 0:
        if time.time() > deadline - 20.0:
            break
            
        # Calculate coverage count for all candidates against remaining words
        # counts[i] = sum of coverage[i, j] where remaining[j] == 1
        counts = coverage.dot(remaining)
        
        # Find candidate with max count
        best_idx = np.argmax(counts)
        best_count = counts[best_idx]
        
        if best_count == 0:
            break
            
        selected_indices.append(best_idx)
        
        # Update remaining
        remaining[coverage[best_idx]] = 0
        remaining_count = np.count_nonzero(remaining)
        
    # Initial solution
    current_selected = np.array(selected_indices)
    current_size = len(current_selected)
    
    # Local Search to improve the solution
    # We maintain a boolean array 'in_set' of size num_candidates
    in_set = np.zeros(num_candidates, dtype=bool)
    in_set[current_selected] = True
    
    # cover_counts[j] = number of selected codewords that cover word j
    cover_counts = np.zeros(total_words, dtype=np.int16)
    for idx in current_selected:
        cover_counts[coverage[idx]] += 1
        
    best_size = current_size
    
    # Helper to check if we can remove idx
    # idx can be removed if for all j where coverage[idx, j] is True, cover_counts[j] > 1
    # i.e., there is no j such that coverage[idx, j] is True and cover_counts[j] == 1
    
    # We iterate:
    # 1. Try to remove any redundant codeword.
    # 2. If no removal possible, try to swap a codeword i with j not in set.
    #    A swap is beneficial if it allows future removals. 
    #    Heuristic: Replace i with j if j covers all "essential" words of i (words covered only by i)
    #    and j has higher total coverage of remaining words or similar.
    #    Actually, a simpler effective heuristic for small instances:
    #    Try all pairs (i in set, j not in set).
    #    If replacing i with j keeps coverage valid, and results in a set from which
    #    we can remove at least one codeword, do it.
    #    Even better: If replacing i with j allows the set size to decrease (by enabling removals), do it.
    
    # Given the small size (186 codewords, 2187 candidates), checking all swaps is O(K * (N-K) * N) which is too slow.
    # We need a smarter heuristic.
    
    # Strategy:
    # 1. Identify "critical" codewords: those covering at least one word with cover_count == 1.
    # 2. Non-critical codewords can be removed immediately.
    # 3. For critical codewords, we look for replacements.
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # Phase 1: Remove redundant codewords
        to_remove = []
        for i in range(num_candidates):
            if not in_set[i]:
                continue
            # Check if i covers any word with cover_count == 1
            # words_covered_by_i = np.where(coverage[i])[0]
            # if np.any(cover_counts[words_covered_by_i] == 1):
            #     continue
            # else:
            #     to_remove.append(i)
            
            # Vectorized check:
            # mask = coverage[i] & (cover_counts == 1)
            # if not np.any(mask):
            #     to_remove.append(i)
            
            if not np.any(coverage[i] & (cover_counts == 1)):
                to_remove.append(i)
                
        if to_remove:
            for i in to_remove:
                in_set[i] = False
                cover_counts[coverage[i]] -= 1
            best_size = np.sum(in_set)
            improved = True
            continue # Restart loop as cover_counts changed significantly
            
        # Phase 2: Try swaps
        # We only consider swapping "critical" codewords.
        # A codeword i is critical if it covers at least one word with count 1.
        critical_indices = []
        for i in range(num_candidates):
            if in_set[i]:
                if np.any(coverage[i] & (cover_counts == 1)):
                    critical_indices.append(i)
                    
        # If no critical indices, we are in a state where we can remove someone? 
        # No, if no critical indices, Phase 1 would have removed them.
        # So if we are here, all in_set codewords are critical.
        
        # To avoid O(K^2 * N) complexity, we limit the search.
        # We pick a random subset of critical codewords to try swapping.
        # Or we iterate through them.
        
        # Let's try a specific heuristic:
        # For each critical codeword i, find a codeword j not in set that covers all essential words of i.
        # Essential words of i = E_i = { w | coverage[i, w] and cover_counts[w] == 1 }
        # If we replace i with j, we must ensure j covers E_i.
        # Then we re-evaluate the set. If the new set allows removals, we accept.
        
        # To speed up, we precompute for each word w, the list of codewords that cover it.
        # covers[w] = list of i such that coverage[i, w]
        # This is the transpose of coverage.
        # Since coverage is (2187, 2187), covers is also (2187, 2187).
        
        # We can precompute the intersection of covers for the essential words.
        
        # Let's try a limited number of swaps per iteration to stay within time.
        # We will try to swap the first few critical codewords.
        
        swap_found = False
        # Shuffle critical indices to avoid bias? 
        # For now, iterate.
        
        for i in critical_indices:
            if time.time() > deadline - 5.0:
                break
                
            # Find essential words for i
            essential_words = np.where(coverage[i] & (cover_counts == 1))[0]
            
            if len(essential_words) == 0:
                continue # Should not happen if i is critical
                
            # Find candidates j not in set that cover ALL essential words
            # candidates_j = []
            # for j in range(num_candidates):
            #    if not in_set[j]:
            #       if np.all(coverage[j, essential_words]):
            #          candidates_j.append(j)
            
            # Vectorized:
            # mask_j = coverage[:, essential_words].all(axis=1) & ~in_set
            # potential_j = np.where(mask_j)[0]
            
            potential_j = np.where(coverage[:, essential_words].all(axis=1) & ~in_set)[0]
            
            if len(potential_j) == 0:
                continue
                
            # Evaluate potential swaps
            # We want a swap that minimizes the number of new "essential" words created,
            # or maximizes the number of redundant codewords created.
            
            best_j = -1
            best_score = -1
            
            # Limit evaluation to top few candidates by total coverage
            # Sort potential_j by coverage count (descending)
            cov_counts_potential = coverage[potential_j].dot(remaining) # Wait, remaining is not updated in local search
            # Use total coverage count of j
            total_cov = np.sum(coverage[potential_j], axis=1)
            sorted_idx = np.argsort(-total_cov)
            
            # Only evaluate top 50 candidates to save time
            for idx in sorted_idx[:50]:
                j = potential_j[idx]
                
                # Simulate swap: remove i, add j
                # new_cover_counts = cover_counts - coverage[i] + coverage[j]
                # Check validity: all words must have new_cover_counts >= 1
                # Actually, we already know j covers essential words of i.
                # We need to ensure that words covered ONLY by i are covered by j. (Done by essential check)
                # We need to ensure that words covered by i and others are still covered.
                # If a word w was covered by i and k (count 2), removing i makes count 1. Still valid.
                # If a word w was covered ONLY by i (count 1), j must cover it. (Done)
                # So the swap is always valid if j covers essential words of i.
                
                # Now, does this swap allow removals?
                # Calculate new cover counts
                new_cover_counts = cover_counts.copy()
                new_cover_counts[coverage[i]] -= 1
                new_cover_counts[coverage[j]] += 1
                
                # Check if any codeword in the new set (excluding j, including i's neighbors) can be removed.
                # A codeword k can be removed if it doesn't cover any word with new_cover_counts == 1.
                
                # We don't want to check ALL codewords.
                # We check if j itself is redundant? No, j is new.
                # We check if removing i and adding j makes some OTHER codeword k redundant.
                # Or simply, does the new set have a smaller "greedy potential"?
                
                # Heuristic: Score = - (number of words with new_cover_counts == 1)
                # Fewer essential words is better.
                num_essentials = np.sum(new_cover_counts == 1)
                
                # Also prefer j with high total coverage
                score = -num_essentials + 0.1 * total_cov[idx]
                
                if score > best_score:
                    best_score = score
                    best_j = j
                    
            if best_j != -1:
                # Perform swap
                in_set[i] = False
                in_set[best_j] = True
                cover_counts[coverage[i]] -= 1
                cover_counts[coverage[j]] += 1
                improved = True
                swap_found = True
                break # Restart Phase 1 with new configuration
                
        if not swap_found:
            # No simple swap found. 
            # We might be stuck in a local minimum.
            # To escape, we could do a random perturbation, but given time limits,
            # we might just stop.
            # However, the greedy solution is usually close.
            # Let's try one more thing: random restart of local search? No, too expensive.
            break

    # Extract final codewords
    final_indices = np.where(in_set)[0]
    selected_codewords = [candidates[i].tolist() for i in final_indices]
    
    return selected_codewords
