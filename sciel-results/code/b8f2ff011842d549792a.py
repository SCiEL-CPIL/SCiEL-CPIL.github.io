"""Combined program for ternary covering code K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # Generate all words in the space
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    total_words = len(all_words)
    
    # Precompute coverage masks for all candidates
    coverage = np.zeros((total_words, total_words), dtype=bool)
    
    # Chunked distance computation to manage memory
    chunk_size = 500
    for i_start in range(0, total_words, chunk_size):
        i_end = min(i_start + chunk_size, total_words)
        diff = all_words[i_start:i_end, None, :] != all_words[None, :, :]
        dists = np.sum(diff, axis=2)
        coverage[i_start:i_end, :] = dists <= radius

    # Greedy Set Cover with improved strategy
    remaining = np.ones(total_words, dtype=bool)
    remaining_count = total_words
    selected_indices = []
    
    # Use int8 for faster dot product
    remaining_int8 = remaining.astype(np.int8)
    
    while remaining_count > 0:
        if time.time() > deadline - 15.0:
            break
            
        # Calculate coverage count for all candidates against remaining words
        counts = coverage.dot(remaining_int8)
        
        # Find candidate with max count
        best_idx = np.argmax(counts)
        best_count = counts[best_idx]
        
        if best_count == 0:
            break
            
        selected_indices.append(best_idx)
        
        # Update remaining
        remaining &= ~coverage[best_idx]
        remaining_int8 = remaining.astype(np.int8)
        remaining_count = np.count_nonzero(remaining)
        
    # Convert indices to codewords
    selected_codewords = [all_words[i].tolist() for i in selected_indices]
    
    # Post-processing: Remove redundant codewords
    if time.time() < deadline - 10.0:
        K = len(selected_indices)
        if K > 0:
            sel_cov = coverage[selected_indices, :]
            
            local_cover_counts = np.sum(sel_cov, axis=0)
            removed = [False] * K
            
            # Iterative removal of redundant codewords
            for _ in range(K):
                if time.time() > deadline - 3.0:
                    break
                
                to_remove = []
                for i in range(K):
                    if removed[i]:
                        continue
                    # Check if i covers any word with count 1 (essential)
                    if np.any(sel_cov[i] & (local_cover_counts == 1)):
                        continue
                    to_remove.append(i)
                
                if not to_remove:
                    break
                
                for i in to_remove:
                    removed[i] = True
                    local_cover_counts -= sel_cov[i]
                
            final_indices = [i for i in range(K) if not removed[i]]
            selected_codewords = [all_words[selected_indices[i]].tolist() for i in final_indices]
            
            # Additional local search: try swapping codewords to reduce size
            # Try to replace pairs of codewords with a single codeword if possible
            if time.time() < deadline - 5.0 and len(selected_codewords) > 2:
                K2 = len(selected_codewords)
                sel_cov2 = np.array([coverage[all_words.tolist().index(cw)] if cw in [all_words[i].tolist() for i in selected_indices] else coverage[0] for cw in selected_codewords])
                
                # Simpler: work with indices
                active_idx = [selected_indices[i] for i in final_indices]
                
                # Try removing one codeword and re-covering with another
                for attempt in range(K2):
                    if time.time() > deadline - 2.0:
                        break
                    
                    # Try to remove codeword at position 'attempt'
                    test_idx = active_idx[attempt]
                    words_to_cover = coverage[test_idx]
                    
                    # Check if all words covered by test_idx are covered by other active codewords
                    other_cov = np.sum([coverage[idx] for idx in active_idx if idx != test_idx], axis=0)
                    
                    if np.all(other_cov[words_to_cover] > 0):
                        # Can remove test_idx
                        active_idx = [idx for idx in active_idx if idx != test_idx]
                        K2 -= 1
                        attempt -= 1  # Re-check same position
                    
                    if K2 <= 0:
                        break
                
                selected_codewords = [all_words[i].tolist() for i in active_idx]
    
    return selected_codewords
# EVOLVE-BLOCK-END
