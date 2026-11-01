"""Combined program for ternary covering code K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Generate all words in the space
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    total_words = len(all_words)

    # Generate candidate codewords using tetracode direct-sum construction
    # This gives 9 * 3^(n-4) candidates, much fewer than all_words
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        candidates = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        candidates = np.array(candidates, dtype=np.int8)
        num_candidates = len(candidates)
    else:
        # Fallback: use all words as candidates
        candidates = all_words
        num_candidates = total_words

    # Precompute coverage matrix: coverage[i, j] = True if candidates[i] covers all_words[j]
    # Use int32 for distance calculation to avoid overflow
    coverage = np.zeros((num_candidates, total_words), dtype=bool)
    
    # Vectorized distance calculation in chunks
    chunk_size = 256
    for i_start in range(0, num_candidates, chunk_size):
        i_end = min(i_start + chunk_size, num_candidates)
        # candidates[i_start:i_end] shape (C, n)
        # all_words shape (N, n)
        diff = candidates[i_start:i_end, None, :].astype(np.int32) != all_words[None, :, :].astype(np.int32)
        dists = np.sum(diff, axis=2)
        coverage[i_start:i_end, :] = dists <= radius

    # Greedy Set Cover using BLAS-accelerated dot product
    remaining = np.ones(total_words, dtype=np.int8)
    remaining_count = total_words
    selected_indices = []
    
    while remaining_count > 0:
        if time.time() > deadline - 10.0:
            break
            
        # Calculate coverage count for all candidates against remaining words
        # coverage shape (M, N), remaining shape (N,)
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
        
    # Convert indices to codewords
    selected_codewords = [candidates[i].tolist() for i in selected_indices]
    
    # Post-processing: Remove redundant codewords and apply swap-based local search
    if time.time() < deadline - 5.0 and len(selected_indices) > 0:
        K = len(selected_indices)
        sel_cov = coverage[selected_indices, :]
        
        # Count how many selected codewords cover each word
        local_cover_counts = np.sum(sel_cov, axis=0)
        
        # Iteratively remove redundant codewords
        removed = [False] * K
        
        for _ in range(K):
            if time.time() > deadline - 10.0:
                break
                
            to_remove = []
            for i in range(K):
                if removed[i]:
                    continue
                # Check if i covers any word with count 1
                if np.any(sel_cov[i] & (local_cover_counts == 1)):
                    continue
                to_remove.append(i)
                
            if not to_remove:
                break
                
            for i in to_remove:
                removed[i] = True
                local_cover_counts -= sel_cov[i]
                
        # Final list of codewords after deletion
        final_indices = [i for i in range(K) if not removed[i]]
        selected_codewords = [candidates[selected_indices[i]].tolist() for i in final_indices]
        
        # Swap-based local search: try replacing one selected codeword with another
        # to reduce the total number of codewords while maintaining coverage
        if time.time() < deadline - 2.0 and len(final_indices) > 0:
            K2 = len(final_indices)
            # Rebuild sel_cov for the current set
            sel_cov2 = np.array([coverage[selected_indices[fi]] for fi in final_indices])
            cover_counts = np.sum(sel_cov2, axis=0)
            
            # Try to find swaps that reduce size
            improved = True
            while improved and time.time() < deadline - 2.0:
                improved = False
                for i in range(K2):
                    if time.time() > deadline - 2.0:
                        break
                    # Check if removing i leaves some words uncovered
                    uncovered = sel_cov2[i] & (cover_counts == 1)
                    if not np.any(uncovered):
                        # Can remove i entirely
                        K2 -= 1
                        sel_cov2 = np.delete(sel_cov2, i, axis=0)
                        cover_counts -= sel_cov2[i] if i < K2 else np.zeros(total_words, dtype=np.int8)
                        # Recalculate cover counts properly
                        cover_counts = np.sum(sel_cov2, axis=0)
                        final_indices.pop(i)
                        improved = True
                        break
                    else:
                        # Try to find a replacement that covers all uncovered words
                        # and has minimal additional coverage
                        # This is expensive, so we limit the search
                        if K2 > 50:  # Only do this for smaller sets
                            continue
                        
                        # For each uncovered word, find candidates that cover it
                        # and check if they cover all uncovered words
                        uncovered_indices = np.where(uncovered)[0]
                        
                        # Find all candidates that cover all uncovered words
                        # candidates[j] covers uncovered word w if coverage[j, w] is True
                        # We need candidates that cover ALL uncovered words
                        # This is a set intersection problem
                        # For small K2, we can try each candidate
                        best_replace = -1
                        best_extra = 0
                        for j in range(num_candidates):
                            if time.time() > deadline - 2.0:
                                break
                            # Skip if j is already in the set
                            if j in selected_indices[final_indices]:
                                continue
                            
                            # Check if candidate j covers all uncovered words
                            if np.all(coverage[j, uncovered_indices]):
                                # Calculate extra coverage (words covered by j but not by i)
                                extra = np.sum(coverage[j] & ~sel_cov2[i] & (cover_counts == 0))
                                if extra > best_extra:
                                    best_extra = extra
                                    best_replace = j
                        
                        if best_replace >= 0:
                            # Perform the swap
                            # Remove i's coverage, add best_replace's coverage
                            cover_counts = np.sum(sel_cov2, axis=0)  # Recalculate
                            sel_cov2[i] = coverage[best_replace]
                            final_indices[i] = best_replace
                            cover_counts = np.sum(sel_cov2, axis=0)
                            improved = True
                            break
            
            # Update selected_codewords
            selected_codewords = [candidates[selected_indices[fi]].tolist() for fi in final_indices]
        
    return selected_codewords
# EVOLVE-BLOCK-END
