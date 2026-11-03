"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
import scipy.sparse as sp
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline to be safe within 150s
    
    # Total number of words in the space
    total_words = q ** n
    
    # Convert words to indices and back
    def word_to_idx(word):
        idx = 0
        for v in word:
            idx = idx * q + v
        return idx
        
    def idx_to_word(idx):
        word = []
        for _ in range(n):
            word.append(idx % q)
            idx //= q
        return word[::-1]
    
    # Precompute coverage sets for all centers
    # center_covers[c_idx] = list of word indices covered by center c_idx
    # For radius=1, q=3, n=7: 1 + 7*2 = 15 neighbors
    
    center_covers = []
    for c_idx in range(total_words):
        c_word = idx_to_word(c_idx)
        covered = [c_idx]
        for pos in range(n):
            for new_val in range(q):
                if new_val != c_word[pos]:
                    new_word = c_word.copy()
                    new_word[pos] = new_val
                    new_idx = word_to_idx(new_word)
                    covered.append(new_idx)
        center_covers.append(covered)
    
    # Build sparse coverage matrix
    # coverage_matrix[c, w] = 1 if center c covers word w
    # Shape: (total_words, total_words)
    rows = []
    cols = []
    for c_idx in range(total_words):
        for w_idx in center_covers[c_idx]:
            rows.append(c_idx)
            cols.append(w_idx)
            
    coverage_matrix = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), 
        shape=(total_words, total_words)
    )
    
    # Greedy Set Cover
    # uncovered: boolean array of length total_words
    uncovered = np.ones(total_words, dtype=bool)
    selected = []
    selected_mask = np.zeros(total_words, dtype=bool)
    
    # To speed up greedy, we can use the fact that coverage is symmetric for this specific problem?
    # No, but we can use sparse matrix multiplication.
    
    while np.any(uncovered):
        if time.time() > deadline - 20:
            break
            
        # Calculate how many uncovered words each center covers
        # counts[c] = sum(coverage_matrix[c, w] for w where uncovered[w] is True)
        # This is equivalent to coverage_matrix @ uncovered_vector
        # Since uncovered is boolean, convert to float for dot product or use sum
        # sparse dot product is efficient
        uncovered_vec = uncovered.astype(np.float32)
        counts = coverage_matrix.dot(uncovered_vec)
        
        # Find center with max count
        if np.max(counts) == 0:
            break
            
        best_center = np.argmax(counts)
        selected.append(best_center)
        selected_mask[best_center] = True
        
        # Update uncovered: remove words covered by best_center
        # uncovered[center_covers[best_center]] = False
        # Vectorized update:
        # coverage_matrix[best_center, :] gives the row
        # We need to set uncovered[w] = False for all w where coverage_matrix[best_center, w] == 1
        # This is: uncovered &= ~coverage_matrix[best_center, :].toarray()
        # But .toarray() is dense. 
        # Better: get the non-zero indices of the row
        row = coverage_matrix.getrow(best_center)
        cols_to_uncover = row.indices
        uncovered[cols_to_uncover] = False
        
        # Optional: If we want to be smarter, we could stop if uncovered is empty
        
    # Check if fully covered, if not, add more (should not happen if loop finished correctly)
    if np.any(uncovered):
        # Fallback: pick centers for remaining uncovered words
        remaining_uncovered_indices = np.where(uncovered)[0]
        for w_idx in remaining_uncovered_indices:
            # Find a center that covers w_idx
            # centers covering w_idx are in column w_idx of coverage_matrix
            col = coverage_matrix.getcol(w_idx)
            candidate_centers = col.indices
            # Pick the first one not already selected
            for c in candidate_centers:
                if not selected_mask[c]:
                    selected.append(c)
                    selected_mask[c] = True
                    # Update uncovered
                    row = coverage_matrix.getrow(c)
                    uncovered[row.indices] = False
                    break
            # If all candidates were already selected, something is wrong, but logically one should be new or we missed it
            if selected_mask[best_center if selected else 0] == False and len(selected)==0:
                 pass # Safety

    # Local Search: Iterative Removal
    # Goal: Remove any center that is redundant.
    # A center c is redundant if for all w in center_covers[c], 
    # the number of selected centers covering w is >= 2.
    
    # Initialize coverage counts
    # coverage_counts[w] = number of selected centers covering w
    # coverage_counts = coverage_matrix.T @ selected_mask
    selected_vec = selected_mask.astype(np.float32)
    coverage_counts = coverage_matrix.T.dot(selected_vec)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Iterate through selected centers
        # We need to be careful with indices if we remove from the list
        # It's safer to iterate through a copy of the selected list or indices
        for i in range(len(selected) - 1, -1, -1):
            c_idx = selected[i]
            # Get words covered by c_idx
            covered_words = center_covers[c_idx]
            
            # Check if all covered words have count >= 2
            # coverage_counts[covered_words] gives the counts for these words
            if np.min(coverage_counts[covered_words]) >= 2:
                # Redundant, remove it
                selected.pop(i)
                selected_mask[c_idx] = False
                # Update coverage_counts
                # Decrement counts for all words covered by c_idx
                coverage_counts[covered_words] -= 1
                improved = True
                # Restart loop to check again, as removal might make others redundant
                # Breaking and restarting is O(N^2) but N is small (~200)
                break
        
    # Additional Polish: Try to swap?
    # Given time constraints, iterative removal is the most robust improvement over the seed.
    # The seed's local search was very weak.
    
    # Convert selected indices to words
    result = [idx_to_word(c_idx) for c_idx in selected]
    
    # Ensure distinctness
    seen = set()
    unique_result = []
    for word in result:
        key = tuple(word)
        if key not in seen:
            seen.add(key)
            unique_result.append(word)
            
    return unique_result
# EVOLVE-BLOCK-END
