import itertools
import numpy as np

def construct(q, n, radius):
    # Generate all possible words
    all_words = np.array(list(itertools.product(range(q), repeat=n)))
    num_words = len(all_words)
    
    # Pre-compute the ball for each word (words within Hamming distance 'radius')
    # is_in_ball[i, j] is True if word j is within distance 'radius' of word i.
    
    a = all_words[:, None, :]
    b = all_words[None, :, :]
    dist = np.sum(a != b, axis=2)
    
    is_in_ball = dist <= radius
    
    covered = np.zeros(num_words, dtype=bool)
    code = []
    
    while covered.sum() < num_words:
        uncovered_mask = ~covered
        uncovered_indices = np.where(uncovered_mask)[0]
        
        if len(uncovered_indices) == 0:
            break
            
        # Restrict candidates to uncovered words.
        # We want to find the candidate c in uncovered_indices that covers the most
        # words in uncovered_indices.
        
        # Extract the submatrix of is_in_ball for candidates x targets both in uncovered_indices
        # sub_matrix[i, j] is True if uncovered_indices[j] is in ball of uncovered_indices[i]
        sub_matrix = is_in_ball[np.ix_(uncovered_indices, uncovered_indices)]
        
        # Sum along axis 1 to get the count of uncovered words covered by each candidate
        gains = np.sum(sub_matrix, axis=1)
        
        # Pick the candidate with maximum gain
        best_local_idx = np.argmax(gains)
        best_global_idx = uncovered_indices[best_local_idx]
        
        # If gain is 0, we are stuck (should not happen if covered.sum() < N)
        if gains[best_local_idx] == 0:
             break
            
        code.append(all_words[best_global_idx].tolist())
        
        # Update covered
        covered |= is_in_ball[best_global_idx]
        
    return code
