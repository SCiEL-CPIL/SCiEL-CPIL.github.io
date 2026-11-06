# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin

    total_words = q ** n
    
    # Precompute all words as indices
    # Map index to word and word to index
    # We will work with indices 0..total_words-1
    
    # Precompute neighbors for each word (the ball of radius 'radius')
    # neighbors[i] is a list of indices covered by codeword i
    # For q=3, n=8, radius=1, ball size is 1 + 8*2 = 17
    
    # Generate all words
    words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    
    # Function to get neighbors of an index
    # We can compute this on the fly or precompute.
    # Precomputing 6561 lists of 17 ints is fast.
    
    def get_neighbors(idx):
        w = words[idx]
        neigh = [idx]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                # Create new word
                # To avoid creating new lists, we can compute the index directly
                # Or just use the words array and find the index?
                # Better: compute index directly.
                # Index is base-q representation.
                # Let's precompute powers of q.
                pass
        return neigh

    # More efficient neighbor generation using index arithmetic
    # idx = sum(w[i] * q^(n-1-i))
    # Changing w[pos] to v changes idx by (v - w[pos]) * q^(n-1-pos)
    
    q_powers = [q ** (n - 1 - i) for i in range(n)]
    
    def get_neighbor_indices(idx):
        # Get word from index
        w = [0] * n
        temp = idx
        for i in range(n - 1, -1, -1):
            w[i] = temp % q
            temp //= q
        
        neigh = [idx]
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                delta = (new_val - orig) * q_powers[pos]
                neigh.append(idx + delta)
        return neigh

    # Precompute neighbors for all words
    # This takes a bit of time but is worth it
    all_neighbors = [get_neighbor_indices(i) for i in range(total_words)]
    
    # Greedy Set Cover
    # uncovered: boolean array of size total_words
    uncovered = np.ones(total_words, dtype=bool)
    code = []
    
    # To speed up "find best center", we can iterate through all candidates
    # and count how many neighbors are uncovered.
    # We can vectorize this?
    # For each candidate c, count = sum(uncovered[neighbors[c]])
    # This is hard to vectorize over candidates simultaneously without a matrix.
    # A 6561 x 17 matrix of neighbor indices is small.
    
    # Let's create a matrix 'neighbor_matrix' of shape (total_words, max_ball_size)
    # But ball sizes are constant for radius 1 in this case?
    # Yes, 1 + n*(q-1) is constant.
    ball_size = 1 + n * (q - 1)
    
    # Build neighbor matrix
    # If some balls are smaller (they shouldn't be for uniform radius in full space), pad with -1
    neighbor_matrix = np.full((total_words, ball_size), -1, dtype=np.int32)
    for i in range(total_words):
        neigh = all_neighbors[i]
        neighbor_matrix[i, :len(neigh)] = neigh
    
    # Mask for valid neighbors
    valid_mask = neighbor_matrix != -1
    
    # Greedy loop
    while uncovered.any():
        if time.time() > deadline - 5:
            break
            
        # For each candidate, compute number of uncovered neighbors
        # uncovered is bool array.
        # We want: for each row i, sum(uncovered[neighbor_matrix[i, :]])
        
        # We can use advanced indexing:
        # idx_to_fetch = neighbor_matrix[valid_mask]
        # This gives a flat array of indices.
        # We need to sum per row.
        
        # Alternative:
        # Since ball_size is small (17), we can loop through candidates in Python?
        # 6561 candidates * 17 ops = ~110k ops.
        # If we do this in a Python loop, it's slow.
        # Using NumPy:
        
        # Fetch all neighbor indices
        flat_neigh = neighbor_matrix[valid_mask]
        # Get uncovered status for these neighbors
        # This creates an array of bools
        unc_status = uncovered[flat_neigh]
        
        # We need to sum these per original row.
        # We can use np.bincount or similar, but we need to map back to rows.
        # The number of valid entries is total_words * ball_size (since all balls are full size here).
        # So we can reshape unc_status to (total_words, ball_size) and sum along axis 1.
        
        # Check if all balls are same size
        if np.all(valid_mask):
            unc_status_reshaped = unc_status.reshape(total_words, ball_size)
            counts = unc_status_reshaped.sum(axis=1)
        else:
            # Fallback for variable size (not expected here)
            counts = np.zeros(total_words, dtype=np.int32)
            # This would be slower, but let's assume uniform for now.
            # For q=3, n=8, r=1, it is uniform.
            pass

        # Find candidate with max count
        best_idx = np.argmax(counts)
        
        # If count is 0, we are stuck (shouldn't happen if uncovered is not empty and we cover all)
        if counts[best_idx] == 0:
            break
            
        code.append(best_idx)
        
        # Update uncovered: remove all neighbors of best_idx
        # neighbors of best_idx are neighbor_matrix[best_idx, :]
        # Filter out -1 if any
        best_neigh = neighbor_matrix[best_idx, :]
        best_neigh = best_neigh[best_neigh != -1]
        uncovered[best_neigh] = False
        
    # Convert indices back to words
    result_words = [words[i].tolist() for i in code]
    
    return result_words
# EVOLVE-BLOCK-END
