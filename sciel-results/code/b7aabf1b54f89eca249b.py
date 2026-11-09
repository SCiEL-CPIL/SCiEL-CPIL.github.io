import itertools
import numpy as np
import time

def construct(q, n, radius):
    # We assume q=3, n=7, radius=1 for this specific task, 
    # but implement a generic vectorized greedy set-cover.
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Total number of words in the space
    total_words = q ** n
    
    # Generate all words in the space as a numpy array of shape (total_words, n)
    # We can generate them efficiently using base-q representation.
    # For q=3, n=7, total_words = 2187. This is small enough to hold in memory.
    all_words = np.array([list(itertools.product(range(q), repeat=n))], dtype=np.uint8)
    
    # Precompute the "balls" for each word.
    # ball[i] is a boolean mask of length total_words, True if word j is within radius of word i.
    # Storing 2187 * 2187 booleans is ~4.7 MB, which is fine.
    balls = np.zeros((total_words, total_words), dtype=bool)
    
    # Compute distances efficiently using broadcasting? 
    # all_words has shape (N, n). 
    # dist[i, j] = sum(all_words[i] != all_words[j])
    # For N=2187, N*N*n operations is ~31 million, which is fast in numpy.
    
    # Let's compute the distance matrix or just the ball masks directly.
    # A word j is in ball(i) if Hamming distance <= radius.
    
    # We can iterate over i and compute the mask for i.
    # Or, we can compute the full distance matrix.
    # D = np.sum(all_words[:, np.newaxis, :] != all_words[np.newaxis, :, :], axis=2)
    # This creates a (N, N, n) array which is 2187*2187*7 bytes ~ 33 MB. Acceptable.
    
    D = np.sum(all_words[:, np.newaxis, :] != all_words[np.newaxis, :, :], axis=2)
    balls = D <= radius
    
    # Greedy Set Cover
    # covered_mask: boolean array of length N, initially all False
    covered_mask = np.zeros(total_words, dtype=bool)
    selected_indices = []
    
    # To speed up greedy selection, we can maintain a count of how many uncovered words each candidate covers.
    # However, recalculating this every step is O(N^2). 
    # With N=2187 and potentially ~200 steps, 200 * 2187 * 2187 ~ 950 million ops. 
    # In numpy, this might be slow if done in a python loop.
    # Let's try to optimize the selection step.
    
    # Alternative: 
    # At each step, find the index i that maximizes sum(balls[i] & ~covered_mask).
    # We can maintain an array `uncovered_counts` where uncovered_counts[i] = sum(balls[i] & ~covered_mask).
    # Initially, uncovered_counts[i] = sum(balls[i]) = 1 + n*(q-1) for radius 1.
    # When we select index k, we update covered_mask.
    # For every i, uncovered_counts[i] decreases by sum(balls[i] & balls[k] & ~old_covered_mask).
    # This is still heavy.
    
    # Given N=2187 is small, a simpler approach might be fast enough with numpy vectorization inside the loop.
    # Let's estimate: 
    # Loop runs ~200 times.
    # Inside loop:
    #   1. Calculate gain for all i: gain = balls[:, ~covered_mask].sum(axis=1)
    #      This involves indexing balls with a boolean mask. 
    #      Actually, balls[:, ~covered_mask] creates a view or copy? 
    #      If ~covered_mask has ~2000 True values, this is a (2187, 2000) boolean array.
    #      Summing over axis 1 is fast.
    #   2. Find argmax.
    #   3. Update covered_mask |= balls[k]
    #      This is a vectorized OR operation.
    
    # Let's implement this simple vectorized greedy.
    
    while not np.all(covered_mask):
        if time.time() > deadline:
            break
        
        # Calculate how many new words each candidate covers
        # uncovered_words = ~covered_mask
        # gain[i] = number of True in balls[i, uncovered_words]
        # We can do: balls[:, ~covered_mask].sum(axis=1)
        
        # Optimization: if we maintain the list of uncovered indices, it gets smaller.
        # But boolean indexing in numpy is generally fast.
        
        gains = balls[:, ~covered_mask].sum(axis=1)
        
        # Select the best candidate
        best_idx = np.argmax(gains)
        
        # If no gain, we are stuck (should not happen for radius 1)
        if gains[best_idx] == 0:
            break
            
        selected_indices.append(best_idx)
        covered_mask |= balls[best_idx]
        
        # Early exit if we have covered everything
        if np.all(covered_mask):
            break

    # Convert selected indices to codewords
    # selected_indices are integers 0..total_words-1
    # We need to map them back to the word lists.
    # all_words is already the list of words.
    
    # Ensure distinctness (they are distinct by index)
    codewords = [list(all_words[i]) for i in selected_indices]
    
    return codewords
