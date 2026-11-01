"""Greedy set-cover for ternary covering code K_3(6,1) using NumPy vectorization."""
# EVOLVE-BLOCK-START
import numpy as np


def construct(q, n, radius):
    """
    Construct a covering code using a greedy set-cover heuristic.
    
    Args:
        q: Alphabet size
        n: Length of codewords
        radius: Covering radius
    
    Returns:
        List of codewords (each is a list of integers)
    """
    total = q ** n
    
    # Generate all words as a 2D numpy array
    # Using base-q expansion
    indices = np.arange(total)
    words = np.zeros((total, n), dtype=np.uint8)
    for i in range(n):
        words[:, i] = (indices // (q ** i)) % q
        
    # Precompute the coverage mask: cover_mask[i, j] is True if words[j] is within Hamming distance <= radius of words[i]
    # For efficiency, we can compute Hamming distances for all pairs or just use the structure.
    # Given N=729, N^2 = 531,441. A vectorized Hamming distance calculation is feasible.
    
    # Compute pairwise Hamming distances
    # words shape: (N, n)
    # We want dist[i, j] = sum(words[i] != words[j])
    # Broadcasting: words[:, :, None] != words[None, :, :] gives (N, N, n) boolean array
    # This might be memory heavy: 729 * 729 * 6 * 1 byte = ~3MB. This is fine.
    
    # However, we only need distance <= radius.
    # Let's compute the full distance matrix to be safe and simple, or optimize if needed.
    # Actually, for radius=1, we can be smarter, but N is small enough for full matrix.
    
    # dist_matrix[i, j] = Hamming distance between words[i] and words[j]
    # Using broadcasting
    # words_expanded1 = words[:, np.newaxis, :] # (N, 1, n)
    # words_expanded2 = words[np.newaxis, :, :] # (1, N, n)
    # dist_matrix = np.sum(words_expanded1 != words_expanded2, axis=2)
    
    # Let's do it in chunks to avoid massive memory spikes if N was larger, but N=729 is small.
    # dist_matrix = np.zeros((total, total), dtype=np.int8)
    # for i in range(total):
    #     dist_matrix[i, :] = np.sum(words[i] != words, axis=1)
    
    # Vectorized way:
    # This creates a (729, 729, 6) array. 729*729*6 = 3,188,646 bytes. ~3MB.
    # This is perfectly fine.
    dist_matrix = np.sum(words[:, np.newaxis, :] != words[np.newaxis, :, :], axis=2)
    
    # Coverage mask: True if dist <= radius
    cover_mask = dist_matrix <= radius
    
    # Greedy Set Cover
    uncovered = np.ones(total, dtype=bool)
    codewords = []
    
    while np.any(uncovered):
        # For each candidate i, count how many currently uncovered words it covers
        # cover_mask[:, uncovered] gives a submatrix of shape (total, num_uncovered)
        # We sum along axis=1 to get the count for each candidate
        counts = cover_mask[:, uncovered].sum(axis=1)
        
        # Find the candidate with the maximum coverage
        best_c = np.argmax(counts)
        
        # If no word is covered (should not happen if uncovered is not empty and space is covered), break
        if counts[best_c] == 0:
            break
            
        # Add the codeword
        codewords.append(words[best_c].tolist())
        
        # Update uncovered: set to False all words covered by best_c
        # cover_mask[best_c, :] is a boolean array of length total
        # We want to update uncovered where cover_mask[best_c, :] is True
        # uncovered[cover_mask[best_c, :]] = False
        uncovered[cover_mask[best_c, :]] = False
        
    return codewords
# EVOLVE-BLOCK-END
