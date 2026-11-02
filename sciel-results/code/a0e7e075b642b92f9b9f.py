"""Geometric Hyperplane Projection approach for K_3(7,1)."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools
from collections import defaultdict

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet of size q with covering radius r.
    Uses geometric hyperplane projection to guide greedy selection.
    """
    if n == 0:
        return [[]]
    if n == 1:
        return [[i] for i in range(q)]
    
    # Map words to integers 0..q^n-1
    def word_to_int(word):
        val = 0
        for v in word:
            val = val * q + v
        return val

    def int_to_word(val):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = val % q
            val //= q
        return w

    total_words = q ** n
    
    # Precompute neighbors as a numpy array
    # Radius 1 neighbors: self + 1-step changes
    # Number of neighbors: 1 + n * (q - 1)
    num_neighbors = 1 + n * (q - 1)
    neighbors_arr = np.zeros((total_words, num_neighbors), dtype=np.int32)
    
    for w_int in range(total_words):
        w = int_to_word(w_int)
        w_list = list(w)
        nb_idx = 0
        neighbors_arr[w_int, nb_idx] = w_int
        nb_idx += 1
        
        for pos in range(n):
            old_val = w_list[pos]
            for new_val in range(q):
                if new_val == old_val:
                    continue
                w_list[pos] = new_val
                nb_int = word_to_int(tuple(w_list))
                neighbors_arr[w_int, nb_idx] = nb_int
                nb_idx += 1
            w_list[pos] = old_val

    # Geometric Hyperplane Projection
    # Project to 2D tile space using two linear functionals
    weights1 = [1] * n
    weights2 = [(i + 1) % q if q > 1 else 0 for i in range(n)]
    
    tile_of_word = np.zeros(total_words, dtype=np.int32)
    for w_int in range(total_words):
        w = int_to_word(w_int)
        p1 = sum(w[i] * weights1[i] for i in range(n)) % q
        p2 = sum(w[i] * weights2[i] for i in range(n)) % q
        tile_of_word[w_int] = p1 * q + p2

    tile_size = q * q
    words_per_tile = np.zeros(tile_size, dtype=np.int32)
    for t in tile_of_word:
        words_per_tile[t] += 1

    # State
    covered = np.zeros(total_words, dtype=bool)
    covered_in_tile = np.zeros(tile_size, dtype=np.int32)
    codewords = []
    
    # Precompute inverse of words_per_tile for density calculation
    # Avoid division by zero
    inv_words_per_tile = np.zeros(tile_size, dtype=np.float64)
    for t in range(tile_size):
        if words_per_tile[t] > 0:
            inv_words_per_tile[t] = 1.0 / words_per_tile[t]
    
    for step in range(400):
        if np.sum(covered) == total_words:
            break
        
        # Calculate density and weights for tiles
        # density[t] = covered_in_tile[t] / words_per_tile[t]
        # weight[t] = 1 - density[t]
        # We want to maximize coverage of under-covered tiles.
        
        # Vectorized calculation of 'newly covered' for all candidates
        # ~covered[neighbors_arr] is a boolean array of shape (total_words, num_neighbors)
        newly_all = np.sum(~covered[neighbors_arr], axis=1)
        
        # Calculate tile score for all candidates
        # Weight for each word index j:
        # If covered[j], weight is 0.
        # Else, weight is (1 - density[tile_of_word[j]])
        
        # Compute density vector
        density = covered_in_tile * inv_words_per_tile
        tile_weights = 1.0 - density
        
        # Create word_weights vector
        # word_weights[j] = tile_weights[tile_of_word[j]] if not covered[j] else 0
        word_weights = np.where(covered, 0.0, tile_weights[tile_of_word])
        
        # Sum weights of neighbors for each candidate
        # tile_scores[i] = sum(word_weights[neighbors_arr[i]])
        tile_scores = np.sum(word_weights[neighbors_arr], axis=1)
        
        # Combined score
        # Primary: newly covered
        # Secondary: geometric tile score
        # Use a large multiplier for newly to ensure it dominates
        total_scores = newly_all * 10000.0 + tile_scores
        
        best_idx = np.argmax(total_scores)
        
        if newly_all[best_idx] == 0:
            break
            
        codewords.append(int_to_word(best_idx))
        
        # Update coverage
        nbs = neighbors_arr[best_idx]
        # Find which neighbors are newly covered
        new_mask = ~covered[nbs]
        new_nbs = nbs[new_mask]
        
        covered[nbs] = True
        if len(new_nbs) > 0:
            # Update covered_in_tile
            tiles_of_new = tile_of_word[new_nbs]
            # Use bincount to update counts
            increments = np.bincount(tiles_of_new, minlength=tile_size)
            covered_in_tile += increments

    return [list(w) for w in codewords]
# EVOLVE-BLOCK-END
