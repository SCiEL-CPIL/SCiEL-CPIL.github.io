"""
Football pool problem: ternary covering code K_3(8,1)
Minimize number of codewords so every word is within Hamming distance 1.
"""
import numpy as np
import time
import random

def _int_to_word(val, n, q):
    """Convert integer to list of digits in base q."""
    word = []
    for _ in range(n):
        word.append(val % q)
        val //= q
    return word

def _precompute_neighbors(q, n, radius=1):
    """Precompute neighbor indices for all words. Returns list of numpy arrays."""
    total = q ** n
    neighbors = []
    
    for w in range(total):
        # Convert to word
        word = []
        tmp = w
        for _ in range(n):
            word.append(tmp % q)
            tmp //= q
        
        neigh = [w]
        if radius >= 1:
            for pos in range(n):
                orig = word[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    word[pos] = val
                    # Convert back
                    nv = 0
                    for i in range(n - 1, -1, -1):
                        nv = nv * q + word[i]
                    neigh.append(nv)
                word[pos] = orig
        neighbors.append(np.array(neigh, dtype=np.int32))
    return neighbors

def _greedy_init(q, n, radius, neighbors, total_words, deadline):
    """Fast greedy set cover initialization with vectorized gain computation."""
    covered = np.zeros(total_words, dtype=bool)
    codewords = []
    
    # Precompute neighbor arrays as a 2D array for vectorized operations
    # Each row is the neighbors of one word
    neighbor_arrays = np.array(neighbors, dtype=np.int32)
    
    # Sample size for greedy candidates
    sample_size = min(400, total_words)
    
    while not covered.all():
        if time.time() > deadline - 15.0:
            break
            
        best_word = -1
        best_gain = -1
        
        # Sample candidates
        candidates = random.sample(range(total_words), sample_size)
        
        for w in candidates:
            gain = np.sum(~covered[neighbor_arrays[w]])
            if gain > best_gain:
                best_gain = gain
                best_word = w
        
        if best_word == -1:
            # Fallback: pick any uncovered word
            uncovered = np.where(~covered)[0]
            if len(uncovered) == 0:
                break
            best_word = int(uncovered[0])
        
        codewords.append(best_word)
        covered[neighbor_arrays[best_word]] = True
            
    return codewords

def _local_search(cw_indices, neighbors, total_words, deadline):
    """Local search to minimize codeword count with vectorized operations."""
    # Precompute neighbor arrays as a 2D array
    neighbor_arrays = np.array(neighbors, dtype=np.int32)
    
    # Compute coverage counts
    count = np.zeros(total_words, dtype=np.int32)
    for ci in cw_indices:
        count[neighbor_arrays[ci]] += 1
    
    best_size = len(cw_indices)
    best_cw = cw_indices[:]
    
    # Local search loop
    while time.time() < deadline - 5.0:
        improved = False
        
        # Phase 1: Remove redundant codewords (vectorized check)
        indices_to_check = list(range(len(cw_indices)))
        random.shuffle(indices_to_check)
        
        for i in indices_to_check:
            if i >= len(cw_indices):
                continue
            ci = cw_indices[i]
            # A codeword is redundant if no word in its neighborhood has count == 1
            redundant = np.all(count[neighbor_arrays[ci]] > 1)
            
            if redundant:
                count[neighbor_arrays[ci]] -= 1
                cw_indices.pop(i)
                improved = True
                if len(cw_indices) < best_size:
                    best_size = len(cw_indices)
                    best_cw = cw_indices[:]
        
        # Phase 2: Swap to improve
        if not improved and cw_indices:
            swaps_attempted = 0
            while swaps_attempted < 30 and time.time() < deadline - 5.0:
                swaps_attempted += 1
                i = random.randint(0, len(cw_indices) - 1)
                ci = cw_indices[i]
                
                # Find critical words (covered only by ci)
                ci_neighbors = neighbor_arrays[ci]
                critical_mask = count[ci_neighbors] == 1
                critical_words = ci_neighbors[critical_mask]
                
                if len(critical_words) == 0:
                    # ci is redundant
                    count[ci_neighbors] -= 1
                    cw_indices.pop(i)
                    improved = True
                    if len(cw_indices) < best_size:
                        best_size = len(cw_indices)
                        best_cw = cw_indices[:]
                    break
                
                # Try to find a swap candidate that covers ALL critical words
                # Use the first critical word to generate candidates, then verify all
                u_crit = int(random.choice(critical_words))
                candidates_for_swap = neighbor_arrays[u_crit].copy()
                np.random.shuffle(candidates_for_swap)
                
                for cj in candidates_for_swap:
                    if cj == ci:
                        continue
                    
                    # Check if swap is valid: all critical words must be in cj's neighborhood
                    cj_neighbors = neighbor_arrays[cj]
                    # Create a set for fast lookup
                    cj_set = set(cj_neighbors.tolist())
                    valid = all(u in cj_set for u in critical_words)
                    
                    if valid:
                        count[ci_neighbors] -= 1
                        count[cj_neighbors] += 1
                        cw_indices[i] = cj
                        improved = True
                        break
    # End while local search
    
    return best_cw

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # Leave 2s margin
    
    total_words = q ** n
    if total_words == 1:
        return [[0] * n]
        
    # Precompute neighbors
    neighbors = _precompute_neighbors(q, n, radius)
    
    # 1. Greedy Initialization
    cw_indices = _greedy_init(q, n, radius, neighbors, total_words, deadline)
    
    # 2. Local Search to Minimize
    best_cw = _local_search(cw_indices, neighbors, total_words, deadline)
    
    # 3. If time permits, do a second restart with different random seed
    if time.time() < deadline - 40.0:
        random.seed(random.randint(0, 1000000))
        cw_indices2 = _greedy_init(q, n, radius, neighbors, total_words, deadline)
        best_cw2 = _local_search(cw_indices2, neighbors, total_words, deadline)
        if len(best_cw2) < len(best_cw):
            best_cw = best_cw2
    
    # Return best found
    result = []
    for ci in best_cw:
        result.append(_int_to_word(ci, n, q))
        
    return result

# EVOLVE-BLOCK-END
