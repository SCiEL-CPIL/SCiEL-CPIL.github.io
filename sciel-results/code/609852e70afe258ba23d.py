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
    """Precompute neighbor indices for all words. Returns list of lists."""
    total = q ** n
    neighbors = [[] for _ in range(total)]
    
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
        neighbors[w] = neigh
    return neighbors

def _greedy_init(q, n, radius, neighbors, total_words, deadline):
    """Fast greedy set cover initialization with sampling."""
    covered = np.zeros(total_words, dtype=bool)
    codewords = []
    
    # Sample size for greedy candidates
    sample_size = min(300, total_words)
    
    while not covered.all():
        if time.time() > deadline - 10.0:
            break
            
        best_word = -1
        best_gain = -1
        
        # Sample candidates
        candidates = random.sample(range(total_words), sample_size)
        
        for w in candidates:
            gain = 0
            for u in neighbors[w]:
                if not covered[u]:
                    gain += 1
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
        for u in neighbors[best_word]:
            covered[u] = True
            
    return codewords

def _local_search(cw_indices, neighbors, total_words, deadline):
    """Local search to minimize codeword count."""
    # Compute coverage counts
    count = np.zeros(total_words, dtype=np.int32)
    for ci in cw_indices:
        for u in neighbors[ci]:
            count[u] += 1
    
    best_size = len(cw_indices)
    best_cw = cw_indices[:]
    
    # Local search loop
    while time.time() < deadline - 2.0:
        improved = False
        
        # Phase 1: Remove redundant codewords
        indices_to_check = list(range(len(cw_indices)))
        random.shuffle(indices_to_check)
        
        for i in indices_to_check:
            if i >= len(cw_indices):
                continue
            ci = cw_indices[i]
            redundant = True
            for u in neighbors[ci]:
                if count[u] == 1:
                    redundant = False
                    break
            
            if redundant:
                for u in neighbors[ci]:
                    count[u] -= 1
                cw_indices.pop(i)
                improved = True
                if len(cw_indices) < best_size:
                    best_size = len(cw_indices)
                    best_cw = cw_indices[:]
        
        # Phase 2: Swap to improve
        if not improved and cw_indices:
            swaps_attempted = 0
            while swaps_attempted < 15 and time.time() < deadline - 2.0:
                swaps_attempted += 1
                i = random.randint(0, len(cw_indices) - 1)
                ci = cw_indices[i]
                
                # Find critical words (covered only by ci)
                critical_words = [u for u in neighbors[ci] if count[u] == 1]
                
                if not critical_words:
                    # ci is redundant
                    for u in neighbors[ci]:
                        count[u] -= 1
                    cw_indices.pop(i)
                    improved = True
                    if len(cw_indices) < best_size:
                        best_size = len(cw_indices)
                        best_cw = cw_indices[:]
                    break
                
                # Try to find a swap candidate
                u_crit = random.choice(critical_words)
                candidates_for_swap = neighbors[u_crit]
                random.shuffle(candidates_for_swap)
                
                for cj in candidates_for_swap:
                    if cj == ci:
                        continue
                    
                    # Check if swap is valid
                    valid = True
                    for u in critical_words:
                        if u not in neighbors[cj]:
                            valid = False
                            break
                    
                    if valid:
                        for u in neighbors[ci]:
                            count[u] -= 1
                        for u in neighbors[cj]:
                            count[u] += 1
                        cw_indices[i] = cj
                        improved = True
                        break
    # End while local search
    
    return best_cw

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin
    
    total_words = q ** n
    if total_words == 1:
        return [[0] * n]
        
    # Precompute neighbors
    neighbors = _precompute_neighbors(q, n, radius)
    
    # 1. Greedy Initialization
    cw_indices = _greedy_init(q, n, radius, neighbors, total_words, deadline)
    
    # 2. Local Search to Minimize
    best_cw = _local_search(cw_indices, neighbors, total_words, deadline)
    
    # Return best found
    result = []
    for ci in best_cw:
        result.append(_int_to_word(ci, n, q))
        
    return result

# EVOLVE-BLOCK-END
