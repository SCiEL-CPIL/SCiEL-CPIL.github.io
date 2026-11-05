import numpy as np
import itertools
import time

def construct(q, n, radius):
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s limit
    
    total = q ** n
    if total > 1000000:
        # Fallback for large spaces - use direct product construction
        return _direct_product(q, n, radius)
    
    # Precompute all neighbors for radius 1
    # For each word, its neighborhood includes itself and all words at distance 1
    neighbors = _precompute_neighbors(q, n, radius, total)
    
    # Greedy set cover
    uncovered = np.ones(total, dtype=bool)
    remaining = np.arange(total)
    selected = []
    
    while any(uncovered):
        if time.time() > DEADLINE:
            break
        # Find the word that covers the most uncovered positions
        best_idx = -1
        best_count = -1
        for idx in remaining:
            count = uncovered[neighbors[idx]].sum()
            if count > best_count:
                best_count = count
                best_idx = idx
            if time.time() > DEADLINE:
                break
        if best_idx < 0 or best_count <= 0:
            break
        selected.append(best_idx)
        uncovered[neighbors[best_idx]] = False
        remaining = remaining[remaining != best_idx]
    
    # Convert to list and apply local search
    selected_list = list(selected)
    if time.time() < DEADLINE - 5:
        selected_list = _local_search(q, n, radius, total, neighbors, selected_list, DEADLINE)
    
    # Convert indices to codewords
    result = []
    for idx in selected_list:
        digits = [0] * n
        temp = idx
        for k in range(n):
            digits[k] = temp % q
            temp //= q
        result.append(digits)
    
    return result

def _precompute_neighbors(q, n, radius, total):
    """Precompute neighbor lists for each word."""
    neighbors = []
    for idx in range(total):
        # Get digits of this word
        digits = [0] * n
        temp = idx
        for k in range(n):
            digits[k] = temp % q
            temp //= q
        
        # Collect neighbors within radius
        neigh = {idx}
        if radius >= 1:
            for pos in range(n):
                original = digits[pos]
                for new_val in range(q):
                    if new_val != original:
                        temp_digits = digits[:]
                        temp_digits[pos] = new_val
                        new_idx = 0
                        for d in temp_digits:
                            new_idx = new_idx * q + d
                        neigh.add(new_idx)
        neighbors.append(np.array(list(neigh), dtype=np.int64))
    return neighbors

def _local_search(q, n, radius, total, neighbors, selected_list, deadline):
    """Apply local search with swap and remove operations."""
    selected_set = set(selected_list)
    
    # Precompute coverage count for each word
    coverage = np.zeros(total, dtype=np.int32)
    for c in selected_list:
        coverage[neighbors[c]] += 1
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try removals
        i = 0
        while i < len(selected_list) and time.time() < deadline:
            c = selected_list[i]
            # Check if removing c still covers everything
            if np.all(coverage[neighbors[c]] >= 2):
                for j in neighbors[c]:
                    coverage[j] -= 1
                selected_list.pop(i)
                selected_set.remove(c)
                improved = True
            else:
                i += 1
        
        # Try swaps
        for i in range(len(selected_list)):
            if time.time() > deadline:
                break
            c = selected_list[i]
            # Try swapping c with some other word
            # Only consider words that are not already selected
            best_swap = -1
            best_redundancy = 0
            current_redundancy = (coverage > 1).sum()
            
            # Sample candidate swaps for efficiency
            candidates = np.random.choice(total, size=min(100, total), replace=False)
            for c_prime in candidates:
                if c_prime in selected_set:
                    continue
                # Simulate the swap
                # Coverage change: -1 for neighbors of c, +1 for neighbors of c_prime
                temp_cov = coverage.copy()
                temp_cov[neighbors[c]] -= 1
                temp_cov[neighbors[c_prime]] += 1
                
                # Check validity
                if np.all(temp_cov >= 1):
                    new_redundancy = (temp_cov > 1).sum()
                    if new_redundancy < current_redundancy - best_redundancy:
                        best_redundancy = current_redundancy - new_redundancy
                        best_swap = c_prime
                        best_temp_cov = temp_cov
            
            if best_swap >= 0:
                coverage = best_temp_cov
                selected_list[i] = best_swap
                selected_set.remove(c)
                selected_set.add(best_swap)
                improved = True
    
    # Final cleanup
    for i in range(len(selected_list)):
        c = selected_list[i]
        if np.all(coverage[neighbors[c]] >= 2):
            for j in neighbors[c]:
                coverage[j] -= 1
            selected_list.pop(i)
            selected_set.remove(c)
    
    return selected_list

def _direct_product(q, n, radius):
    """Fallback using direct product construction."""
    if n < 4:
        # Brute force for small n
        all_words = list(itertools.product(range(q), repeat=n))
        return all_words
    
    # Use tetracode-based construction for ternary case
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = []
        for tail in itertools.product(range(3), repeat=n - 4):
            for h in tetracode:
                result.append(h + list(tail))
        return result
    else:
        # General fallback: all words (valid but not optimal)
        return [list(w) for w in itertools.product(range(q), repeat=n)]
