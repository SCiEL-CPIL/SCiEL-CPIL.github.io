"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import numpy as np
import time
import math

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius r.
    Uses exact backtracking search with bitsets for small instances,
    falling back to greedy heuristics for larger ones.
    """
    if radius < 0:
        radius = 0
    if radius >= n:
        return [[0] * n]
    
    total_words = q ** n
    
    if total_words <= 1000:
        return _exact_search(q, n, radius)
    else:
        return _greedy_cover(q, n, radius)

def _exact_search(q, n, radius):
    """
    Find a minimal covering code using backtracking with bitset optimization.
    """
    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    
    # Precompute coverage masks
    coverage = [0] * total_words
    for i, w in enumerate(all_words):
        mask = 0
        for j, v in enumerate(all_words):
            dist = sum(1 for a, b in zip(w, v) if a != b)
            if dist <= radius:
                mask |= (1 << j)
        coverage[i] = mask
        
    full_mask = (1 << total_words) - 1
    
    # Volume of ball for lower bound
    def calc_volume(n, q, r):
        vol = 1
        for i in range(1, r + 1):
            vol += (math.factorial(n) // (math.factorial(i) * math.factorial(n - i))) * ((q - 1) ** i)
        return vol
    
    vol = calc_volume(n, q, radius)
    lower_bound = int(np.ceil(total_words / vol))
    
    # Get upper bound from greedy
    greedy_sol = _greedy_cover(q, n, radius)
    upper_bound = len(greedy_sol)
    
    start_time = time.time()
    time_limit = 100
    node_limit = 2000000
    
    nodes_explored = 0
    best_code = None
    best_size = float('inf')
    
    def search(k, current_code, uncovered_mask, used_indices):
        nonlocal best_code, best_size, nodes_explored, start_time, time_limit, node_limit
        
        nodes_explored += 1
        if time.time() - start_time > time_limit or nodes_explored > node_limit:
            return False
            
        if len(current_code) == k:
            if uncovered_mask == 0:
                best_code = current_code[:]
                best_size = k
                return True
            return False
            
        if not uncovered_mask:
            best_code = current_code[:]
            best_size = len(current_code)
            return True
            
        remaining_slots = k - len(current_code)
        uncovered_count = bin(uncovered_mask).count('1')
        if remaining_slots * vol < uncovered_count:
            return False
            
        best_gain = -1
        candidates = []
        
        for i in range(total_words):
            if i in used_indices:
                continue
            gain = bin(coverage[i] & uncovered_mask).count('1')
            if gain > best_gain:
                best_gain = gain
                candidates = [i]
            elif gain == best_gain:
                candidates.append(i)
                
        if best_gain <= 0:
            return False
            
        random.shuffle(candidates)
        
        for idx in candidates:
            new_uncovered = uncovered_mask & ~coverage[idx]
            used_indices.add(idx)
            current_code.append(all_words[idx])
            
            if search(k, current_code, new_uncovered, used_indices):
                return True
                
            current_code.pop()
            used_indices.remove(idx)
            
        return False

    for k in range(lower_bound, upper_bound + 1):
        nodes_explored = 0
        if search(k, [], full_mask, set()):
            return best_code
            
    return greedy_sol

def _greedy_cover(q, n, radius, max_iterations=1000):
    """Simple greedy cover for larger spaces."""
    codewords = []
    covered = set()
    
    all_words = list(itertools.product(range(q), repeat=n))
    
    for _ in range(max_iterations):
        best_word = None
        best_new = 0
        for w in all_words:
            if w in covered:
                continue
            new_count = 0
            for v in all_words:
                if v in covered:
                    continue
                dist = sum(1 for a, b in zip(w, v) if a != b)
                if dist <= radius:
                    new_count += 1
            if new_count > best_new:
                best_new = new_count
                best_word = w
        
        if best_word is None or best_new == 0:
            break
        
        codewords.append(list(best_word))
        for v in all_words:
            dist = sum(1 for a, b in zip(best_word, v) if a != b)
            if dist <= radius:
                covered.add(v)
        
        if len(covered) == len(all_words):
            break
    
    return codewords
# EVOLVE-BLOCK-END
