"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

N = 512

def _has_4ap_bitset(bits, x):
    """Check if adding x (1-indexed) to the set represented by bits creates a 4-term AP.
    bits is an integer where bit i is set if element (i+1) is in the set."""
    x_idx = x - 1  # 0-indexed
    
    # Case 1: x is the 4th term: x-3d, x-2d, x-d in selected
    for d in range(1, (x_idx)//3 + 1):
        if ((bits >> (x_idx - 3*d)) & 1) and ((bits >> (x_idx - 2*d)) & 1) and ((bits >> (x_idx - d)) & 1):
            return True
    
    # Case 2: x is the 3rd term: x-2d, x-d, x+d in selected
    for d in range(1, (x_idx)//2 + 1):
        if ((bits >> (x_idx - 2*d)) & 1) and ((bits >> (x_idx - d)) & 1) and ((bits >> (x_idx + d)) & 1):
            return True
    
    # Case 3: x is the 2nd term: x-d, x+d, x+2d in selected
    for d in range(1, x_idx + 1):
        if x_idx + 2*d >= N:
            break
        if ((bits >> (x_idx - d)) & 1) and ((bits >> (x_idx + d)) & 1) and ((bits >> (x_idx + 2*d)) & 1):
            return True
    
    # Case 4: x is the 1st term: x+d, x+2d, x+3d in selected
    for d in range(1, (N - 1 - x_idx)//3 + 1):
        if ((bits >> (x_idx + d)) & 1) and ((bits >> (x_idx + 2*d)) & 1) and ((bits >> (x_idx + 3*d)) & 1):
            return True
    
    return False

def _greedy_construct_bitset(order):
    """Greedy construction using bitset, given an ordering of elements."""
    bits = 0
    count = 0
    for x in order:
        x_idx = x - 1
        if not _has_4ap_bitset(bits, x):
            bits |= (1 << x_idx)
            count += 1
    return bits, count

def _local_search_bitset(bits, max_iter=2000, deadline=None):
    """Improve the set using swap moves: remove one element, add two others (net +1)."""
    current_size = bin(bits).count('1')
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        # Try removing each element and adding two others
        for rem in range(N):
            if not ((bits >> rem) & 1):
                continue
            trial = bits & ~(1 << rem)
            trial_size = current_size - 1
            
            # Try adding pairs
            for add1 in range(N):
                if add1 == rem or ((bits >> add1) & 1):
                    continue
                if _has_4ap_bitset(trial, add1 + 1):
                    continue
                trial_with_1 = trial | (1 << add1)
                
                for add2 in range(add1 + 1, N):
                    if add2 == rem or ((bits >> add2) & 1):
                        continue
                    if _has_4ap_bitset(trial_with_1, add2 + 1):
                        continue
                    new_bits = trial_with_1 | (1 << add2)
                    new_size = trial_size + 2
                    if new_size > current_size:
                        bits = new_bits
                        current_size = new_size
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return bits, current_size

def _try_add_single(bits, excluded):
    """Try to add a single element to bits without creating a 4-AP. Returns new bits or None."""
    for x in range(1, N + 1):
        if x in excluded:
            continue
        if not _has_4ap_bitset(bits, x):
            return bits | (1 << (x - 1))
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Try multiple randomized greedy constructions
    best_bits = 0
    best_size = 0
    
    # First, try natural order
    bits, size = _greedy_construct_bitset(range(1, N + 1))
    best_bits = bits
    best_size = size
    
    # Try several randomized orderings
    num_random_tries = 500
    for attempt in range(num_random_tries):
        if time.time() > deadline - 10.0:
            break
        
        # Random shuffle
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        bits, size = _greedy_construct_bitset(order)
        if size > best_size:
            best_bits = bits
            best_size = size
    
    # Local search on the best set
    if time.time() < deadline - 5.0:
        bits, size = _local_search_bitset(best_bits, max_iter=5000, deadline=deadline - 3.0)
        if size > best_size:
            best_bits = bits
            best_size = size
    
    # Try a few more randomized starts with local search
    for attempt in range(50):
        if time.time() > deadline - 5.0:
            break
        
        order = list(range(1, N + 1))
        random.shuffle(order)
        bits, size = _greedy_construct_bitset(order)
        
        # Quick local search on this candidate
        if time.time() < deadline - 3.0:
            bits, size = _local_search_bitset(bits, max_iter=500, deadline=deadline - 3.0)
        
        if size > best_size:
            best_bits = bits
            best_size = size
    
    # Convert bitset to list
    result = []
    for i in range(N):
        if (best_bits >> i) & 1:
            result.append(i + 1)
    
    return sorted(result)
# EVOLVE-BLOCK-END
