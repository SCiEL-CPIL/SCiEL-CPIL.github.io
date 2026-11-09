"""Improved program: Simulated Annealing with Energy-based Collision Counting."""
import time
import random
import math

START_TIME = time.time()
DEADLINE = 55.0  # Safety margin before 60s limit

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def compute_sums_bitset(elements, max_sum):
    """
    Compute subset-sum bitsets indexed by cardinality.
    sums[k] is a Python int where bit s is set if sum s is achievable with exactly k elements.
    """
    n = len(elements)
    sums = [0] * (n + 1)
    sums[0] = 1
    
    for elem in elements:
        for k in range(n, 0, -1):
            sums[k] |= (sums[k-1] << elem)
            
    mask = (1 << (max_sum + 1)) - 1
    for k in range(n + 1):
        sums[k] &= mask
        
    return sums

def compute_energy(elements):
    """
    Compute the number of collisions between sums of different cardinalities.
    Energy = sum over i < j of popcount(sums[i] & sums[j]).
    """
    n = len(elements)
    if n == 0:
        return 0
    max_sum = sum(elements)
    sums = compute_sums_bitset(elements, max_sum)
    
    energy = 0
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            energy += (sums[i] & sums[j]).bit_count()
            
    return energy

def is_valid(elements):
    """Check if the set is valid (Energy == 0)."""
    if not elements:
        return True
    if len(set(elements)) != len(elements):
        return False
    if any(e < 1 or e > 256 for e in elements):
        return False
    return compute_energy(elements) == 0

def sa_search(target_size, time_budget):
    """
    Run Simulated Annealing to find a valid set of target_size.
    """
    # Initialize with a random set of target_size
    current_set = random.sample(range(1, 257), target_size)
    current_energy = compute_energy(current_set)
    
    best_set = list(current_set)
    best_energy = current_energy
    
    T = 10.0
    cooling = 0.999
    
    start = time.time()
    
    while time.time() - start < time_budget and not is_done():
        # Generate a move: Swap only to maintain size
        if len(current_set) > 0:
            idx = random.randint(0, len(current_set) - 1)
            removed = current_set[idx]
            # Find a replacement
            # Optimization: Pre-compute available? No, just filter.
            available = [x for x in range(1, 257) if x not in current_set]
            if available:
                added = random.choice(available)
                new_set = list(current_set)
                new_set[idx] = added
                
                new_energy = compute_energy(new_set)
                
                if new_energy < current_energy:
                    current_set = new_set
                    current_energy = new_energy
                    if current_energy == 0:
                        return list(current_set)
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_set = list(current_set)
                else:
                    delta = new_energy - current_energy
                    if delta == 0:
                        current_set = new_set
                        current_energy = new_energy
                    else:
                        p = math.exp(-delta / T) if T > 0 else 0
                        if random.random() < p:
                            current_set = new_set
                            current_energy = new_energy
                            
        T *= cooling
        
    if best_energy == 0:
        return best_set
    return None

def construct():
    random.seed(42)
    
    best_found = []
    
    # Try to find a set of increasing size
    # Start with 15, go up to 30
    for size in range(15, 31):
        if is_done():
            break
            
        remaining = DEADLINE - elapsed()
        if remaining < 2.0:
            break
            
        # Allocate time budget
        budget = min(5.0, remaining / 2.0)
        
        result = sa_search(size, budget)
        
        if result is not None:
            best_found = result
            continue
        else:
            # If we couldn't find a valid set of this size, 
            # it might be impossible or we just didn't find it.
            break
            
    if not best_found:
        # Fallback to a small known valid set
        best_found = [1, 2, 4, 8, 16, 32, 64, 128]
        
    # Final check
    if not is_valid(best_found):
        best_found = [1, 2, 4, 8, 16, 32, 64, 128]
        
    return [int(x) for x in best_found]
