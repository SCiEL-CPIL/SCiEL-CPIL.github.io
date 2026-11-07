"""Simulated annealing on bitmask for the no-triple-pair-sums problem."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 128
DEADLINE = 85.0  # safety margin before 90s kill

def _has_bad_triple(S):
    """Return True if S contains three distinct members a,b,c with a+b,a+c,b+c all in S."""
    lst = sorted(S)
    L = len(lst)
    for i in range(L):
        for j in range(i + 1, L):
            for k in range(j + 1, L):
                a, b, c = lst[i], lst[j], lst[k]
                if (a + b) in S and (a + c) in S and (b + c) in S:
                    return True
    return False

def _validate(S):
    return not _has_bad_triple(S)

def _bitmask_from_set(S):
    """Convert a set of ints in 1..128 to a 128-bit integer bitmask."""
    mask = 0
    for x in S:
        mask |= (1 << (x - 1))
    return mask

def _set_from_bitmask(mask):
    """Convert a 128-bit integer bitmask to a set of ints in 1..128."""
    S = set()
    for i in range(N):
        if mask & (1 << i):
            S.add(i + 1)
    return S

def _count_violations(mask):
    """Count the number of forbidden triples (a,b,c) where a+b, a+c, b+c are all in the set."""
    # Precompute which triples are forbidden
    # For each triple (a,b,c) with a<b<c, check if a+b, a+c, b+c are all in mask
    # This is O(n^3) which is too slow, so we use a smarter approach
    # Actually, let's precompute all forbidden triples once
    pass

def construct():
    start = time.time()
    
    # Precompute all forbidden triples
    # A triple (a,b,c) with a<b<c is forbidden if a+b, a+c, b+c are all in the set
    # We need to count how many such triples have all three pair sums in the set
    # For efficiency, we'll precompute the list of all possible (a,b,c) triples
    # and for each, check if the pair sums are in the set
    
    # Actually, let's use a different approach:
    # For a given set S, a triple (a,b,c) is a violation if:
    # 1. a, b, c are all in S
    # 2. a+b, a+c, b+c are all in S
    # We can iterate over all triples (a,b,c) and check this condition
    
    # Precompute all triples
    triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                triples.append((a, b, c))
    
    # For each triple, store the pair sums
    triple_sums = [(a, b, c, a + b, a + c, b + c) for (a, b, c) in triples]
    
    def count_violations(mask):
        """Count violations for a given bitmask."""
        count = 0
        for (a, b, c, ab, ac, bc) in triple_sums:
            # Check if a, b, c are in the set
            if (mask >> (a - 1) & 1) and (mask >> (b - 1) & 1) and (mask >> (c - 1) & 1):
                # Check if pair sums are in the set
                if (mask >> (ab - 1) & 1) and (mask >> (ac - 1) & 1) and (mask >> (bc - 1) & 1):
                    count += 1
        return count
    
    # Initial set: upper half
    initial_set = set(range(N // 2 + 1, N + 1))
    current_mask = _bitmask_from_set(initial_set)
    current_violations = count_violations(current_mask)
    current_size = len(initial_set)
    
    best_mask = current_mask
    best_violations = current_violations
    best_size = current_size
    
    # Simulated annealing
    rng = random.Random(42)
    T = 10.0  # initial temperature
    cooling_rate = 0.9999
    
    while time.time() < DEADLINE:
        # Pick a random element to flip
        x = rng.randint(1, N)
        new_mask = current_mask ^ (1 << (x - 1))
        
        # Calculate new violations
        new_violations = count_violations(new_mask)
        new_size = len(_set_from_bitmask(new_mask))
        
        # Calculate energy change
        # We want to minimize violations and maximize size
        # Energy = violations - size (lower is better)
        old_energy = current_violations - current_size
        new_energy = new_violations - new_size
        delta_energy = new_energy - old_energy
        
        # Accept if better, or with probability exp(-delta/T)
        if delta_energy < 0 or rng.random() < np.exp(-delta_energy / T):
            current_mask = new_mask
            current_violations = new_violations
            current_size = new_size
            
            # Update best
            if current_violations == 0 and current_size > best_size:
                best_mask = current_mask
                best_violations = 0
                best_size = current_size
            elif current_violations < best_violations:
                best_mask = current_mask
                best_violations = current_violations
                best_size = current_size
        
        # Cool down
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
    
    # Return the best valid set
    if best_violations == 0:
        result = sorted(_set_from_bitmask(best_mask))
    else:
        # Fallback: return the initial set if no valid set was found
        result = sorted(initial_set)
    
    assert _validate(set(result)), "Final set is invalid!"
    return result

# EVOLVE-BLOCK-END
