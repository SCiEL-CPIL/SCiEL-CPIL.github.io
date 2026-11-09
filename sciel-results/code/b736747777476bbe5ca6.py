"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _check_dissociated_fast(scaled_weights):
    """Check if all subset sums of scaled_weights are distinct using integer arithmetic.
    Returns True if dissociated (all subset sums distinct)."""
    n = len(scaled_weights)
    # Use a set to track all subset sums
    sums = {0}
    for w in scaled_weights:
        new_sums = set()
        for s in sums:
            ns = s + w
            if ns in sums or ns in new_sums:
                return False
            new_sums.add(ns)
        sums |= new_sums
    return True

def _check_dissociated_mod(scaled_weights, P):
    """Fast necessary check using modular arithmetic. Returns True if all subset sums distinct mod P."""
    n = len(scaled_weights)
    # Use bitset-like approach with a set of residues
    residues = {0}
    for w in scaled_weights:
        w_mod = w % P
        new_residues = set()
        for r in residues:
            nr = (r + w_mod) % P
            if nr in residues or nr in new_residues:
                return False
            new_residues.add(nr)
        residues |= new_residues
    return True

def _lcm_list(lst):
    """Compute LCM of a list of integers."""
    from math import gcd
    result = 1
    for x in lst:
        result = result * x // gcd(result, x)
    return result

def construct():
    """Return a nonempty subset of [1,22] such that ALL reciprocal subset sums are distinct.
    Maximize cardinality."""
    deadline = time.time() + 85  # Safety margin before 90s limit
    
    # The set {1,...,n} with weights 1/i is dissociated iff no two distinct subsets have the same sum of 1/i.
    # Equivalently, if we scale by L = lcm(1,...,n), then the integer weights w_i = L/i must form a dissociated set
    # (all subset sums distinct).
    
    # Known: {1,2,3,5,7,11,13,17,19} works (9 elements).
    # Let's try to find larger sets.
    
    # Strategy: Greedy construction with randomized restarts.
    # Start with a base set, try to add elements one by one.
    
    best_size = 9
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    # Precompute LCM for subsets of [1,22]
    # For a candidate set S, compute L = lcm(S), then weights w_i = L/i for i in S.
    # Check if these weights form a dissociated set.
    
    def is_dissociated(S):
        """Check if S (list of ints from 1-22) forms a dissociated reciprocal set."""
        if not S:
            return True
        L = _lcm_list(S)
        weights = [L // i for i in S]
        return _check_dissociated_fast(weights)
    
    # Try greedy: start with empty, add elements in various orders
    all_elements = list(range(1, 23))
    
    # First, try to extend the known 9-element set
    for _ in range(100):
        if time.time() > deadline:
            break
        
        # Try adding one element to the best set
        candidates = [e for e in all_elements if e not in best_set]
        random.shuffle(candidates)
        
        for c in candidates:
            if time.time() > deadline:
                break
            new_set = best_set + [c]
            if is_dissociated(new_set):
                best_set = new_set
                best_size += 1
                break
    
    # Try randomized greedy from scratch
    for restart in range(500):
        if time.time() > deadline:
            break
        
        # Random ordering
        order = all_elements[:]
        random.shuffle(order)
        
        current_set = []
        for e in order:
            if time.time() > deadline:
                break
            candidate = current_set + [e]
            if is_dissociated(candidate):
                current_set = candidate
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
    
    # Try to extend best_set further
    for _ in range(200):
        if time.time() > deadline:
            break
        candidates = [e for e in all_elements if e not in best_set]
        random.shuffle(candidates)
        for c in candidates:
            if time.time() > deadline:
                break
            new_set = best_set + [c]
            if is_dissociated(new_set):
                best_set = new_set
                best_size += 1
                break
    
    return best_set
# EVOLVE-BLOCK-END
