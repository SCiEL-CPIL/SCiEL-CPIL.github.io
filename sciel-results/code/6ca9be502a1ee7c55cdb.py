"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from math import gcd
from functools import reduce
import time

def _lcm(a, b):
    return a // gcd(a, b) * b

def _lcm_list(lst):
    return reduce(_lcm, lst)

def construct():
    """
    Return a nonempty subset of [1,22] such that ALL reciprocal subset sums,
    including the empty sum, are distinct. Maximize cardinality.
    
    Strategy: 
    1. Transform problem to finding a dissociated subset of integer weights w_i = LCM(1..22)/i.
    2. Use Greedy initialization with random orderings to find a large base set.
    3. Use Iterated Local Search (ILS) with random restarts and perturbations to escape local optima.
    4. The local search tries to add elements; if stuck, it swaps or removes elements to create space.
    """
    N = 22
    universe = list(range(1, N + 1))
    
    # Precompute LCM of 1..22
    L = _lcm_list(universe)
    
    # Integer weights: w_i = L // i
    weights = {i: L // i for i in universe}
    
    def is_dissociated(subset):
        """
        Check if the subset of integer weights has all distinct subset sums.
        Returns True if all 2^k sums are distinct.
        """
        k = len(subset)
        if k == 0:
            return True
        
        # We maintain a set of all sums generated so far
        sums = {0}
        for i in subset:
            w = weights[i]
            new_sums = set()
            for s in sums:
                ns = s + w
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
            
        return len(sums) == (1 << k)

    # 1. Greedy Initialization
    best_set = []
    best_size = 0
    
    # Try multiple random orderings to find a good starting point
    for trial in range(100):
        order = universe[:]
        random.shuffle(order)
        S = []
        for i in order:
            candidate = S + [i]
            if is_dissociated(candidate):
                S = candidate
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]

    # 2. Local Search / ILS
    start_time = time.time()
    timeout = 55.0  # Leave 5s buffer
    
    current_set = best_set[:]
    current_size = len(current_set)
    
    # We will try to improve the current set
    # Moves:
    # A. Try to add an element. If valid, accept.
    # B. If no additions possible, try swaps (remove one, add one).
    # C. If stuck, perturb (remove random elements) and restart local search phase.
    
    while time.time() - start_time < timeout:
        progress = False
        
        # Phase 1: Try to add elements
        # Shuffle universe to break symmetry
        candidates_to_add = [i for i in universe if i not in current_set]
        random.shuffle(candidates_to_add)
        
        for i in candidates_to_add:
            candidate = current_set + [i]
            if is_dissociated(candidate):
                current_set = candidate
                current_size += 1
                progress = True
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set[:]
        
        if progress:
            continue # Restart adding loop with new state
        
        # Phase 2: Stuck, try swaps
        # Try to swap an element in current_set with one not in current_set
        # We iterate through pairs to be systematic, or random to be fast?
        # Systematic is better for finding the *next* improvement.
        
        swaps_made = False
        elements_in = current_set[:]
        elements_out = [i for i in universe if i not in current_set]
        
        if elements_out:
            random.shuffle(elements_in)
            random.shuffle(elements_out)
            
            for i_remove in elements_in:
                for i_add in elements_out:
                    if i_remove == i_add:
                        continue
                    candidate = [x for x in current_set if x != i_remove] + [i_add]
                    if is_dissociated(candidate):
                        current_set = candidate
                        swaps_made = True
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = current_set[:]
                        break # Restart swap search with new set
                if swaps_made:
                    break
        
        if swaps_made:
            continue
        
        # Phase 3: Perturbation (Iterated Local Search step)
        # If we can't add or swap effectively, remove some elements to break the configuration
        # Remove 1 or 2 random elements
        if len(current_set) > 1:
            # Remove 1 element
            remove_count = 1 if random.random() < 0.7 else 2
            if remove_count > len(current_set):
                remove_count = len(current_set) - 1 # Keep at least 1
        
            to_remove = random.sample(current_set, remove_count)
            current_set = [x for x in current_set if x not in to_remove]
            current_size = len(current_set)
            
            # If we lost our best, restore it? No, ILS usually keeps a global best.
            # We continue searching from this perturbed state.
            # If we don't find anything better, we might loop, but the shuffle helps.
            
        else:
            # If set is very small, restart from a new greedy solution
            order = universe[:]
            random.shuffle(order)
            S = []
            for i in order:
                candidate = S + [i]
                if is_dissociated(candidate):
                    S = candidate
            current_set = S
            current_size = len(S)
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]

    if not best_set:
        best_set = [1]
        
    return best_set
# EVOLVE-BLOCK-END
