import time
from math import gcd
import random

def _score(classes):
    return sum(1.0 / m for m, r in classes)

def _greedy_fill(initial_classes, deadline):
    """
    Fills the set starting from initial_classes by adding more moduli
    in ascending order of m (highest 1/m first).
    """
    # Convert to dict for O(1) lookup of existing moduli
    selected = {m: r for m, r in initial_classes}
    
    # List of all possible moduli
    all_moduli = list(range(20, 201))
    
    # We only consider moduli not already in selected
    candidates = [m for m in all_moduli if m not in selected]
    
    # Sort candidates by 1/m descending (i.e., m ascending)
    candidates.sort(key=lambda m: m)
    
    for m in candidates:
        if time.time() > deadline:
            break
            
        # Find a valid residue for m
        valid_r = None
        # We can try residues in a specific order. 
        # Usually, trying all and picking the first valid one is fine.
        # To add diversity, we could shuffle, but greedy fill is deterministic given the set.
        # Let's try to find the residue that minimizes conflicts with *existing* ones?
        # No, if it's valid, it has 0 conflicts with existing ones.
        # So any valid residue works.
        
        # Optimization: Precompute gcds? 
        # For each existing (m2, r2), we need r != r2 mod gcd(m, m2).
        
        found = False
        # Try residues 0 to m-1
        for r in range(m):
            ok = True
            for m2, r2 in selected.items():
                g = gcd(m, m2)
                if (r - r2) % g == 0:
                    ok = False
                    break
            if ok:
                found = True
                valid_r = r
                break
        
        if found:
            selected[m] = valid_r
            
    # Convert back to list
    return [[m, r] for m, r in selected.items()]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Initial greedy solution
    best_classes = _greedy_fill([], deadline)
    best_score = _score(best_classes)
    
    # Iterated Local Search
    # We will try to improve by removing some elements and re-filling
    # The number of iterations is limited by time.
    
    while time.time() < deadline - 5: # Keep 5s for final check/cleanup
        
        # Perturbation: Randomly remove a fraction of the current best solution
        # Let's remove between 10% and 30% of the elements
        n = len(best_classes)
        if n == 0:
            break
            
        removal_fraction = random.uniform(0.1, 0.3)
        n_remove = max(1, int(n * removal_fraction))
        
        # Randomly select elements to remove
        to_remove = random.sample(best_classes, n_remove)
        remove_set = {m for m, r in to_remove}
        
        # Create initial set for the next greedy fill
        remaining = [[m, r] for m, r in best_classes if m not in remove_set]
        
        # Perform greedy fill
        new_classes = _greedy_fill(remaining, deadline)
        new_score = _score(new_classes)
        
        if new_score > best_score:
            best_score = new_score
            best_classes = new_classes
            
    return best_classes
