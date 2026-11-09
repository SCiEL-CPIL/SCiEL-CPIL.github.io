import time
import random
from math import gcd

def construct():
    """
    Finds the maximum number of pairwise disjoint infinite residue classes
    with distinct moduli in [2, 128].
    Two classes (r1, m1) and (r2, m2) are disjoint iff r1 != r2 (mod gcd(m1, m2)).
    """
    start_time = time.time()
    deadline = 80.0  # Safety margin
    
    max_modulus = 128
    moduli_list = list(range(2, max_modulus + 1))
    
    best_solution = []
    best_size = 0
    
    # Precompute gcds for speed? Not strictly necessary, but good.
    # Actually, gcd is very fast in Python 3.9+.
    
    def find_valid_residue(m, existing_classes):
        """
        Returns a valid residue r for modulus m given existing_classes, or None.
        """
        forbidden = [False] * m
        for r_i, m_i in existing_classes:
            g = gcd(m, m_i)
            # r must not be congruent to r_i mod g
            # r_i mod g is the target residue to avoid
            target = r_i % g
            # Mark all r in [0, m) such that r % g == target
            for r in range(target, m, g):
                forbidden[r] = True
        
        # Find an unmarked residue
        valid_residues = [r for r in range(m) if not forbidden[r]]
        if not valid_residues:
            return None
        # Pick randomly to allow different structures
        return random.choice(valid_residues)

    # Run multiple trials with different random orders
    trials = 0
    while time.time() < deadline:
        trials += 1
        # Shuffle moduli to explore different insertion orders
        shuffled_moduli = moduli_list.copy()
        random.shuffle(shuffled_moduli)
        
        current_classes = []
        
        for m in shuffled_moduli:
            r = find_valid_residue(m, current_classes)
            if r is not None:
                current_classes.append((r, m))
        
        # Check if this solution is better
        if len(current_classes) > best_size:
            best_size = len(current_classes)
            best_solution = current_classes.copy()
            
        # If we found a very large solution, we might stop early?
        # Theoretical max is likely < 20.
        # If we hit 15, it's very good.
        
        # Optional: Local Search / Repair
        # Try to swap a modulus in best_solution with one not in it?
        # Or re-assign residues?
        # Given the speed, simple greedy with many restarts is usually enough for N~10.
        
        # To improve diversity, we can bias the shuffle.
        # E.g., prioritize moduli with many factors?
        # Or just rely on random shuffle.
        
        # If time allows, do a local search on the best solution found so far
        if best_size > 0 and time.time() < deadline - 5.0:
             # Simple local search: try to replace a modulus in best_solution 
             # with one that is not in it, to see if we can maintain size or grow.
             # This is complex. Let's stick to restarts.
             pass

    # Convert to required format: list of [residue, modulus]
    result = [[r, m] for r, m in best_solution]
    
    # Ensure non-empty (task requires non-empty)
    if not result:
        # Fallback: a single class is always valid
        result = [[0, 2]]
        
    return result

# EVOLVE-BLOCK-END
