"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time

def _is_disjoint(r1, m1, r2, m2):
    """Check if residue class r1 mod m1 is disjoint from r2 mod m2."""
    g = math.gcd(m1, m2)
    return (r1 - r2) % g != 0

def _is_valid_addition(r, m, current_set):
    """Check if (r, m) is disjoint from all pairs in current_set."""
    for r2, m2 in current_set:
        if not _is_disjoint(r, m, r2, m2):
            return False
    return True

def _greedy_init():
    """Generate an initial solution using a greedy algorithm with random ordering."""
    moduli = list(range(2, 129))
    random.shuffle(moduli)
    chosen = []
    for m in moduli:
        # Shuffle residues to break ties randomly
        residues = list(range(m))
        random.shuffle(residues)
        for r in residues:
            if _is_valid_addition(r, m, chosen):
                chosen.append((r, m))
                break
    return chosen

def construct():
    """
    Maximize the number of pairwise disjoint infinite residue classes with distinct moduli in [2, 128].
    Uses Stochastic Local Search (Randomized Local Search with Kicks) to escape local optima.
    """
    start_time = time.time()
    time_limit = 55.0  # Leave buffer for overhead
    
    # Initialize with a good greedy solution
    best_solution = _greedy_init()
    best_size = len(best_solution)
    
    # Keep a record of the best solution found
    current_solution = list(best_solution)
    
    # Precompute gcds for speed? Maybe not necessary, but let's check.
    # The main bottleneck is checking disjointness.
    
    # Stochastic Local Search Loop
    while time.time() - start_time < time_limit:
        # Phase 1: Try to add any valid pair
        added = False
        # Try adding from unused moduli
        used_moduli = {m for r, m in current_solution}
        unused_moduli = [m for m in range(2, 129) if m not in used_moduli]
        
        # Shuffle unused moduli to try them in random order
        random.shuffle(unused_moduli)
        
        for m in unused_moduli:
            # Try a few random residues for this modulus
            # Instead of all, try random ones first, then all if needed?
            # For efficiency, just try all residues for the first few moduli?
            # Let's try all residues for each unused modulus until one works.
            # This is O(N * M * M) in worst case, but N is small (max 127).
            # Actually, if we just iterate through all unused moduli and all residues,
            # we might find a valid addition.
            found = False
            for r in range(m):
                if _is_valid_addition(r, m, current_solution):
                    current_solution.append((r, m))
                    added = True
                    found = True
                    break
            if found:
                break
        
        if added:
            # If we added something, update best if better
            if len(current_solution) > best_size:
                best_size = len(current_solution)
                best_solution = list(current_solution)
            # Continue trying to add more (restart the while loop logic effectively)
            # But we just added one, so we can immediately try to add another.
            # The while loop will restart the check.
            continue
        
        # Phase 2: No single addition works. Perform a "kick" to escape local optimum.
        # Remove 1 or 2 random elements to free up space/constraints.
        if len(current_solution) > 0:
            # Decide how many to remove (kick strength)
            # Usually 1 is enough, sometimes 2 helps.
            kick_size = 1 if random.random() < 0.7 else 2
            
            if len(current_solution) >= kick_size:
                # Pick random indices to remove
                indices_to_remove = random.sample(range(len(current_solution)), kick_size)
                # Remove them
                for idx in sorted(indices_to_remove, reverse=True):
                    current_solution.pop(idx)
                
                # After kicking, try to add again.
                # We can loop back to Phase 1 immediately.
                # To avoid infinite loop of kick->add->kick->add with same size,
                # we rely on the randomness.
                continue
        
        # If we can't even kick (empty or too small), break?
        # If empty, we can just add.
        if len(current_solution) == 0:
             # Add a random valid pair
             m = random.choice(range(2, 129))
             r = random.randint(0, m-1)
             if _is_valid_addition(r, m, current_solution):
                 current_solution.append((r, m))
             else:
                 # Try to find any valid pair
                 found = False
                 for m_try in range(2, 129):
                     for r_try in range(m_try):
                         if _is_valid_addition(r_try, m_try, current_solution):
                             current_solution.append((r_try, m_try))
                             found = True
                             break
                     if found: break
                 if not found:
                     break # Should not happen for empty set

    # Return the best solution found
    return [[r, m] for r, m in best_solution]

# EVOLVE-BLOCK-END
