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

def construct():
    """
    Maximize the number of pairwise disjoint infinite residue classes with distinct moduli in [2, 128].
    Uses a restricted search space (even moduli) with Randomized Local Search.
    """
    start_time = time.time()
    time_limit = 55.0  # Leave buffer for overhead
    
    # The set of even moduli is the largest pairwise non-coprime set in [2, 128].
    # Using these ensures gcd >= 2, making disjointness feasible.
    candidate_moduli = [m for m in range(2, 129) if m % 2 == 0]
    
    best_solution = []
    best_size = 0
    
    # We will run multiple iterations of Randomized Local Search (RLS)
    # to find a large independent set in the conflict graph of these moduli.
    
    for iteration in range(200):  # Limit iterations to ensure speed
        if time.time() - start_time > time_limit:
            break
            
        # Initialize current solution with a greedy pass
        current_solution = []
        available_moduli = list(candidate_moduli)
        random.shuffle(available_moduli)
        
        for m in available_moduli:
            # Try to find a valid residue for m
            # Shuffle residues to break symmetry
            residues = list(range(m))
            random.shuffle(residues)
            added = False
            for r in residues:
                if _is_valid_addition(r, m, current_solution):
                    current_solution.append((r, m))
                    added = True
                    break
            if not added:
                continue
        
        # Update best
        if len(current_solution) > best_size:
            best_size = len(current_solution)
            best_solution = list(current_solution)
            
        # Local Search Phase: Try to improve current_solution
        # We can try to swap or add/remove.
        # Simple kick: remove a random element, try to re-add or add another.
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            used_moduli = {m for r, m in current_solution}
            unused_moduli = [m for m in candidate_moduli if m not in used_moduli]
            
            if not unused_moduli:
                break
                
            # Try to add an unused modulus
            random.shuffle(unused_moduli)
            added_any = False
            for m in unused_moduli:
                residues = list(range(m))
                random.shuffle(residues)
                for r in residues:
                    if _is_valid_addition(r, m, current_solution):
                        current_solution.append((r, m))
                        added_any = True
                        improved = True
                        break
                if added_any:
                    break
            
            if improved:
                if len(current_solution) > best_size:
                    best_size = len(current_solution)
                    best_solution = list(current_solution)
                continue
            
            # If no addition possible, try a kick
            if len(current_solution) > 0:
                # Remove a random element
                idx = random.randint(0, len(current_solution) - 1)
                removed_pair = current_solution.pop(idx)
                # Try to add the removed one back with a different residue?
                # Or just try to add any unused one.
                # The loop above will try to add unused ones.
                # To ensure progress, we rely on the shuffle.
                # We mark improved to continue the loop, hoping the next add works.
                # But if we just removed and didn't add, size decreased.
                # We should only mark improved if we actually increased size or are in a "repair" phase.
                # For simplicity, let's just restart the greedy if we get stuck.
                pass # The outer for-loop will restart greedy.

    return [[r, m] for r, m in best_solution]

# EVOLVE-BLOCK-END
