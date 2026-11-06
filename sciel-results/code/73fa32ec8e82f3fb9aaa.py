"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Candidate moduli
    candidates = list(range(20, 201))
    
    def try_assign_with_limit(moduli_list, limit=50000):
        """
        Try to assign residues to moduli_list such that r_i != r_j mod gcd(m_i, m_j).
        Uses randomized backtracking with a step limit.
        Returns True if a valid assignment exists, False otherwise.
        """
        n = len(moduli_list)
        if n == 0:
            return True
        
        # Sort by modulus to fail fast on small domains
        indices = list(range(n))
        indices.sort(key=lambda i: moduli_list[i])
        sorted_moduli = [moduli_list[i] for i in indices]
        residues = [None] * n
        steps = 0
        
        def backtrack(i):
            nonlocal steps
            if steps > limit:
                return False
            if i == n:
                return True
            
            m = sorted_moduli[i]
            valid_residues = []
            for r in range(m):
                ok = True
                for j in range(i):
                    m_j = sorted_moduli[j]
                    r_j = residues[j]
                    g = gcd(m, m_j)
                    if (r - r_j) % g == 0:
                        ok = False
                        break
                if ok:
                    valid_residues.append(r)
            
            if not valid_residues:
                return False
            
            random.shuffle(valid_residues)
            
            for r in valid_residues:
                steps += 1
                residues[i] = r
                if backtrack(i + 1):
                    return True
                residues[i] = None
            
            return False
        
        # Try a few random restarts to escape local minima in search
        for _ in range(5):
            if backtrack(0):
                return True
        return False

    def find_residues_final(moduli_list):
        """
        Find a valid residue assignment for the final set.
        Uses more restarts and higher limit.
        """
        n = len(moduli_list)
        if n == 0:
            return []
        
        indices = list(range(n))
        indices.sort(key=lambda i: moduli_list[i])
        sorted_moduli = [moduli_list[i] for i in indices]
        residues = [None] * n
        
        def backtrack(i):
            if i == n:
                return True
            
            m = sorted_moduli[i]
            valid_residues = []
            for r in range(m):
                ok = True
                for j in range(i):
                    m_j = sorted_moduli[j]
                    r_j = residues[j]
                    g = gcd(m, m_j)
                    if (r - r_j) % g == 0:
                        ok = False
                        break
                if ok:
                    valid_residues.append(r)
            
            if not valid_residues:
                return False
            
            # Use random order for final solve as well
            random.shuffle(valid_residues)
            
            for r in valid_residues:
                residues[i] = r
                if backtrack(i + 1):
                    return True
                residues[i] = None
            
            return False
        
        # Try multiple restarts
        for _ in range(20):
            if backtrack(0):
                # Map back to original order
                orig_residues = [None] * n
                for k, idx in enumerate(indices):
                    orig_residues[idx] = residues[k]
                return orig_residues
        
        # Fallback: return zeros (should not happen if set was verified feasible)
        return [0] * n

    best_score = Fraction(0, 1)
    best_set = []
    
    # Strategy: Randomized Greedy Construction
    # We try multiple times with different random orders.
    # For each order, we greedily add moduli if the set remains feasible.
    
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        random.shuffle(candidates)
        current_set = []
        
        for m in candidates:
            if time.time() > deadline - 5:
                break
                
            # Check if adding m to current_set is feasible
            if try_assign_with_limit(current_set + [m]):
                current_set.append(m)
        
        # Calculate score for current_set
        if current_set:
            score = sum(Fraction(1, m) for m in current_set)
            if score > best_score:
                best_score = score
                best_set = list(current_set)
    
    # Final step: Reconstruct residues for the best_set
    if not best_set:
        # Fallback
        best_set = [20]
    
    residues = find_residues_final(best_set)
    best_classes = [[best_set[i], residues[i]] for i in range(len(best_set))]
    
    return best_classes
# EVOLVE-BLOCK-END
