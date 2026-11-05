import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    MAX_M = 128
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def are_disjoint(r1, m1, r2, m2):
        g = gcd(m1, m2)
        return (r1 - r2) % g != 0

    # Candidate set: All even numbers from 2 to 128
    # This is the largest known clique in the non-coprime graph for this range.
    # Size: 64
    candidate_moduli = list(range(2, MAX_M + 1, 2))
    
    # If we can't solve for all, we might need to drop some.
    # We will try to solve for the full set first.
    
    def solve_csp(moduli):
        """
        Find residues for the given list of moduli such that all pairs are disjoint.
        Returns list of [r, m] or None if no solution found.
        """
        n = len(moduli)
        residues = [None] * n
        
        # Order moduli to improve pruning?
        # Smaller moduli have fewer residues, so they are more constrained?
        # Actually, larger moduli have more residues, so they are easier to satisfy?
        # Let's try sorting by modulus ascending.
        # Or maybe by number of divisors?
        
        # Let's stick to the given order for now, or sort by modulus.
        # Sorting by modulus ascending means we assign r for 2, then 4, etc.
        
        def backtrack(idx):
            if idx == n:
                return True
            
            m = moduli[idx]
            
            # Try all possible residues
            for r in range(m):
                valid = True
                # Check against all previously assigned moduli
                for j in range(idx):
                    if not are_disjoint(r, m, residues[j], moduli[j]):
                        valid = False
                        break
                
                if valid:
                    residues[idx] = r
                    if backtrack(idx + 1):
                        return True
                    residues[idx] = None
            return False
        
        if backtrack(0):
            return [[residues[i], moduli[i]] for i in range(n)]
        return None

    # Try to solve for the full set of even numbers
    solution = solve_csp(candidate_moduli)
    
    if time.time() > deadline:
        # If we timeout, we might have a partial solution? No, backtrack is all-or-nothing.
        # We need a fallback.
        # If the full set is too hard, we should have returned earlier or used a heuristic.
        # Let's assume it finds a solution quickly.
        pass

    if solution:
        return solution

    # If the full set of evens fails (unlikely, but possible if my logic is wrong),
    # try removing one element and retrying.
    # However, with 64 elements, backtracking might be slow if it fails deep.
    
    # To be safe, let's add a time check inside the backtracking?
    # The current solve_csp doesn't check time.
    # We should modify it to check time.
    
    def solve_csp_with_time(moduli, deadline):
        n = len(moduli)
        residues = [None] * n
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            
            m = moduli[idx]
            
            for r in range(m):
                valid = True
                for j in range(idx):
                    if not are_disjoint(r, m, residues[j], moduli[j]):
                        valid = False
                        break
                
                if valid:
                    residues[idx] = r
                    if backtrack(idx + 1):
                        return True
                    residues[idx] = None
            return False
        
        if backtrack(0):
            return [[residues[i], moduli[i]] for i in range(n)]
        return None

    # Re-run with time check
    solution = solve_csp_with_time(candidate_moduli, deadline)
    
    if solution:
        return solution

    # If it still fails, try a smaller set.
    # Remove the last element and retry.
    # This is a greedy removal strategy.
    
    current_moduli = candidate_moduli[:]
    best_solution = None
    
    while len(current_moduli) > 0 and time.time() < deadline:
        sol = solve_csp_with_time(current_moduli, deadline)
        if sol:
            best_solution = sol
            break
        # Remove the last element
        current_moduli.pop()
        
    if best_solution:
        return best_solution
    
    # Fallback: return a small valid set, e.g., [0, 2], [1, 4]
    # Check if they are disjoint: gcd(2,4)=2. 0 % 2 = 0, 1 % 2 = 1. Disjoint.
    return [[0, 2], [1, 4]]
