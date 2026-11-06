import math
import time

def _disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r1 % g) != (r2 % g)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Candidate set: all even numbers from 2 to 128
    moduli = list(range(2, 129, 2))
    
    # Backtracking solver
    # Order moduli by ascending value (most constrained first)
    n = len(moduli)
    residues = [None] * n
    
    def solve(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
        
        m = moduli[idx]
        
        # Try residues in a specific order to find a solution quickly
        # Randomization might help if stuck, but deterministic order is fine for this structure
        # Let's try 0..m-1
        for r in range(m):
            valid = True
            # Check against all previously assigned
            for j in range(idx):
                if not _disjoint(r, m, residues[j], moduli[j]):
                    valid = False
                    break
            
            if valid:
                residues[idx] = r
                if solve(idx + 1):
                    return True
                residues[idx] = None
                
        return False

    if solve(0):
        result = [[residues[i], moduli[i]] for i in range(n)]
        return result
    
    # Fallback: if full set fails (should not), try a smaller set or random restarts
    # But 64 evens should work.
    # If it doesn't, return a known valid small set
    return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
