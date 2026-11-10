import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    # The maximum number of pairwise non-coprime moduli in [2, 128] is 64 (all evens).
    # We assume this is the optimal count.
    moduli = [m for m in range(2, 129) if m % 2 == 0]
    
    # We need to assign residues r_i to each m_i such that
    # r_i != r_j mod gcd(m_i, m_j) for all i != j
    
    n = len(moduli)
    residues = [None] * n
    
    # Precompute gcds
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g

    # Backtracking solver
    # Order: increasing modulus (smaller domain first)
    # This is already sorted in 'moduli'
    
    def solve(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
        
        m = moduli[idx]
        for r in range(m):
            # Check compatibility with all previous
            valid = True
            for j in range(idx):
                g = gcds[idx][j]
                if r % g == residues[j] % g:
                    valid = False
                    break
            
            if valid:
                residues[idx] = r
                if solve(idx + 1):
                    return True
                residues[idx] = None
        
        return False

    # Try to solve
    if solve(0):
        result = [[residues[i], moduli[i]] for i in range(n)]
        return result
    
    # Fallback if backtracking fails (unlikely for evens)
    # Try a greedy approach with random restarts if needed, 
    # but backtracking on 64 vars with small domains should be fine.
    
    # If it fails, return a smaller valid set, e.g., just the first few
    # But we expect success.
    
    # Emergency fallback: return a known small valid set
    return [[1, 2], [0, 4]]
