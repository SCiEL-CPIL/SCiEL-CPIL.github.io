import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    # Moduli to consider: all even numbers from 2 to 128
    moduli = list(range(2, 129, 2))
    
    # Precompute gcds
    n = len(moduli)
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            gcds[i][j] = math.gcd(moduli[i], moduli[j])
            gcds[j][i] = gcds[i][j]
            
    # Backtracking search
    residues = [None] * n
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
            
        m = moduli[idx]
        # Try all possible residues
        # Heuristic: try residues that are likely to be valid
        # Just iterate 0 to m-1
        for r in range(m):
            valid = True
            for j in range(idx):
                if residues[j] is None:
                    continue
                r_j = residues[j]
                g = gcds[idx][j]
                if (r - r_j) % g == 0:
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
    else:
        # Fallback: return the seed solution if search fails
        return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
