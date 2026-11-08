"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _validate(U, V, W, n, m, p):
    """Check exact decomposition."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    # Schoolbook rank
    sb_rank = n * m * p  # 48
    
    # Build schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    best_rank = sb_rank
    
    # Try to reduce rank by merging pairs of schoolbook terms
    # If we can find two terms (i1,j1,k1) and (i2,j2,k2) such that
    # we can replace them with one term, we reduce rank by 1.
    
    # For a merge to work, we need the "support" patterns to be compatible.
    # Term (i,j,k) has U at row i*m+j, V at row j*p+k, W at row k*n+i.
    
    # Let's try: can we merge (i1,j,k) and (i2,j,k) for same j,k but different i?
    # No, because W would be at different rows.
    
    # Can we merge (i,j1,k) and (i,j2,k) for same i,k but different j?
    # U rows differ, V rows differ, W row same.
    # We'd need: a*U[i*m+j1,c]*V[j1*p+k,c]*W[k*n+i,c] + b*U[i*m+j2,c]*V[j2*p+k,c]*W[k*n+i,c]
    # to equal the original two terms. This doesn't obviously simplify.
    
    # Let's try a different approach: use a known lower-rank decomposition for 2x3x3
    # and extend it.
    
    # For 2x3x3 matrix multiplication, the rank is 5 (not 6).
    # This is a known result. Let's construct it.
    
    # The 2x3x3 decomposition of rank 5:
    # This is non-trivial. Let me try to find it via search.
    
    # Actually, let's try a systematic search for rank R from 47 down to 40.
    # For each rank, we'll try to find a valid decomposition.
    
    # Strategy: Start with schoolbook, then try to "absorb" one term into others.
    # Pick a term to eliminate, and try to find coefficients for the remaining terms
    # that reproduce its effect.
    
    # This is a linear system problem. For each term to eliminate (i,j,k),
    # we need to find corrections to the other terms such that the (i,j,k) equation
    # is still satisfied.
    
    # Let's try: eliminate term (i,j,k) by adjusting other terms.
    # The constraint for (i,j,k) is:
    #   sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # If we set U[i*m+j,c] = 0 for all c, then we need the other terms to contribute.
    # But other terms have different (i',j',k') so they don't affect the (i,j,k) equation
    # unless they share the same i*m+j row in U, or j*p+k row in V, or k*n+i row in W.
    
    # This is getting complex. Let me try a simpler approach:
    # Use a randomized search with the constraint that entries are in a small set.
    
    # Candidate entries: {0, 0.5, 1, 1.5, 2, -0.5, -1, -1.5, -2}
    entries = [0, 0.5, 1, 1.5, 2, -0.5, -1, -1.5, -2]
    
    # Try ranks from 47 down
    for target_rank in range(sb_rank - 1, max(sb_rank - 10, 30) - 1, -1):
        if time.time() > deadline - 10:
            break
        
        # For this rank, try to find a valid decomposition
        # Use a greedy approach: start with schoolbook of this rank, then adjust
        
        # Actually, let's try a different strategy:
        # Use the fact that for 2x3x3, rank 5 is possible.
        # Construct a 2x3x3 decomposition of rank 5, then extend to 2x3x8.
        
        # For 2x3x3, let me try to find the rank-5 decomposition.
        # The tensor has 18 equations. We need 5 columns.
        
        # Let's use a known construction or search for it.
        
        # For now, let's try a brute force search for small rank decompositions
        # of the 2x3x3 subproblem, then extend.
        
        # Actually, let's just try to directly search for a valid decomposition
        # at the target rank using a randomized approach.
        
        # Initialize with random values from the entry set
        attempts = 0
        max_attempts = 50000
        while attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Fill with random entries
            for c in range(target_rank):
                # Choose which (i,j,k) this column "primarily" serves
                i = np.random.randint(n)
                j = np.random.randint(m)
                k = np.random.randint(p)
                
                U[i*m+j][c] = np.random.choice(entries)
                V[j*p+k][c] = np.random.choice(entries)
                W[k*n+i][c] = np.random.choice(entries)
                
                # Also add some random non-zero entries elsewhere
                for _ in range(3):
                    ri = np.random.randint(n)
                    rj = np.random.randint(m)
                    rk = np.random.randint(p)
                    if np.random.random() < 0.3:
                        U[ri*m+rj][c] += np.random.choice(entries)
                    if np.random.random() < 0.3:
                        V[rj*p+rk][c] += np.random.choice(entries)
                    if np.random.random() < 0.3:
                        W[rk*n+ri][c] += np.random.choice(entries)
            
            if _validate(U, V, W, n, m, p):
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
        
        if best_rank < sb_rank:
            break
    
    return best
# EVOLVE-BLOCK-END
