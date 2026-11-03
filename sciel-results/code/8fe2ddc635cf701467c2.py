"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 3, 8]
_TIME_LIMIT = 120  # seconds; evaluator kills at 150s, keep margin
_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _verify(U, V, W, n, m, p):
    """Check if U, V, W (lists of lists) form a valid decomposition."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    # Build the multiplication tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # T should equal 1 for all (i,j,k) in the multiplication tensor
    # Actually, the tensor for matrix mult is: T[i,j,k] = delta[i,j,k] where the 
    # multiplication (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is M[i,j,k] = 1 if we're computing (A*B)[i,k] from A[i,j] and B[j,k]
    # So M[i,j,k] = 1 for all valid i,j,k (it's the all-ones tensor in this context)
    # Wait, let me reconsider. The matrix multiplication tensor T has entries T[i,j,k]
    # such that (A*B)[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    # T[i,j,k] = 1 for all i,j,k (since (A*B)[i,k] = sum_j A[i,j]*B[j,k])
    
    # Compute: for each (i,k), sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should equal 1
    # And for (i,k) pairs, the cross terms should vanish.
    
    # More precisely: the decomposition is valid iff for all i,k:
    #   sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  (if we want the product)
    # But actually the tensor equation is:
    #   For all (i,j,k): the tensor entry should match.
    # The standard form: sum_c U[a,c]*V[b,c]*W[c,c'] ... no.
    
    # Let me use the standard tensor decomposition check:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And T[i,j,k] should be 1 for all i,j,k (the multiplication tensor is all-ones)
    
    # Actually no. The matrix multiplication tensor is:
    # M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because (A*B)[i,k] = sum_j A[i,j] * B[j,k], so the coefficient of A[i,j]*B[j,k] is 1.
    
    # So we need: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Efficient check using matrix operations:
    # For each c, form the outer product contribution.
    # Let's just do it directly for small dimensions.
    result = np.zeros((n, m, p))
    for c in range(rank):
        uc = U_arr[:, c]  # length n*m
        vc = V_arr[:, c]  # length m*p
        wc = W_arr[:, c]  # length p*n
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    result[i, j, k] += uc[i*m+j] * vc[j*p+k] * wc[k*n+i]
    
    return np.allclose(result, 1.0, atol=1e-9)

def _try_strassen_like(n, m, p, max_rank):
    """Try to find a Strassen-like decomposition by searching over sign patterns."""
    # For 2x2x2, Strassen uses 7 multiplications with +/-1 entries.
    # Generalize: try all combinations of +/-1 entries in U, V, W up to some rank.
    # This is exponential, so we use a heuristic approach.
    
    # Start with known patterns for small cases and extend.
    # For 3x3x8, we can try to find patterns based on 2x2x2 Strassen blocks.
    
    # Approach: use random search with +/-1 and 0 entries, verify.
    best = None
    for rank in range(1, min(max_rank + 1, 73)):
        if _time_left() < 5:
            break
        # Try a few random attempts per rank
        attempts = 50 if rank < 20 else 20
        for _ in range(attempts):
            if _time_left() < 2:
                break
            U = np.random.choice([0, 1, -1], size=(n*m, rank))
            V = np.random.choice([0, 1, -1], size=(m*p, rank))
            W = np.random.choice([0, 1, -1], size=(p*n, rank))
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def _structured_search(n, m, p, max_rank):
    """Search for structured decompositions using laser method ideas."""
    # The laser method suggests that for large matrices, rank can be reduced
    # by using tensor powers and subspaces. For 3x3x8, we try direct approaches.
    
    # Try to find a decomposition where U, V, W have sparse structure.
    # Use a greedy approach: build up the tensor entry by entry.
    
    # Alternative: use the fact that the tensor is all-ones and try to find
    # low-rank approximations that are exact.
    
    # Try rank from 1 upward with structured patterns
    for rank in range(1, min(max_rank + 1, 50)):
        if _time_left() < 5:
            break
        
        # Pattern 1: U and V have one-hot-like structure, W adjusts
        # Pattern 2: symmetric patterns
        # Pattern 3: use 2x2 Strassen blocks embedded
        
        if rank >= 7:
            # Try embedding Strassen 2x2x2 blocks
            # For 3x3x3, we can use 7 multiplications for the 2x2 part + 3 for the third row/col
            # This gives rank <= 7 + 3 = 10 for 3x3x3, then extend to 3x3x8
            pass
    
    return None

def _direct_greedy(n, m, p, max_rank):
    """Greedy search: add one tensor factor at a time to cover the all-ones tensor."""
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # We want sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k.
    
    # Greedy: at each step, find (u,v,w) that maximizes the number of uncovered entries.
    # But with +/-1 values, this is a covering problem.
    
    # For now, try a simple heuristic: use the schoolbook and try to merge terms.
    # Two schoolbook terms (i1,j1,k1) and (i2,j2,k2) can be merged if there's a way
    # to represent both with fewer factors.
    
    return None

# EVOLVE-BLOCK-START
def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 72
    
    # First, return schoolbook as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Try to find better decompositions
    # Strategy: search for rank from low to high
    
    # Try random search with small ranks first
    for target_rank in range(1, min(schoolbook_rank, 40)):
        if _time_left() < 10:
            break
        
        # Number of attempts based on remaining time
        if _time_left() > 60:
            attempts = 100
        elif _time_left() > 30:
            attempts = 50
        else:
            attempts = 10
        
        for _ in range(attempts):
            if _time_left() < 2:
                break
            
            # Random +/-1 and 0 entries
            U = np.random.choice([0, 1, -1], size=(n*m, target_rank))
            V = np.random.choice([0, 1, -1], size=(m*p, target_rank))
            W = np.random.choice([0, 1, -1], size=(p*n, target_rank))
            
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try structured patterns for specific ranks
    # Rank 7: Strassen for 2x2x2, need to extend to 3x3x8
    # This is complex; skip for now and return best found
    
    return best_result
# EVOLVE-BLOCK-END
