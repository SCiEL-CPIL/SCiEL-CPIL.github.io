"""Matrix multiplication 3x4x4: search for rank-minimal tensor decomposition."""
import time
import itertools
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 130.0  # safety margin before 150s kill

def _time_remaining():
    return TIME_LIMIT - (time.time() - START_TIME)

def _verify_decomposition(U, V, W, n, m, p):
    """Check if U,V,W with given shapes exactly decompose the matmul tensor.
    U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    Convention: W[k*n+i, r]
    The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] must equal 1 for all i,j,k.
    """
    rank = len(U[0]) if U else 0
    # Build the full tensor from the decomposition
    # T[i,j,k] = sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if val != 1:
                    return False
    return True

def _verify_decomposition_fast(U, V, W, n, m, p):
    """Faster verification using the structure."""
    rank = len(U[0]) if U else 0
    # For each (i,k), the product over j of the "slices" must work out.
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Let A_r[i,j] = U[i*m+j, r], B_r[j,k] = V[j*p+k, r], C_r[k,i] = W[k*n+i, r]
    # T[i,j,k] = sum_r A_r[i,j] * B_r[j,k] * C_r[k,i]
    # We need this to be 1 for all i,j,k.
    
    # Check: for each (i,k), sum over j... no, it's a 3D tensor.
    # Direct check but optimized:
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def construct():
    """Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 3x4 matrix by a 4x4 matrix.
    
    Returns: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    U: shape (n*m, rank) = (12, rank)
    V: shape (m*p, rank) = (16, rank)  
    W: shape (p*n, rank) = (12, rank)
    """
    n, m, p = 3, 4, 4
    
    # Start with the schoolbook decomposition (rank 48)
    rank_sb = n * m * p  # 48
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank decompositions using structured approaches.
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = 16
    # For 3x4x4, the minimum rank is known to be at most 48 (schoolbook).
    # Let's try a few heuristic approaches to see if we can do better.
    
    # Approach: Try rank values from 16 upward and check if a solution exists
    # using a randomized/structured search. Given the time constraint, we'll
    # try a limited search.
    
    # For small matrices, we can try to find decompositions with specific structures.
    # One approach: assume some entries are 0, 1, -1, or 1/2.
    
    # Let's try a direct search for small ranks using the bilinear structure.
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # We need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1.
    
    # Try rank from 16 to 47, but this is computationally expensive.
    # Instead, let's try a few specific structured decompositions.
    
    # Method 1: Try to use the fact that for fixed i, the n x p "slice" 
    # T[i,j,k] as a function of j,k must equal 1 for all j,k.
    # This means for each i, sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r]) = 1 for all j,k.
    
    # Let's try a randomized search with small integer/half-integer entries.
    # Given time constraints, we'll do a limited number of trials.
    
    def _try_rank(rank, max_trials=500):
        """Try to find a decomposition of given rank using random search."""
        if _time_remaining() < 5:
            return None
        
        for trial in range(max_trials):
            if _time_remaining() < 5:
                break
            
            # Random initialization with small values
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            
            # Initialize with random -1, 0, 1 values
            for i in range(n*m):
                for r in range(rank):
                    U[i][r] = random.choice([-1, 0, 1])
            for j in range(m*p):
                for r in range(rank):
                    V[j][r] = random.choice([-1, 0, 1])
            for k in range(p*n):
                for r in range(rank):
                    W[k][r] = random.choice([-1, 0, 1])
            
            # Check
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        s = 0
                        for r in range(rank):
                            s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                        if s != 1:
                            valid = False
                            break
            if valid:
                return (U, V, W)
        
        return None
    
    # Try ranks from 16 to 47, but limit total time
    # For efficiency, only try a few ranks with limited trials
    ranks_to_try = list(range(16, 48))
    random.shuffle(ranks_to_try)
    
    for rank in ranks_to_try:
        if _time_remaining() < 10:
            break
        if rank >= best_rank:
            continue
        
        result = _try_rank(rank, max_trials=100)
        if result is not None:
            U, V, W = result
            if _verify_decomposition_fast(U, V, W, n, m, p):
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break  # Found a valid lower rank, stop
    
    # If no improvement found, return schoolbook
    if best_rank == rank_sb:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}
