"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank as fallback
    rank_schoolbook = n * m * p
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, rank):
        """Verify that the decomposition exactly reproduces matrix multiplication."""
        # Check: for all i,k: sum_r W[k*n+i][r] * (sum over c where U[i*m+j][c] and V[j*p+k][c] both nonzero)
        # More precisely: The tensor T[i,j,k] should equal sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r]
        # For matrix mult: T[i,j,k] = 1 if this contributes to (A*B)[i,k], i.e., always 1 for valid (i,j,k)
        # Actually the MM tensor has T[i,j,k] = 1 for all i,j,k (each term A[i,j]*B[j,k] contributes)
        # So we need: for all i,j,k: sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] == 1
        # And for the "wrong" combinations, it should be 0, but in MM tensor all (i,j,k) are 1.
        # Wait, the MM tensor is: T_{i,j,k} = 1 if the (i,j) entry of A and (j,k) entry of B multiply to give (i,k) of AB.
        # In standard MM, every triple (i,j,k) with i in [0,n), j in [0,m), k in [0,p) contributes exactly once.
        # So T[i,j,k] = 1 for all valid i,j,k.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def verify_full(U, V, W, rank):
        """Full verification: reconstruct AB from A,B and compare."""
        # The decomposition means: (A*B)[i,k] = sum_r W[k*n+i][r] * (sum_j U[i*m+j][r]*A[i,j]) * (sum_j V[j*p+k][r]*B[j,k])
        # Wait, that's not quite right. Let me think again.
        # Standard: (A*B)[i,k] = sum_j A[i,j] * B[j,k]
        # Tensor decomposition: (A*B)[i,k] = sum_r W[k*n+i][r] * C_r[i] * D_r[k]
        # where C_r[i] = sum_j U[i*m+j][r] * A[i,j]  -- no, that's not standard either.
        # 
        # Actually the standard tensor decomposition of MM is:
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # where T[i,j,k] = 1 for the MM tensor (each (i,j,k) triple contributes A[i,j]*B[j,k])
        # So the condition is: for all i,j,k: sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] = 1
        # And this is necessary and sufficient.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Try known constructions
    # For 3x3x3, rank 21 is achievable. Let me try to find 3x3x5 decompositions.
    # Lower bound: max(n*m, m*p, p*n) = max(9, 15, 15) = 15, but tensor rank is higher.
    # For 3x3x5, let's try ranks from 30 upward (since 3x3x3 needs 21, 3x3x5 should need more).
    
    # Let me try a structured approach: use the 3x3x3 rank-21 decomposition 
    # and extend it to handle the extra 2 columns in the 5-dimension.
    
    # Actually, let me try random search with restarts at various ranks.
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline for safety
    
    best_result = schoolbook()
    best_rank = rank_schoolbook
    
    # Try to find better decompositions
    # Use the fact that for 3x3x3, rank 21 works. For 3x3x5, 
    # we can think of it as 3x3x3 + 3x3x2, so rank <= 21 + 18 = 39 (schoolbook for 3x3x2)
    # But we might do better.
    
    # Let me try a more intelligent search.
    # The key constraint: for all i in [0,3), j in [0,3), k in [0,5):
    #   sum_r U[i*3+j][r] * V[j*5+k][r] * W[k*3+i][r] = 1
    
    # Try ranks from a lower estimate up to schoolbook
    # Lower bound for 3x3x5: at least max(9,15,15)=15, but realistically higher.
    # Let's try from 30 up.
    
    for target_rank in range(30, rank_schoolbook + 1):
        if time.time() > deadline - 30:
            break
        if target_rank >= best_rank:
            break
            
        # Try several random initializations
        found = False
        for attempt in range(20):
            if time.time() > deadline - 20:
                break
                
            # Random initialization with small integers/half-integers
            scale = 2.0
            U = [[random.choice([-scale, -scale/2, 0, scale/2, scale]) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.choice([-scale, -scale/2, 0, scale/2, scale]) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.choice([-scale, -scale/2, 0, scale/2, scale]) for _ in range(target_rank)] for _ in range(p*n)]
            
            if verify_full(U, V, W, target_rank):
                found = True
                best_rank = target_rank
                best_result = {"U": U, "V": V, "W": W}
                break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
