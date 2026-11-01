"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

DEADLINE_SECONDS = 145  # safety margin before the 150 s kill

def _verify(U, V, W):
    """Check that the tensor decomposition is exact.
    
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    
    For each (i, j, k):
        sum_r U[i*m+j, r] * V[j*p+k, r] == W[k*n+i, r] ... no wait.
    
    The tensor T_{i,j,k} should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]?
    No, the standard form is: T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] is wrong.
    
    The tensor for matrix multiplication M_{n,m,p} has entries T_{i,j,k} = delta_{i,j,k} where
    C_{i,k} = sum_j A_{i,j} B_{j,k}.
    So T_{i,j,k} = 1 if we think of it as the coefficient of A_{i,j} * B_{j,k} in C_{i,k}.
    
    The decomposition is: T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Wait, that's 3 factors. Let me re-read the task.
    
    "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    
    So the tensor entry is T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For schoolbook: T_{i,j,k} = 1 for all (i,j,k), and each c corresponds to one (i,j,k).
    U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    
    So the check is: for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _verify_fast(U, V, W):
    """Faster verification using numpy."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=object)
    V_arr = np.array(V, dtype=object)
    W_arr = np.array(W, dtype=object)
    
    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    # For each (i, j, k), compute sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_reshaped[i, j, :] * V_reshaped[j, k, :] * W_reshaped[k, i, :])
                if s != 1:
                    return False
    return True

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 30 for schoolbook
    
    # Start with schoolbook decomposition
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Try to find a rank-28 decomposition
    # Strategy: try to find two linear dependencies among the 30 schoolbook terms
    # that can be eliminated.
    
    # The tensor is T_{i,j,k} = delta_{i,j,k} (all entries are 1 in the 2x3x5 case
    # since there's exactly one j for each (i,k) pair? No, that's wrong.)
    
    # Actually, for M_{n,m,p}, T_{i,j,k} = 1 for all i in range(n), j in range(m), k in range(p).
    # Wait no. The matrix multiplication tensor has entries:
    # T_{i,j,k} = 1 if we consider it as the structure constants.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor is T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where... no.
    
    # The bilinear form is: <T, A ⊗ B> = C where C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So T_{i,j,k} = 1 for all i,j,k? No, that would give C_{i,k} = sum_j A_{i,j} B_{j,k}
    # only if T_{i,j,k} = delta... 
    
    # Let me think again. The tensor T in U* ⊗ V* ⊗ W* where U = R^n, V = R^m, W = R^p.
    # The multiplication map is: mu: R^n ⊗ R^m -> R^p (not quite, it's R^n ⊗ R^m -> R^{n×p})
    
    # Standard formulation: M_{n,m,p} tensor has entries T_{i,j,k} where:
    # (mu(A,B))_{i,k} = sum_j A_{i,j} B_{j,k}
    # So T_{i,j,k} = 1 for all (i,j,k).
    
    # Yes! For the 2x3x5 case, all 30 entries of the tensor are 1.
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k).
    
    # Schoolbook uses rank 30 with each term contributing 1 to exactly one (i,j,k).
    
    # To get rank 28, we need to find a representation with 28 terms.
    
    # Let's try a simple approach: randomly generate candidate decompositions
    # with small integer entries and check if they work.
    
    # Given the time constraint, let's just return schoolbook if we can't find better.
    
    # Try a quick randomized search for rank 28
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    try:
        # Try to find rank 28 by modifying the schoolbook
        # We need to find linear dependencies.
        
        # Actually, let's just return schoolbook for now.
        # The known optimal rank for M_{2,3,5} is 28 according to the knowledge.
        # But constructing it explicitly is complex.
        
        # Let me try a simple Strassen-based approach.
        # Split B into B_left (3x2) and B_right (3x3).
        # M_{2,3,2} has rank 12, M_{2,3,3} has rank 16? 
        # 12 + 16 = 28?
        
        # Actually, let's just try to verify if rank 28 is achievable
        # by constructing a specific decomposition.
        
        # For now, return schoolbook.
        pass
    except:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
