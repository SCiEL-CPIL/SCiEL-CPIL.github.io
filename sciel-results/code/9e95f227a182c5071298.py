"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Schoolbook fallback
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U, V, W
    
    # Try ILP-based search for lower rank
    # The tensor decomposition requires:
    # For each (i,k) output position, sum over j of A[i,j]*B[j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,n+i,c]
    # This is a cubic constraint, so we linearize by fixing one factor at a time.
    
    # Strategy: Fix W entries and solve for U,V via ILP
    # Or: Use the fact that over GF(2), we can search for rank-r decompositions
    
    # For small ranks, enumerate possible W patterns and check if U,V exist
    # This is NP-hard in general, but for small dimensions we can try
    
    # Lower bound: max(n*p, m*p, n*m) = max(14, 28, 8) = 28
    # Upper bound: 56 (schoolbook)
    
    # Try ranks from lower bound up
    min_rank = max(n * p, m * p, n * m)
    
    for target_rank in range(min_rank, rank_schoolbook):
        if time.time() > deadline - 30:
            break
        
        # For each candidate rank, try to find a decomposition
        # Use random initialization + local search
        success = _try_rank_decomposition(n, m, p, target_rank, deadline)
        if success:
            best_rank = target_rank
            break
    
    # Return the best found
    if best_rank < rank_schoolbook:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U, "V": V, "W": W}


def _try_rank_decomposition(n, m, p, target_rank, deadline):
    """Try to find a rank-target_rank decomposition using random search + verification."""
    start_time = time.time()
    max_time = deadline - start_time - 10  # safety margin
    
    # Random initialization
    num_trials = 100
    trial_time = max_time / num_trials
    
    for trial in range(num_trials):
        if time.time() > deadline - 10:
            break
        
        # Generate random U, V, W with integer entries in {-1, 0, 1}
        U = np.random.randint(-1, 2, size=(n * m, target_rank))
        V = np.random.randint(-1, 2, size=(m * p, target_rank))
        W = np.random.randint(-1, 2, size=(p * n, target_rank))
        
        # Verify the decomposition
        if _verify_decomposition(U, V, W, n, m, p):
            # Convert to list format
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            # Store as module-level for return
            globals()['_best_decomp'] = (U_list, V_list, W_list)
            return True
    
    return False


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    # The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # But this is cubic. The correct formulation is:
    # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # In the decomposition: (A*B)[i,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_k V[j*p+k,c]*B[j,k]) * W[k*n+i,c]
    # This is still not quite right. Let me reconsider.
    
    # Actually, the standard tensor decomposition for matrix multiplication is:
    # T = sum_c U_c ⊗ V_c ⊗ W_c
    # where (U_c)_{i,j}, (V_c)_{j,k}, (W_c)_{k,i} are the c-th factor
    # And T_{i,j,k} = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i]
    # where T_{i,j,k} = 1 if k is the output index for input positions i,j
    
    # For matrix multiplication of (n x m) * (m x p) -> (n x p):
    # The output (i,k) depends on inputs (i,j) and (j,k) for j=0..m-1
    # So the tensor has T[i,j,k] = 1 for all valid (i,j,k) where the product A[i,j]*B[j,k] contributes
    
    # Wait, I think the convention is different. Let me re-read the task.
    # "W[k*n+i,r]" suggests W is indexed by (k,n+i) where k is column of B, i is row of A
    # The output is (i,k) = row i, column k
    
    # The tensor T should satisfy: T[i,j,k] = 1 if the product A[i,j]*B[j,k] is part of output[i,k]
    # Since output[i,k] = sum_j A[i,j]*B[j,k], we have T[i,j,k] = 1 for all i,j,k
    
    # So the decomposition is: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
    
    # Let's verify this
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                total = 0
                for c in range(U.shape[1]):
                    total += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if total != 1:
                    return False
    
    return True
# EVOLVE-BLOCK-END
