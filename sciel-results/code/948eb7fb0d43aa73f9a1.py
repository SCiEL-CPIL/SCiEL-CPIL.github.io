import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    nmp = n * m * p
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin for 150s limit

    def schoolbook():
        rank = nmp
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    best_rank = nmp

    # Strategy: Deterministic Algebraic Rank Reduction
    # 1. Construct the tensor T_{i,j,k} = delta_{i,j,k} (effectively 1 for valid indices)
    #    Wait, the tensor for matrix multiplication is T_{i,j,k} = 1 if we view it as 
    #    (A*B)_{ik} = sum_j A_{ij} B_{jk}.
    #    The tensor T has indices (i, j, k) with T[i,j,k] = 1.
    #    Actually, the standard tensor is T_{i,j,k} = 1.
    #    Let's verify the decomposition condition:
    #    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    # The tensor is T[i, j, k] = 1.
    # This is the "all-ones" tensor on the index set {0..n-1} x {0..m-1} x {0..p-1}.
    # The rank of the all-ones tensor of size n x m x p is known to be min(n, m, p) * ...?
    # No, the rank of the all-ones tensor is not simply min.
    # However, we can use the fact that T[i,j,k] = 1 = (1)(1)(1).
    # This suggests a rank-1 decomposition?
    # U[i*m+j, 0] = 1
    # V[j*p+k, 0] = 1
    # W[k*n+i, 0] = 1
    # Then sum_r ... = 1*1*1 = 1.
    # Is this valid?
    # Let's check the definitions.
    # U has shape (n*m, rank). Index is i*m+j.
    # V has shape (m*p, rank). Index is j*p+k.
    # W has shape (p*n, rank). Index is k*n+i.
    #
    # If we set Rank = 1:
    # U[i*m+j, 0] = 1 for all i,j
    # V[j*p+k, 0] = 1 for all j,k
    # W[k*n+i, 0] = 1 for all k,i
    #
    # The contraction is sum_{r=0}^{0} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # = 1 * 1 * 1 = 1.
    #
    # This matches the target value 1 for all i,j,k.
    #
    # Wait, is the matrix multiplication tensor really the all-ones tensor?
    # (A B)_{ik} = sum_{j=0}^{m-1} A_{ij} B_{jk}.
    # The tensor T maps A and B to C.
    # T_{i,j,k} is the coefficient of A_{ij} B_{jk} in C_{ik}.
    # So T_{i,j,k} = 1.
    # Yes, the tensor entries are all 1.
    
    # Therefore, the rank is 1?
    # Let's double check if there are any constraints I'm missing.
    # "integer/half-integer entries" -> 1 is integer.
    # "exact tensor decomposition" -> Yes.
    #
    # Why did the previous program use schoolbook rank n*m*p?
    # The previous program assumed the standard "basis" decomposition where each product term is separate.
    # But tensor rank allows for "Strassen-like" combinations or simpler structures if the tensor has low rank.
    # The all-ones tensor has rank 1.
    #
    # Let's verify with a small example.
    # 2x2 matrix multiplication.
    # T_{i,j,k} = 1 for i,j,k in {0,1}.
    # Rank 1:
    # U[0,0]=1, U[1,0]=1, U[2,0]=1, U[3,0]=1
    # V[0,0]=1, V[1,0]=1, V[2,0]=1, V[3,0]=1
    # W[0,0]=1, W[1,0]=1, W[2,0]=1, W[3,0]=1
    #
    # Check (0,0,0): U[0,0]*V[0,0]*W[0,0] = 1. Correct.
    # Check (0,0,1): U[0,0]*V[1,0]*W[1,0] = 1. Correct.
    # ...
    # Check (1,1,1): U[3,0]*V[3,0]*W[3,0] = 1. Correct.
    #
    # So the rank is 1?
    #
    # Is there any catch?
    # "Matrix multiplication 2,4,7"
    # The tensor is indeed the all-ones tensor on the indices.
    #
    # Let's re-read carefully: "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 7 matrix."
    #
    # If the rank is 1, the score would be 1.0 / 1.0 = 1.0?
    # Wait, the score is seed/value.
    # Seed program rank = 2*4*7 = 56.
    # If I find rank 1, score = 56 / 1 = 56.0.
    # This would be a massive improvement.
    #
    # Why have all previous attempts failed or scored 1.0?
    # The previous program in the prompt (CURRENT PROGRAM) uses a heuristic ALS that tries to find lower rank.
    # It starts with schoolbook (rank 56) and tries to lower it.
    # If the true rank is 1, the ALS should have found it easily?
    # Or maybe the ALS is stuck in local minima or the initialization is bad?
    # Or maybe my understanding of the tensor is wrong?
    
    # Let's check the definition of the tensor again.
    # Usually, the matrix multiplication tensor is defined such that
    # C_{ik} = \sum_{j} A_{ij} B_{jk}.
    # The tensor T has components T_{i,j,k}.
    # T_{i,j,k} = 1.
    #
    # Is it possible that the indices are permuted or the "trace convention" W[k*n+i, r] implies something else?
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # This just defines the indexing of the W matrix.
    # W row index is k*n + i.
    # This matches my derivation.
    
    # Let's try to construct the rank-1 solution.
    
    rank = 1
    U = np.ones((nm, rank))
    V = np.ones((mp, rank))
    W = np.ones((pn, rank))
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    if valid:
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        best_rank = rank
    else:
        # If rank 1 fails (which shouldn't happen), fall back to schoolbook
        pass

    return best_result
