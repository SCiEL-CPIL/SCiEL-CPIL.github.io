"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Schoolbook decomposition as baseline
    schoolbook_rank = n * m * p
    
    def _schoolbook():
        rank = n * m * p
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
        return {"U": U, "V": V, "W": W}
    
    # Try to find a better decomposition using known results or search
    # For 3x3x8, we can try to use the fact that 8 = 2*4 or similar structures
    # The known minimum rank for 3x3x3 is 23 (Landsberg's bound is 19, but 23 is known)
    # For 3x3x8, we could potentially do better than 72
    
    # Strategy: Try a structured decomposition
    # We know that matrix multiplication can sometimes be decomposed using 
    # Kronecker products or block structures
    
    # Let's try to construct a decomposition of rank < 72
    # One approach: use the fact that 3x3x8 = 3x3x(2*4)
    
    # Actually, let me think about this differently.
    # The tensor for 3x3x8 has dimensions 9 x 24 x 24 (n*m x m*p x p*n)
    # Schoolbook gives rank 72.
    
    # Can we do better? Let's try a random search approach with verification.
    # But random search for exact integer decompositions is very hard.
    
    # Instead, let's try a structured approach based on known matrix multiplication decompositions.
    # For 3x3x3, the best known rank is 23.
    # For 3x3x8, one idea is to use 3x3x4 twice (if we know 3x3x4), 
    # or use other known decompositions.
    
    # Let me try a different approach: use the Coppersmith-Winograd style 
    # or other structured decompositions.
    
    # Actually, the simplest improvement might be to check if there's a 
    # known decomposition for small cases that we can exploit.
    
    # For now, let's try a simple heuristic: if we can find any valid decomposition
    # with rank < 72, use it. Otherwise return schoolbook.
    
    # Let me try to construct a decomposition using the fact that 
    # 3 = 2 + 1 and use block matrix multiplication.
    
    # A 3x3 matrix can be viewed as a 2x2 block matrix with some structure.
    # Let's try to use rank 23 for 3x3x3 and extend to 3x3x8.
    
    # Actually, let's just try to see if we can find a valid decomposition
    # with a specific rank using a systematic approach.
    
    # For 3x3x8, the theoretical lower bound is around 19-20 per 3x3x3 block,
    # so 3x3x8 might have rank around 38-50.
    
    # Let me try a pragmatic approach: generate random matrices and check if 
    # a low-rank decomposition works. But this is computationally expensive.
    
    # Given the time constraint, let's stick with schoolbook unless we find
    # a quick improvement.
    
    # Let me try one specific idea: use the identity that 
    # multiplying by 8 can be done with some sharing.
    
    # Actually, I'll just return the schoolbook decomposition to be safe.
    # The task requires EXACT decomposition, and any error scores 0.
    
    if time.time() > deadline - 5.0:
        return _schoolbook()
    
    # Try a quick random search for improvement (very limited)
    np.random.seed(42)
    best_result = _schoolbook()
    best_rank = schoolbook_rank
    
    # Try rank 60, 50, 40 with random initialization
    for target_rank in [64, 56, 48]:
        if time.time() > deadline - 10.0:
            break
        try:
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Verify: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal delta_{i,k}... 
            # Wait, no. The tensor T[i,j,k] = 1 if the multiplication A[i,j]*B[j,k] contributes to C[i,k].
            # Actually, the matrix multiplication tensor M[i,k,(j)] where we compute C[i,k] = sum_j A[i,j]*B[j,k]
            # The decomposition is: M = sum_c U_c ⊗ V_c ⊗ W_c
            # where U_c is a vector in R^{n*m}, V_c in R^{m*p}, W_c in R^{p*n}
            # and the tensor entry T[i,j,k] = sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i]
            # This should equal 1 if j is the "middle" index connecting i and k, else 0.
            # More precisely: T[i,j,k] = delta_{i,k...} no.
            
            # Let me clarify: For matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p:
            # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
            # The tensor M has entries M[i,j,k] = 1 for all valid i,j,k (since each term A[i,j]*B[j,k] 
            # appears exactly once in the sum for C[i,k] with middle index j).
            # Wait, that's not right either. The tensor is over (i, j, k) where:
            # - i indexes rows of A and C (0 to n-1)
            # - j indexes columns of A and rows of B (0 to m-1)  
            # - k indexes columns of B and C (0 to p-1)
            # And M[i,j,k] = 1 for all i,j,k because every combination contributes.
            
            # Actually no. The standard tensor decomposition of matrix multiplication:
            # The multiplication tensor M_{n,m,p} has entries M_{i,j,k} where:
            # i in [0,n), j in [0,m), k in [0,p)
            # M_{i,j,k} = 1 (it's just the all-ones tensor in this indexing)
            
            # Wait, I think I'm confusing myself. Let me re-read the problem.
            # The output factor W follows W[k*n+i, r]. So W is indexed by (k, i) where k is column of C/B and i is row of A/C.
            # U is indexed by (i*m+j, r) where i is row of A, j is col of A/row of B
            # V is indexed by (j*p+k, r) where j is row of B, k is col of B
            # W is indexed by (k*n+i, r) where k is col of C, i is row of C
            
            # The decomposition satisfies:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
            # because C[i,k] = sum_j A[i,j] * B[j,k], and each (i,j,k) triple contributes exactly 1.
            
            # So we need: for all i in [0,n), j in [0,m), k in [0,p):
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # Let's verify this condition
            valid = True
            for i in range(n):
                if time.time() > deadline - 5.0:
                    valid = False
                    break
                for j in range(m):
                    if time.time() > deadline - 5.0:
                        valid = False
                        break
                    for k in range(p):
                        val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(target_rank))
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
                
            # If valid, convert to list format
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = target_rank
            print(f"Found valid decomposition with rank {target_rank}")
            
        except Exception as e:
            continue
    
    return best_result
# EVOLVE-BLOCK-END
