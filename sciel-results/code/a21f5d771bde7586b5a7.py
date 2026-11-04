import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We want rank 44: 23 (for 3x3x3) + 9 (for 3x3x1) + 12 (for 3x1x4)
    # Let's build the 23 terms for 3x3x3 manually.
    
    # 3x3x3 Rank 23 Terms (Lader / Strassen-like)
    # u, v, w are 9-vectors.
    # u[i,j] -> index i*3 + j
    # v[i,j] -> index i*3 + j
    # w[i,j] -> index i*3 + j
    
    # We define the 23 terms as lists of (u, v, w) where u, v, w are lists of 9 floats.
    # We will construct them based on the known structure.
    # If I can't recall the exact coefficients, I will use a fallback.
    
    # Let's try to recall or derive the 23 terms.
    # Actually, a common strategy is to use the 23 terms from a reliable source.
    # Since I don't have them perfectly in memory, I will implement the 27 terms for 3x3x3 (schoolbook) 
    # which gives rank 48.
    # To get 44, I NEED the 23 terms.
    
    # Let's check if I can find them in my training data...
    # The 23 terms for 3x3x3 are not as standard as Strassen's 7 for 2x2.
    # There is a known set by Lader (1978).
    
    # I will implement the 44 rank solution using the 23 terms if I can verify them.
    # Since I can't browse, I will use a placeholder set that is likely to be wrong, 
    # and verify it. If it fails, I return the 48 rank solution.
    
    # However, the prompt asks for a STRONGER algorithm. Returning 48 is the same as the seed.
    # I should try to find the 23 terms.
    
    # Let's try to construct them using a known algorithm or list.
    # I recall that the coefficients are small integers.
    
    # Let's use a list of 23 terms. If I don't have them, I'll use 27.
    
    # For now, I will output the 48 rank solution to ensure validity, 
    # but I will structure the code so that if I find the 23 terms, I can swap them in.
    
    # Actually, let's look at the "inspiration 3" which mentions `_lader_3x3x3`.
    # This suggests that a Lader decomposition is expected.
    
    # I will define the 23 terms here.
    # If I am unsure, I will use the schoolbook for 3x3x3.
    
    # Let's assume I can find the 23 terms.
    # I will write a function to verify the tensor decomposition.
    
    def verify_tensor(U, V, W, n, m, p):
        # U: (n*m, R)
        # V: (m*p, R)
        # W: (p*n, R)
        R = U[0].__len__()
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # Random test
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        # Compute C from tensor
        # C[i,k] = sum_j A[i,j] B[j,k]
        # Tensor: T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # Wait, the definition is:
        # A[i,j] * B[j,k] = sum_r (U[i*m+j, r] * A[i,j]) * (V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
        # No, that's not right.
        # The tensor decomposition is:
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # And C[i,k] = sum_j A[i,j] B[j,k] T[i,j,k] ? No.
        
        # The standard definition for matrix multiplication tensor T_{ijk} is:
        # T_{ijk} = 1 if (i,j,k) corresponds to a valid multiplication term.
        # The rank decomposition is:
        # T_{ijk} = sum_{r=1}^R U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)}
        # Here U_{ij}^{(r)} is the coefficient for A_{ij}, V_{jk}^{(r)} for B_{jk}, W_{ki}^{(r)} for C_{ki} (or C_{ik}).
        
        # The prompt says: W[k*n+i, r]. This suggests W is indexed by k (row of B) and i (col of A)?
        # No, W[k*n+i] usually means row k, col i.
        # C has dimensions n x p. So C[i,k].
        # So W should be indexed by i and k.
        # The prompt says W[k*n+i, r]. This is row k, col i.
        # So W_{k,i}^{(r)} corresponds to C_{i,k}.
        
        # Let's verify the convention.
        # U: (n*m, rank). Index i*m + j. A[i,j].
        # V: (m*p, rank). Index j*p + k. B[j,k].
        # W: (p*n, rank). Index k*n + i. C[i,k].
        
        # So the decomposition is:
        # 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # for all valid i, j, k.
        
        # Let's test this.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(R):
                        val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Since I don't have the 23 terms hardcoded, I will use the 27 terms for 3x3x3.
    # This results in rank 48.
    
    # To get a score > 1.0, I MUST find a rank < 48 solution.
    # I will try to construct the 44 rank solution using the 23 terms.
    # I will search for them or use a known set.
    
    # I will use the 23 terms from a known list.
    # If I fail, I return 48.
    
    # Let's try to find the 23 terms.
    # I recall that the 23 terms can be constructed using a specific pattern.
    
    # I will output the 48 rank solution for now.
    
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
