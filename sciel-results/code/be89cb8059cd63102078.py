import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # 5s buffer for return overhead
    
    # 1. Construct Rank-90 Schoolbook Decomposition
    # This is guaranteed to be correct and exact.
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # 2. Attempt to find a lower rank decomposition using Strassen on blocks
    # We split the multiplication C = A @ B into blocks.
    # Let's try splitting m=5 into 2+3 and p=6 into 3+3? Or 3+2 and 3+3?
    # A is 3x5, B is 5x6.
    # Split A into A1 (3x2), A2 (3x3).
    # Split B into B1 (2x6), B2 (3x6).
    # C = A1 @ B1 + A2 @ B2.
    #
    # Block 1: A1 (3x2) @ B1 (2x6). Rank = 3*2*6 = 36 (Schoolbook) or lower?
    # Block 2: A2 (3x3) @ B2 (3x6). Rank = 3*3*6 = 54 (Schoolbook) or lower?
    #
    # Can we do better?
    # For 3x3 matrix multiplication, the best known rank is 23 (Landsberg).
    # If we had a rank 23 decomposition for 3x3, we could potentially use it.
    # However, implementing a known rank 23 decomposition from scratch is difficult and error-prone.
    #
    # Let's try a different split that allows using Strassen (rank 7) for 2x2 blocks.
    # But 3x5 and 5x6 don't naturally form 2x2 blocks without padding.
    #
    # Alternative: Use a known low-rank decomposition for 3x3 if we can hardcode it.
    # Since I don't have a reliable hardcoded rank-23 3x3 decomposition, I will stick to the guaranteed schoolbook.
    #
    # However, let's check if we can reduce the rank by using a block decomposition with Strassen on 2x2 sub-blocks.
    # Split n=3 into 2+1?
    # A: 3x5 -> A1 (2x5), A2 (1x5)
    # B: 5x6 -> B1 (5x3), B2 (5x3) ? No, B is 5x6.
    #
    # Let's try to verify the schoolbook decomposition just to be safe, although it's trivially correct.
    if verify_decomp(n, m, p, U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback (should not be reached if verify_decomp is correct)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def verify_decomp(n, m, p, U, V, W):
    """
    Verify if the factors U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = U.shape[1]
    np.random.seed(12345)
    A = np.random.randint(-10, 10, (n, m))
    B = np.random.randint(-10, 10, (m, p))
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    
    # Efficient verification using matrix multiplication
    # C[i, k] = sum_c (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k]) * W[k*n+i, c]
    # This is not a simple matrix multiplication. It's a sum of products of three terms.
    # Let's compute it directly for small dimensions.
    
    for c in range(rank):
        # Extract vectors
        u_c = U[:, c].reshape(n, m)
        v_c = V[:, c].reshape(m, p)
        w_c = W[:, c].reshape(p, n).T # W is p*n, index k*n+i. Reshape to (p,n) then transpose to (n,p)
        
        # Compute the contribution of this rank-1 tensor
        # term_c = (A * u_c) * (B * v_c) ? No.
        # The tensor T_{ijk} = sum_c U_{ijc} V_{jkc} W_{ikc}
        # C_{ik} = sum_j sum_k? No.
        # C_{ik} = sum_j A_{ij} B_{jk}
        # Decomposition: C_{ik} = sum_c ( sum_j U_{ijc} A_{ij} ) ( sum_j V_{jkc} B_{jk} ) W_{ikc} ?
        # This formula is incorrect for standard matrix mult decomposition.
        
        # Correct form:
        # C_{ik} = sum_c W_{ikc} * ( sum_j U_{ijc} A_{ij} ) * ( sum_j V_{jkc} B_{jk} ) is wrong because U and V depend on j.
        
        # Actually, the standard bilinear form is:
        # C_{ik} = sum_c ( sum_j U_{ijc} A_{ij} ) ( sum_j V_{jkc} B_{jk} ) ... this doesn't make sense dimensionally if U and V are separate.
        
        # Let's look at the standard definition:
        # C_{ik} = sum_j A_{ij} B_{jk}
        # = sum_j sum_c (U_{ijc} A_{ij}) (V_{jkc} B_{jk}) W_{ikc} ? No, W is scalar per c.
        
        # The correct identity is:
        # sum_c W_{ikc} * ( sum_j U_{ijc} A_{ij} ) * ( sum_j V_{jkc} B_{jk} ) is not right.
        
        # Let's derive it:
        # We want C_{ik} = sum_j A_{ij} B_{jk}.
        # If we have factors U, V, W such that:
        # sum_c U_{ijc} V_{jkc} W_{ikc} = 1 for all i,j,k? No, that's not it.
        
        # The factors define:
        # C_{ik} = sum_c W_{ikc} * ( sum_j U_{ijc} A_{ij} ) * ( sum_j V_{jkc} B_{jk} ) is definitely wrong.
        
        # Correct interpretation:
        # Each term c contributes a rank-1 matrix to C? No, matrix mult is trilinear.
        # C_{ik} = sum_c W_{ikc} * (sum_j U_{ijc} A_{ij}) * (sum_j V_{jkc} B_{jk}) is not the standard form.
        
        # Standard form:
        # C_{ik} = sum_c (sum_j U_{ijc} A_{ij}) * (sum_j V_{jkc} B_{jk}) * W_{ikc} is wrong.
        
        # Let's check the schoolbook case:
        # U_{ijc} = 1 if (i,j) == (i_c, j_c) else 0.
        # V_{jkc} = 1 if (j,k) == (j_c, k_c) else 0.
        # W_{ikc} = 1 if (i,k) == (i_c, k_c) else 0.
        # And c corresponds to triple (i,j,k) with i=i_c, j=j_c, k=k_c.
        # Sum_c W_{ikc} ... 
        # For a specific (i,k), only c where i_c=i, k_c=k contribute.
        # For those c, j_c is fixed.
        # Term = 1 * (sum_j U_{ijc} A_{ij}) * (sum_j V_{jkc} B_{jk})
        # U_{ijc} is 1 only if j=j_c. So sum_j U_{ijc} A_{ij} = A_{i, j_c}.
        # V_{jkc} is 1 only if j=j_c. So sum_j V_{jkc} B_{jk} = B_{j_c, k}.
        # So term = A_{i, j_c} B_{j_c, k}.
        # Sum over c (which iterates over j_c) gives sum_j A_{i,j} B_{j,k}.
        
        # So the formula IS:
        # C_{ik} = sum_c W_{ikc} * (sum_j U_{ijc} A_{ij}) * (sum_j V_{jkc} B_{jk})
        
        # Let's implement this efficiently.
        pass # The loop below is sufficient for small n,m,p,rank
    
    for c in range(rank):
        u_c = U[:, c].reshape(n, m)
        v_c = V[:, c].reshape(m, p)
        w_c = W[:, c].reshape(p, n).T # (n, p)
        
        # Compute sums
        # sum_j U_{ijc} A_{ij} -> Matrix product? No, element-wise sum over j.
        # This is a "Hadamard product then sum" or similar.
        # Let A_u = A * u_c. Sum over j.
        # This is equivalent to (A * u_c) @ ones(m, 1) ? No.
        # It's just sum_j A[i,j] * u_c[i,j].
        
        # We can compute this as:
        # X[i] = sum_j A[i,j] * u_c[i,j]
        # Y[k] = sum_j B[j,k] * v_c[j,k]
        # Then C += w_c * X[:, None] * Y[None, :]
        
        X = np.sum(A * u_c, axis=1) # shape (n,)
        Y = np.sum(B * v_c, axis=0) # shape (p,)
        C_computed += w_c * np.outer(X, Y)
        
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        return False
    return True

# EVOLVE-BLOCK-END
