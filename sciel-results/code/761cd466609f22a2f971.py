import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # 5s buffer for return overhead
    
    # 1. Construct Rank-90 Schoolbook Decomposition as fallback
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Verify schoolbook
    if verify_decomp(n, m, p, U_school, V_school, W_school):
        best_U, best_V, best_W = U_school, V_school, W_school
        best_rank = rank_schoolbook
    else:
        # Should not happen, but keep going
        best_U, best_V, best_W = U_school, V_school, W_school
        best_rank = rank_schoolbook
    
    # 2. Try block decomposition with Strassen for 2x2 blocks
    # Split m=5 into 2+3, and use Strassen on the 2x2 part
    # A (3x5) = [A1 (3x2), A2 (3x3)]
    # B (5x6) = [B1 (2x6); B2 (3x6)]
    # C = A1 @ B1 + A2 @ B2
    
    # For A1 @ B1: 3x2 @ 2x6
    # We can pad A1 to 3x3 and B1 to 3x6, but that's wasteful.
    # Instead, let's try a different approach: split n=3 into 1+2
    # A (3x5) = [A1 (1x5); A2 (2x5)]
    # B (5x6) = B
    # C = A1 @ B + A2 @ B
    # A1 @ B: 1x5 @ 5x6 -> rank 30 schoolbook
    # A2 @ B: 2x5 @ 5x6 -> rank 60 schoolbook
    # Total: 90 (same as before)
    
    # Let's try Strassen on 2x2 blocks by padding
    # Pad n from 3 to 4, m from 5 to 6, p from 6 to 6
    # Then use Strassen recursively? That's complex.
    
    # Alternative: Use the fact that 2x2 @ 2x2 has rank 7 (Strassen)
    # Split m=5 into 2+2+1
    # A (3x5) = [A1 (3x2), A2 (3x2), A3 (3x1)]
    # B (5x6) = [B1 (2x6); B2 (2x6); B3 (1x6)]
    # C = A1 @ B1 + A2 @ B2 + A3 @ B3
    # A1 @ B1: 3x2 @ 2x6. Pad to 3x3 @ 3x6? Or use a 2x2 Strassen-like approach?
    # For 3x2 @ 2x6, we can think of it as 3x3 @ 3x6 with the 3rd column/row zero.
    # The rank of 3x2 @ 2x6 is at most 3*2*6 = 36, but can we do better?
    # Actually, 3x2 @ 2x6 = (3x2) @ (2x6). We can write this as:
    # Let M = A1 (3x2), N = B1 (2x6)
    # C = M @ N
    # We can use Strassen if we pad M to 3x3 and N to 3x6, but that adds overhead.
    
    # Let's try a simpler approach: use known low-rank decompositions for small cases
    # For 2x2 @ 2x2, Strassen gives rank 7
    # For 3x3 @ 3x3, best known is 23 (Landsberg) but hard to implement
    
    # Let's try to construct a decomposition using Strassen on 2x2 blocks
    # Split m=5 into 2+2+1
    # A = [A1, A2, A3] where A1 is 3x2, A2 is 3x2, A3 is 3x1
    # B = [B1; B2; B3] where B1 is 2x6, B2 is 2x6, B3 is 1x6
    # C = A1 @ B1 + A2 @ B2 + A3 @ B3
    
    # For A3 @ B3: 3x1 @ 1x6, rank = 3*1*6 = 18 (schoolbook)
    # For A1 @ B1: 3x2 @ 2x6
    # We can pad A1 to 3x3 by adding a zero column, and B1 to 3x6 by adding a zero row
    # Then use Strassen for 3x3 @ 3x6? No, Strassen is for square matrices.
    
    # Let's just use schoolbook for now but verify it's correct
    if time.time() < deadline:
        # Try to find a better decomposition using random search
        # This is a heuristic approach that may not find the optimal rank
        pass
    
    # Return the best decomposition found
    if verify_decomp(n, m, p, best_U, best_V, best_W):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback to schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

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
