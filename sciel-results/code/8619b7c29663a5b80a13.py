"""Matrix multiplication 2x5x5 rank-minimal exact tensor decomposition."""

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=5
    # We can decompose the 2x5x5 multiplication into two independent 1x5x5 multiplications
    # by using the fact that the first dimension is 2.
    # Actually, a better approach:
    # The tensor is M_{i,j,k} where i in {0,1}, j in {0..4}, k in {0..4}
    # This is equivalent to two independent 1x5x5 tensors for i=0 and i=1.
    # The rank of 1x5x5 matrix multiplication is 5 (since it's just element-wise multiplication of vectors, 
    # but wait, 1x5 * 5x5 = 1x5, so C_k = sum_j A_j B_jk. This is rank 5).
    # So total rank would be 5 + 5 = 10.
    
    # Let's verify: 
    # For i=0: C_0k = sum_j A_j B_jk
    # For i=1: C_1k = sum_j A'_j B'_jk (if we separate the rows)
    # But actually the input is a 2x5 matrix A and 5x5 matrix B.
    # C = A * B, where C is 2x5.
    # C_ik = sum_j A_ij B_jk
    # This is two independent dot products for each k.
    # For each k, we compute two dot products of length 5.
    # Each dot product of length 5 has rank 5.
    # So total rank is 5 (for i=0) + 5 (for i=1) = 10.
    
    rank = 10
    U = [[0] * rank for _ in range(n * m)]  # 10 x 10
    V = [[0] * rank for _ in range(m * p)]  # 25 x 10
    W = [[0] * rank for _ in range(p * n)]  # 10 x 10
    
    # For i=0 (first row of A), rank indices 0-4
    for j in range(m):
        for k in range(p):
            # Term for (i=0, j, k)
            r = j  # 0 to 4
            U[i * m + j][r] = 1  # A_0j
            V[j * p + k][r] = 1  # B_jk
            W[k * n + i][r] = 1  # C_0k
            # Wait, this doesn't work because for fixed i=0, we need:
            # C_0k = sum_j A_0j B_jk
            # So for each k, we have a sum over j.
            # The rank terms should be indexed by (i, j) for the U part? No.
            # Let's re-think.
            # We have rank 10. Let's say terms r=0..4 correspond to i=0, j=0..4
            # and r=5..9 correspond to i=1, j=0..4.
            # For term r (0<=r<5): i=0, j=r
            #   U[0*m + r][r] = 1  (A_0r)
            #   V[r*p + k][r] = 1 for all k? No, V is (m*p, rank).
            #   V index is j*p + k. For fixed j=r, V[r*p + k][r] should be B_rk.
            #   But we can't have different values for different k in the same term.
            #   
            # I made a mistake. The rank of 1x5x5 is not 5 in the tensor decomposition sense.
            # Let's reconsider.
            # The tensor for 1x5x5 is T_{j,k} = A_j B_k? No, it's T_{j,k} = A_j B_{j,k} where B is 5x5.
            # Actually, the multiplication is: A (1x5) times B (5x5) = C (1x5)
            # C_k = sum_j A_j B_jk
            # The tensor has indices (j, k) for the sum, and the output is indexed by k.
            # Wait, the standard matrix multiplication tensor has indices (i, j, k) where
            # C_ik = sum_j A_ij B_jk.
            # For 1x5x5: i is fixed (i=0), so C_k = sum_j A_j B_jk.
            # The tensor is T_{j,k} -> contributes to C_k.
            # This is a bilinear form. The rank is the minimum number of terms of the form
            # (sum_j u_j A_j)(sum_k v_k B_jk) w_k? No.
            # 
            # Let's use the standard decomposition.
            # For 1x5x5, the trivial decomposition has rank 5*5=25? No.
            # C_k = sum_j A_j B_jk
            # This can be written as sum over j of A_j * B_jk.
            # Each term j contributes to all k.
            # So we can have 5 terms, one for each j.
            # Term j: u_j = A_j, v_{jk} = B_jk, w_k = 1 for all k? No.
            # The decomposition is: sum_r (sum_i u_{ir} A_i)(sum_j v_{jr} B_j) w_r
            # For 1x5x5: A has 5 elements, B has 25 elements, C has 5 elements.
            # C_k = sum_j A_j B_jk
            # = sum_j (A_j)(B_jk)
            # So for each j, we have a term where:
            #   u_j = A_j (coefficient 1)
            #   v_{j,k} = B_jk (coefficient 1 for this j, 0 for others)
            #   w_k = 1 for all k? No, w is a vector that multiplies the result.
            # 
            # Actually, in the tensor decomposition for matrix multiplication:
            # C_ik = sum_j A_ij B_jk = sum_r U_{ir} V_{jr} W_{kr}
            # Wait, the standard form is:
            # C_{ik} = sum_j A_{ij} B_{jk}
            # The tensor T_{ijk} has T_{ijk} = 1 if the indices match the multiplication.
            # Decomposition: T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}
            # For 1x5x5: i is fixed to 0.
            # T_{0,j,k} = 1 for all j,k (since C_k = sum_j A_j B_jk, each term A_j B_jk contributes to C_k)
            # So we need: 1 = sum_r U_{0r} V_{jr} W_{kr} for all j,k.
            # 
            # If we use rank 5, with r=j:
            # U_{0r} = 1 for r=0..4
            # V_{jr} = 1 if j=r, 0 otherwise
            # W_{kr} = 1 for all k,r? Then sum_r U_{0r} V_{jr} W_{kr} = sum_r 1 * delta_{jr} * 1 = 1.
            # Yes! So for 1x5x5, rank 5 works.
            # 
            # So for 2x5x5, we can use rank 10:
            # Terms r=0..4 for i=0, r=5..9 for i=1.
            # For r=0..4 (i=0):
            #   U_{0,r} = 1, U_{1,r} = 0
            #   V_{j,r} = 1 if j=r, 0 otherwise
            #   W_{k,r} = 1 for all k
            # For r=5..9 (i=1):
            #   U_{0,r} = 0, U_{1,r} = 1
            #   V_{j,r} = 1 if j=r-5, 0 otherwise
            #   W_{k,r} = 1 for all k
            # 
            # Let's verify:
            # C_{0,k} = sum_j A_{0,j} B_{j,k} = sum_r U_{0,r} (sum_j V_{j,r} B_{j,k}) W_{k,r}
            # Wait, the decomposition is: C_{ik} = sum_r (sum_j U_{ij} ... no.
            # 
            # The correct form is:
            # C_{ik} = sum_r U_{ir} * (sum_j V_{jr} B_{jk}) * W_{kr}? No.
            # 
            # Standard: C_{ik} = sum_j A_{ij} B_{jk}
            # Decomposition: sum_r (sum_i U_{ir} A_{i}) * (sum_j V_{jr} B_{j}) ... no.
            # 
            # Actually: C_{ik} = sum_r U_{ir} V_{kr} (sum_j A_{ij} ... no.
            # 
            # Let me look at the seed code:
            # U[i*m+j][column] = 1  # A_ij
            # V[j*p+k][column] = 1  # B_jk
            # W[k*n+i][column] = 1  # C_ik
            # So the decomposition is: C_{ik} = sum_r U_{ij,r} V_{jk,r} W_{ik,r}
            # Where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
            # 
            # So for our case:
            # C_{ik} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
            # 
            # For i=0, k fixed:
            # C_{0,k} = sum_j A_{0,j} B_{j,k}
            # = sum_r (sum_j U_{0*m+j, r} A_{0,j}) * ... no, it's a trilinear form.
            # 
            # Actually, the product is:
            # sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
            # This sums over j and r for each (i,k).
            # 
            # For our proposed rank 10 decomposition:
            # r=0..4: 
            #   U_{0*m+j, r} = 1 if j=r, 0 otherwise (for i=0)
            #   U_{1*m+j, r} = 0
            #   V_{j*p+k, r} = 1 if j=r, 0 otherwise (for all k)
            #   W_{k*n+0, r} = 1 for all k
            #   W_{k*n+1, r} = 0
            # 
            # r=5..9 (let r' = r-5, 0..4):
            #   U_{0*m+j, r} = 0
            #   U_{1*m+j, r} = 1 if j=r', 0 otherwise
            #   V_{j*p+k, r} = 1 if j=r', 0 otherwise (for all k)
            #   W_{k*n+0, r} = 0
            #   W_{k*n+1, r} = 1 for all k
            # 
            # Let's check C_{0,k}:
            # sum_r U_{0*m+j, r} V_{j*p+k, r} W_{k*n+0, r}
            # = sum_{r=0}^4 U_{j, r} V_{j*p+k, r} * 1 + sum_{r=5}^9 0 * ... * 0
            # = sum_{r=0}^4 (1 if j=r else 0) * (1 if j=r else 0) * 1
            # = sum_{r=0}^4 delta_{jr} * delta_{jr}
            # = sum_{r=0}^4 delta_{jr}
            # = 1 if j is in 0..4, which is always true.
            # But we sum over j!
            # C_{0,k} = sum_j sum_r U_{0*m+j, r} V_{j*p+k, r} W_{k*n+0, r}
            # = sum_j sum_{r=0}^4 delta_{jr} * delta_{jr} * 1
            # = sum_j sum_{r=0}^4 delta_{jr}
            # = sum_j 1 (since for each j, there's exactly one r=j)
            # = 5
            # But we want C_{0,k} = sum_j A_{0,j} B_{j,k}.
            # 
            # I see the issue. The U, V, W matrices multiply the input matrices.
            # The actual computation is:
            # C = sum_r (U_r^T A) * (V_r^T B) * W_r^T? No.
            # 
            # Let's re-read the contract:
            # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
            # "exactly decompose the matrix multiplication tensor"
            # 
            # The tensor T_{i,j,k} for matrix multiplication satisfies:
            # C_{ik} = sum_j A_{ij} B_{jk}
            # The decomposition is:
            # T_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
            # 
            # So for the trivial decomposition, U_{i*m+j, r} = 1 if r = i*m+j, else 0, etc.
            # 
            # For our rank 10 attempt, we need:
            # sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = 1 if (i,j,k) is a valid multiplication term, else 0.
            # But actually, T_{i,j,k} = 1 for all i,j,k in the range, because each A_{ij} B_{jk} contributes to C_{ik}.
            # Wait no, T_{i,j,k} is the coefficient of A_{ij} B_{jk} in C_{ik}, which is 1.
            # 
            # So we need: sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = 1 for all i,j,k.
            # 
            # With our proposed decomposition:
            # For i=0, j, k:
            # sum_r U_{j, r} V_{j*p+k, r} W_{k, r}
            # = sum_{r=0}^4 delta_{jr} * delta_{jr} * 1 = 1 (since j is in 0..4)
            # 
            # For i=1, j, k:
            # sum_r U_{5+j, r} V_{j*p+k, r} W_{k+2, r}  (since n=2, k*n+i = k*2+1)
            # = sum_{r=5}^9 delta_{j, r-5} * delta_{j, r-5} * 1 = 1
            # 
            # So it works! The rank is 10.
            
    rank = 10
    U = [[0] * rank for _ in range(n * m)]  # 10 x 10
    V = [[0] * rank for _ in range(m * p)]  # 25 x 10
    W = [[0] * rank for _ in range(p * n)]  # 10 x 10
    
    # For r=0..4 (i=0, j=r)
    for r in range(5):
        j = r
        U[0 * m + j][r] = 1  # A_0j
        for k in range(p):
            V[j * p + k][r] = 1  # B_jk
        for k in range(p):
            W[k * n + 0][r] = 1  # C_0k
    
    # For r=5..9 (i=1, j=r-5)
    for r in range(5, 10):
        j = r - 5
        U[1 * m + j][r] = 1  # A_1j
        for k in range(p):
            V[j * p + k][r] = 1  # B_jk
        for k in range(p):
            W[k * n + 1][r] = 1  # C_1k
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
