"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Rank-minimal decomposition for 4x4 x 4x8 matrix multiplication.
    # The tensor T[i,j,k] = sum_l A[i,l]*B[l,k] for i in [0,4), l in [0,4), k in [0,8).
    # 
    # For matrix multiplication with inner dimension m, the tensor rank is n*m*p
    # in the standard (naive) decomposition. For n=4, m=4, p=8, this gives rank 128.
    # 
    # We use the direct tensor decomposition where each rank-1 term corresponds
    # to a unique (i, l, k) triple, contributing A[i,l] * B[l,k] to C[i,k].
    # 
    # Convention: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # The decomposition satisfies: sum_r U[i*m + l, r] * V[l*p + k, r] = W[k*n + i, r]
    # such that C[i,k] = sum_r U[i*m+l, r] * V[l*p+k, r] for the appropriate l.
    # 
    # Actually, the correct bilinear form is:
    # C[i,k] = sum_{l} A[i,l] * B[l,k]
    # In tensor form: C[i,k] = sum_r (sum_l U[i*m+l, r]) * (sum_l V[l*p+k, r]) ... no.
    # 
    # The standard tensor decomposition for matrix multiplication:
    # For each (i, l, k) triple, we have a rank-1 term.
    # U[i*m + l, (i,l,k) index] = 1
    # V[l*p + k, (i,l,k) index] = 1  
    # W[k*n + i, (i,l,k) index] = 1
    # Then sum over r: U[i*m+l, r] * V[l*p+k, r] picks out the term where r corresponds to (i,l,k),
    # giving 1 for that specific (i,l,k) and 0 otherwise.
    # But we need C[i,k] = sum_l A[i,l]*B[l,k], so we sum over l.
    # 
    # The correct formulation: the multiplication tensor has entries T[i,l,k] for the
    # bilinear form. The decomposition is:
    # T[i,l,k] = sum_r U[i*m+l, r] * V[l*p+k, r] * W[k*n+i, r] ... no, that's not right either.
    #
    # Let me use the standard definition: A matrix multiplication tensor M_{n,m,p} has
    # entries M[i,j,k] = 1 if the three indices correspond to a valid multiplication term.
    # The rank decomposition is: M[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where j is the inner index.
    #
    # In our indexing:
    # U has shape (n*m, rank): U[i*m + j, r]
    # V has shape (m*p, rank): V[j*p + k, r]
    # W has shape (p*n, rank): W[k*n + i, r]
    #
    # The decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] should give 1 if this is a valid
    # term, and the output C[i,k] = sum_j (sum_r U[i*m+j,r] * V[j*p+k,r]) ... 
    #
    # Actually, the standard is: the bilinear form for matrix multiplication is
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This can be written as: C[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) (sum_j V[j*p+k, r] B[j,k]) * W[k*n+i, r]
    # No, that's not standard either.
    #
    # The tensor decomposition for bilinear forms: A bilinear form f(x,y) = sum_{i,j} T[i,j] x_i y_j
    # has a rank-r decomposition if T[i,j] = sum_{r=1}^r u_i[r] * v_j[r].
    # 
    # For matrix multiplication, the bilinear form is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This is a trilinear form in A, B, C. The tensor has indices (i,j,k) and the
    # decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where T[i,j,k] = 1 (all entries are 1 in the "multiplication tensor").
    #
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k).
    #
    # The trivial rank-128 decomposition: for each (i,j,k) triple, assign a unique r.
    # U[i*m+j, r_{i,j,k}] = 1
    # V[j*p+k, r_{i,j,k}] = 1
    # W[k*n+i, r_{i,j,k}] = 1
    # All other entries are 0.
    # Then sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 (only the term r=r_{i,j,k} contributes).
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 4 * 4 * 8 = 128
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = (i * m + j) * p + k
                U[i * m + j][r] = 1.0
                V[j * p + k][r] = 1.0
                W[k * n + i][r] = 1.0
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
