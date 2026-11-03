"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is m * p = 24
    rank = m * p
    
    # U shape: (n * m, rank) -> (2 * 4, 24) = (8, 24)
    # V shape: (m * p, rank) -> (4 * 6, 24) = (24, 24)
    # W shape: (p * n, rank) -> (6 * 2, 24) = (12, 24)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            c = j * p + k
            
            # U[i * m + j, c] = 1
            for i in range(n):
                U[i * m + j][c] = 1
                
            # V[j * p + k, c] = 1
            V[j * p + k][c] = 1
            
            # W[k * n + i, c] = 1 ?
            # Wait, my derivation above:
            # W_{k,i,c} needs to handle the summation.
            # In the derivation: W_{k,i,c} = 1 if k=k_c?
            # But W index is k * n + i.
            # If we set W[k * n + i, c] = 1 for all i?
            # Let's check the sum:
            # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A[i,j]) * (sum_j' V[j'*p+k, c] B[j',k])
            # With U[i*m+j, c] = 1 if j=j_c else 0.
            # Term U sum = A[i, j_c].
            # V sum: V[j'*p+k, c] = 1 if j'=j_c AND k=k_c?
            # My V definition: V[j * p + k, c] = 1.
            # So V[j'*p+k, c] is 1 only if j'=j_c and k=k_c.
            # If k != k_c, V sum is 0.
            # If k = k_c, V sum = B[j_c, k_c].
            # So product is A[i, j_c] * B[j_c, k_c] if k=k_c, else 0.
            # We sum over c.
            # C[i,k] = sum_{c: k_c=k} W[k*n+i, c] A[i, j_c] B[j_c, k_c].
            # We want C[i,k] = sum_j A[i,j] B[j,k].
            # Since k_c=k, B[j_c, k_c] = B[j_c, k].
            # We need sum_{c: k_c=k} W[k*n+i, c] A[i, j_c] = sum_j A[i,j] B[j,k] / B[j_c,k]? No.
            # The term is A[i, j_c] B[j_c, k].
            # We want sum_{j_c} A[i, j_c] B[j_c, k].
            # So we need W[k*n+i, c] = 1 for all c with k_c=k.
            
            for i in range(n):
                W[k * n + i][c] = 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
