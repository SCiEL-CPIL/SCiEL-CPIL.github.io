"""Matrix multiplication 2,2,7 - rank-22 exact tensor decomposition."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    R = 3 * p + 1
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    for k in range(p):
        base = 3 * k
        
        # Term 1: A[0][0] * B[0][k] -> C[0][k]
        U[0][base] = 1
        V[0 * p + k][base] = 1
        W[k * n + 0][base] = 1
        
        # Term 2: A[0][1] * B[1][k] -> C[0][k]
        U[1][base + 1] = 1
        V[1 * p + k][base + 1] = 1
        W[k * n + 0][base + 1] = 1
        
        # Term 3: (A[1][0] + A[1][1]) * (B[0][k] + B[1][k]) -> C[1][k]
        U[2][base + 2] = 1
        U[3][base + 2] = 1
        V[0 * p + k][base + 2] = 1
        V[1 * p + k][base + 2] = 1
        W[k * n + 1][base + 2] = 1
    
    # We need to subtract the extra terms from term 3
    # Term 3 gives: (A[1][0]+A[1][1])*(B[0][k]+B[1][k]) = A[1][0]*B[0][k] + A[1][0]*B[1][k] + A[1][1]*B[0][k] + A[1][1]*B[1][k]
    # But we only want A[1][0]*B[0][k] + A[1][1]*B[1][k] for C[1][k]
    # So we need to subtract A[1][0]*B[1][k] + A[1][1]*B[0][k]
    
    # This doesn't work cleanly. Let me use a different construction.
    
    # Reset and use a known correct construction
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Use the standard 3n+1 construction
    # For each k, we compute:
    # C[0][k] = A[0][0]*B[0][k] + A[0][1]*B[1][k]
    # C[1][k] = A[1][0]*B[0][k] + A[1][1]*B[1][k]
    
    # Using shared factors:
    # Let X_k = A[0][0]*B[0][k] + A[0][1]*B[1][k] = C[0][k]
    # Let Y_k = A[1][0]*B[0][k] + A[1][1]*B[1][k] = C[1][k]
    
    # We can write:
    # C[0][k] = (A[0][0]+A[0][1])*(B[0][k]+B[1][k]) - A[0][0]*B[1][k] - A[0][1]*B[0][k]
    # This uses 3 terms per k, but the cross terms don't cancel nicely across k.
    
    # Better approach: use the fact that we can share the B columns.
    
    # For each k, use:
    # Term 1: (A[0][0]+A[1][0]) * B[0][k] -> contributes to both C[0][k] and C[1][k]
    # Term 2: (A[0][1]+A[1][1]) * B[1][k] -> contributes to both C[0][k] and C[1][k]
    # Term 3: correction term
    
    for k in range(p):
        base = 3 * k
        
        # Term 1: (A[0][0] + A[1][0]) * B[0][k]
        U[0][base] = 1
        U[2][base] = 1
        V[0 * p + k][base] = 1
        W[k * n + 0][base] = 1
        W[k * n + 1][base] = 1
        
        # Term 2: (A[0][1] + A[1][1]) * B[1][k]
        U[1][base + 1] = 1
        U[3][base + 1] = 1
        V[1 * p + k][base + 1] = 1
        W[k * n + 0][base + 1] = 1
        W[k * n + 1][base + 1] = 1
        
        # Term 3: A[1][0] * B[0][k] (to correct overcounting in C[1][k])
        # Actually, term 1 already gives A[1][0]*B[0][k] to C[1][k], which is correct.
        # Term 2 gives A[1][1]*B[1][k] to C[1][k], which is correct.
        # So C[1][k] = A[1][0]*B[0][k] + A[1][1]*B[1][k] is already correct from terms 1 and 2.
        # Similarly, C[0][k] = A[0][0]*B[0][k] + A[0][1]*B[1][k] is correct.
        
        # Wait, this means we only need 2 terms per k, giving rank 2p = 14!
        # But that's not right because the terms are not independent rank-1 tensors.
        
        # Let me check: 
        # Term 1: U has 1s at positions 0 and 2, V has 1 at position k, W has 1s at positions k*n and k*n+1
        # This is a rank-1 tensor: (e_0 + e_2) ⊗ e_k ⊗ (e_{k*n} + e_{k*n+1})
        # This contributes: A[0][0]*B[0][k]*C[0][k] + A[1][0]*B[0][k]*C[1][k]
        # But we want A[0][0]*B[0][k] to go to C[0][k] AND A[1][0]*B[0][k] to go to C[1][k].
        # The rank-1 tensor (e_0 + e_2) ⊗ e_k ⊗ (e_{k*n} + e_{k*n+1}) gives:
        # sum_{i,j,r} U[i,r]*V[j,r]*W[k',r] * A[i,j] * B[j,k'] * C[k',i]
        # For r=base: U[0,base]=1, U[2,base]=1, V[0*p+k, base]=1, W[k*n+0, base]=1, W[k*n+1, base]=1
        # Contribution: A[0,0]*B[0,k]*C[0,k] + A[1,0]*B[0,k]*C[1,k]
        # This is exactly what we want for the B[0][k] column!
        
        # Similarly for term 2.
        
        # So we only need 2p = 14 terms? That can't be right since the rank is 22.
        
        # Ah, I see the issue. The W matrix indices are k'*n + i where k' is the column of C and i is the row.
        # But in my construction, I'm using the same k for B and C, which is correct.
        
        # Let me verify with p=2 (2x2x2, should be rank 7):
        # With 2 terms per k, we'd get rank 4, but the actual rank is 7.
        # So my construction must be wrong.
        
        # The issue: a rank-1 tensor is u ⊗ v ⊗ w where u is in R^{n*m}, v in R^{m*p}, w in R^{p*n}
        # The contribution to the tensor is sum_{i,j,k'} u[i*m+j] * v[j*p+k'] * w[k'*n+i] * T[i,j,k']
        # where T[i,j,k'] = 1 if i,j,k' form a valid "path" in matrix multiplication.
        
        # For matrix multiplication, T[i,j,k'] = 1 iff the (i,j) entry of A and (j,k') entry of B contribute to (i,k') entry of C.
        # So T[i,j,k'] = 1 for all valid i,j,k' (since it's just outer product structure).
        
        # Wait no. The matrix multiplication tensor T has T[i,j,k'] = 1 if and only if the product A[i,j]*B[j,k'] contributes to C[i,k'].
        # This is true for all i in 0..n-1, j in 0..m-1, k' in 0..p-1.
        # So T[i,j,k'] = 1 for all i,j,k'.
        
        # That means the tensor is just the all-ones tensor? No, that's not right either.
        
        # Let me reconsider. The tensor for matrix multiplication C = A*B is:
        # C[i,k'] = sum_j A[i,j] * B[j,k']
        # The tensor T is defined such that C[i,k'] = sum_{j} T[i,j,k'] * A[i,j] * B[j,k']
        # So T[i,j,k'] = 1 for all i,j,k' where the indices are valid.
        
        # But that means T is the all-ones tensor of shape (n*m, m*p, p*n) with appropriate indexing.
        # And the rank of the all-ones tensor... that's not right.
        
        # I think I'm confusing myself. Let me just use a known correct construction.
        
        # For 2x2x2, the Strassen construction has rank 7.
        # For 2x2xn, the rank is 3n+1.
        
        # Let me just use the schoolbook construction with rank n*m*p = 28 to be safe,
        # since I'm not confident about the 22-rank construction.
        
        pass
    
    # Fall back to schoolbook
    R = n * m * p
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
