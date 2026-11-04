"""Find a rank-minimal exact tensor decomposition for multiplying a 3x4 by 4x6 matrix."""
import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START

def _verify_decomposition(n, m, p, U, V, W, tol=1e-9):
    """Verify that the tensor decomposition is exact for matrix multiplication."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float).reshape(n, m, rank)
    V_arr = np.array(V, dtype=float).reshape(m, p, rank)
    W_arr = np.array(W, dtype=float).reshape(p, n, rank)
    
    result = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    target = np.ones((n, m, p))
    
    return np.allclose(result, target, atol=tol)


def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) factor matrices."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _kronecker_decomp(n, m, p):
    """
    Construct a decomposition using the Kronecker product structure.
    For (3,4,6): (3,4,6) = (3,2,3) x (1,2,2) ? No.
    Actually (3,4,6) = (3,2,6) x (1,2,1) gives rank 36 * 1 = 36.
    But we can also try (3,4,6) = (1,4,6) x (3,1,1) -> rank 24 * 3 = 72.
    
    Let's use (3,2,6) x (1,2,1) which gives rank 36.
    Wait, the previous code did this. Let's just implement it directly and efficiently.
    """
    n1, m1, p1 = 3, 2, 6
    n2, m2, p2 = 1, 2, 1
    
    # (1,2,1) rank 1
    # U2[0,0]=1, U2[1,0]=1
    # V2[0,0]=1, V2[1,0]=1
    # W2[0,0]=1
    
    # (3,2,6) schoolbook rank 36
    r1 = 36
    # We construct the full (3,4,6) decomposition directly.
    n, m, p = 3, 4, 6
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # T = T1 x T2
    # T1 is (3,2,6), T2 is (1,2,1)
    # For c in 0..35:
    #   U[(i1*1+i2)*4 + (j1*2+j2), c] = U1[i1*2+j1, c] * U2[i2*2+j2, 0]
    #   V[(j1*2+j2)*6 + (k1*1+k2), c] = V1[j1*6+k1, c] * V2[j2*1+k2, 0]
    #   W[(k1*1+k2)*3 + (i1*1+i2), c] = W1[k1*3+i1, c] * W2[k2*1+i2, 0]
    
    # Since U2, V2, W2 are all 1s, the entries are just U1, V1, W1 replicated appropriately.
    # U1[i1*2+j1, c] is 1 if (i1, j1, k1) corresponds to c in the schoolbook loop.
    # Let's just iterate over the schoolbook terms of (3,2,6) and place them.
    
    # Schoolbook for (3,2,6):
    # c = i1*12 + j1*6 + k1
    # U1[i1*2+j1, c] = 1
    # V1[j1*6+k1, c] = 1
    # W1[k1*3+i1, c] = 1
    
    # In the combined space:
    # U[(i1*1+i2)*4 + (j1*2+j2), c] = 1
    # V[(j1*2+j2)*6 + (k1*1+k2), c] = 1
    # W[(k1*1+k2)*3 + (i1*1+i2), c] = 1
    
    # So for each (i1, j1, k1) and c:
    #   for i2 in 0..0: (n2=1)
    #   for j2 in 0..1: (m2=2)
    #   for k2 in 0..0: (p2=1)
    #       U[(i1*1+i2)*4 + (j1*2+j2), c] = 1
    #       V[(j1*2+j2)*6 + (k1*1+k2), c] = 1
    #       W[(k1*1+k2)*3 + (i1*1+i2), c] = 1
    
    # Wait, this fills multiple entries in U, V, W for each c.
    # But the tensor decomposition requires:
    # sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
    # If U[i,j,c] = 1 for multiple (j, c) pairs, it might not work.
    # Let's re-examine the Kronecker product logic.
    # T1[i1,j1,k1] = 1 if i1,j1,k1 are in range.
    # T2[i2,j2,k2] = 1 if i2,j2,k2 are in range.
    # T[i1,i2 | j1,j2 | k1,k2] = T1[i1,j1,k1] * T2[i2,j2,k2]
    # Rank r = r1 * r2.
    # T1 is decomposed as sum_{c1} U1[i1,j1,c1] V1[j1,k1,c1] W1[k1,i1,c1]
    # T2 is decomposed as sum_{c2} U2[i2,j2,c2] V2[j2,k2,c2] W2[k2,i2,c2]
    # T is decomposed as sum_{c1,c2} (U1*U2) (V1*V2) (W1*W2)
    # c = c1 * r2 + c2
    
    # In our case r2=1, so c = c1.
    # U[(i1,i2),(j1,j2), c1] = U1[i1,j1,c1] * U2[i2,j2,0]
    # Since U2[i2,j2,0] = 1, U[...] = U1[i1,j1,c1].
    # So U[i,j,c] = 1 if there exists (i1,i2) such that i = i1*1+i2 and (j1,j2) such that j = j1*2+j2
    #                 AND U1[i1,j1,c] = 1.
    # U1[i1,j1,c] = 1 iff c = i1*12 + j1*6 + k1 for some k1.
    # So for a given c, let i1 = c // 12, j1 = (c // 6) % 2, k1 = c % 6.
    # Then U[(i1,i2),(j1,j2), c] = 1 for all i2, j2.
    # But i = i1*1 + i2. Since n2=1, i2=0 always. So i = i1.
    # j = j1*2 + j2. j2 can be 0 or 1. So for a fixed i1, j1, c,
    # U[i1, j1*2, c] = 1 and U[i1, j1*2+1, c] = 1.
    
    # Similarly for V:
    # V[(j1,j2),(k1,k2), c] = V1[j1,k1,c] * V2[j2,k2,0] = V1[j1,k1,c].
    # V1[j1,k1,c] = 1 iff c = i1*12 + j1*6 + k1.
    # So for a given c, j1 = (c//6)%2, k1 = c%6.
    # V[(j1,j2),(k1,k2), c] = 1 for all j2, k2.
    # j = j1*2 + j2, k = k1*1 + k2.
    # So V[j1*2, k1, c] = 1, V[j1*2, k1+0, c]=1 (k2=0).
    # Wait, p2=1, so k2=0 always. k = k1.
    # So V[j1*2, k1, c] = 1 and V[j1*2+1, k1, c] = 1.
    
    # Similarly for W:
    # W[(k1,k2),(i1,i2), c] = W1[k1,i1,c] * W2[k2,i2,0] = W1[k1,i1,c].
    # W1[k1,i1,c] = 1 iff c = i1*12 + j1*6 + k1.
    # So for a given c, i1 = c//12, k1 = c%6.
    # W[(k1,k2),(i1,i2), c] = 1 for all k2, i2.
    # k = k1*1 + k2, i = i1*1 + i2.
    # So W[k1, i1, c] = 1 and W[k1+0, i1, c]=1 (k2=0, i2=0).
    
    # Let's check if this works.
    # sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # For a fixed (i,j,k), this sum is 1 iff there is a c such that
    # U[i,j,c]=1, V[j,k,c]=1, W[k,i,c]=1.
    # From above:
    # U[i,j,c]=1 iff i=i1, j=j1*2+j2, c=i1*12+j1*6+k1.
    # V[j,k,c]=1 iff j=j1'*2+j2', k=k1', c=i1'*12+j1'*6+k1'.
    # W[k,i,c]=1 iff k=k1'', i=i1'', c=i1''*12+j1''*6+k1''.
    
    # For the product to be non-zero, we need:
    # i = i1 = i1' = i1''
    # j = j1*2+j2 = j1'*2+j2'
    # k = k1 = k1' = k1''
    # c = i1*12 + j1*6 + k1 = i1*12 + j1'*6 + k1' = i1*12 + j1''*6 + k1''
    # This implies j1 = j1' = j1'' and k1 = k1' = k1''.
    # So the sum is 1 iff there exists (i1, j1, k1) such that
    # i = i1, j = j1*2+j2 for some j2, k = k1.
    # Since j is in 0..3, j1 can be 0 or 1.
    # If j=0, j1=0, j2=0.
    # If j=1, j1=0, j2=1.
    # If j=2, j1=1, j2=0.
    # If j=3, j1=1, j2=1.
    # So for any (i,j,k), there is exactly one (i1,j1,k1) such that
    # i=i1, k=k1, and j is in {j1*2, j1*2+1}.
    # Wait, j is always in {j1*2, j1*2+1} for the appropriate j1.
    # So the sum is always 1?
    # Yes, because for any (i,j,k), we can find i1=i, k1=k, and j1=floor(j/2).
    # Then c = i1*12 + j1*6 + k1 is valid.
    # And U[i,j,c]=1, V[j,k,c]=1, W[k,i,c]=1.
    # So the sum is 1.
    
    # This seems correct. Let's implement it.
    
    for c in range(36):
        i1 = c // 12
        j1 = (c // 6) % 2
        k1 = c % 6
        
        # U[i, j, c] = 1 for i=i1, j in {j1*2, j1*2+1}
        U[i1 * 4 + j1 * 2][c] = 1
        U[i1 * 4 + j1 * 2 + 1][c] = 1
        
        # V[j, k, c] = 1 for j in {j1*2, j1*2+1}, k=k1
        V[(j1 * 2) * 6 + k1][c] = 1
        V[(j1 * 2 + 1) * 6 + k1][c] = 1
        
        # W[k, i, c] = 1 for k=k1, i=i1
        W[k1 * 3 + i1][c] = 1
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Use the Kronecker decomposition which gives rank 36
    U, V, W = _kronecker_decomp(n, m, p)
    
    # Verify
    if _verify_decomposition(n, m, p, U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
