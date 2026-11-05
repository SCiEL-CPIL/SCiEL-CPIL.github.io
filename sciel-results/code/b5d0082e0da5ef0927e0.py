import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    def verify_decomp(U, V, W):
        """Check if sum_r U[:,r] V[:,r] W[:,r] == T"""
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=1e-9)

    def get_schoolbook():
        """Schoolbook decomposition with rank n*m*p"""
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
        return U, V, W

    # Build a rank-50 decomposition using Strassen for 2x2 blocks
    # A is 2x4, B is 4x7, C is 2x7
    # Split B columns into (0,1), (2,3), (4,5), (6)
    # Split A columns into (0,1), (2,3)
    # Use Strassen for 2x2 A cols x 2x2 B cols blocks (6 blocks)
    # Use direct for remaining column 6
    
    max_rank = 100
    U = np.zeros((n * m, max_rank))
    V = np.zeros((m * p, max_rank))
    W = np.zeros((p * n, max_rank))
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        """Add a Strassen 2x2x2 block"""
        nonlocal col
        # M1: (a+d)(e+h)
        U[i0 * m + j0, col] += 1
        U[(i0 + 1) * m + (j0 + 1), col] += 1
        V[j0 * p + k0, col] += 1
        V[(j0 + 1) * p + (k0 + 1), col] += 1
        W[k0 * n + i0, col] += 1
        W[(k0 + 1) * n + (i0 + 1), col] += 1
        col += 1
        
        # M2: (c+d)e
        U[(i0 + 1) * m + j0, col] += 1
        U[(i0 + 1) * m + (j0 + 1), col] += 1
        V[j0 * p + k0, col] += 1
        W[k0 * n + (i0 + 1), col] += 1
        W[(k0 + 1) * n + (i0 + 1), col] -= 1
        col += 1
        
        # M3: a(f-h)
        U[i0 * m + j0, col] += 1
        V[j0 * p + (k0 + 1), col] += 1
        V[(j0 + 1) * p + (k0 + 1), col] -= 1
        W[(k0 + 1) * n + i0, col] += 1
        W[(k0 + 1) * n + (i0 + 1), col] += 1
        col += 1
        
        # M4: d(g-e)
        U[(i0 + 1) * m + (j0 + 1), col] += 1
        V[(j0 + 1) * p + k0, col] += 1
        V[j0 * p + k0, col] -= 1
        W[k0 * n + i0, col] += 1
        W[k0 * n + (i0 + 1), col] += 1
        col += 1
        
        # M5: (a+b)h
        U[i0 * m + j0, col] += 1
        U[i0 * m + (j0 + 1), col] += 1
        V[(j0 + 1) * p + (k0 + 1), col] += 1
        W[k0 * n + i0, col] -= 1
        W[(k0 + 1) * n + i0, col] += 1
        col += 1
        
        # M6: (c-a)(e+f)
        U[(i0 + 1) * m + j0, col] += 1
        U[i0 * m + j0, col] -= 1
        V[j0 * p + k0, col] += 1
        V[j0 * p + (k0 + 1), col] += 1
        W[(k0 + 1) * n + (i0 + 1), col] += 1
        col += 1
        
        # M7: (b-d)(g+h)
        U[i0 * m + (j0 + 1), col] += 1
        U[(i0 + 1) * m + (j0 + 1), col] -= 1
        V[(j0 + 1) * p + k0, col] += 1
        V[(j0 + 1) * p + (k0 + 1), col] += 1
        W[k0 * n + i0, col] += 1
        col += 1

    # Add Strassen blocks
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            add_strassen_block(0, j0, k0)
            
    # Handle column 6
    for j in range(m):
        for i in range(n):
            U[i * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + i, col] = 1
            col += 1

    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]

    # Attempt rank reduction via sparse linear algebra
    if time.time() < deadline - 10:
        target_rank = col - 1
        while target_rank >= 40 and time.time() < deadline - 5:
            # Try to find a valid decomposition with target_rank
            # Use a greedy approach: try removing one column at a time
            success = False
            for r in range(col):
                if time.time() >= deadline - 5:
                    break
                # Try removing column r
                U_test = np.delete(U, r, axis=1)
                V_test = np.delete(V, r, axis=1)
                W_test = np.delete(W, r, axis=1)
                
                if verify_decomp(U_test, V_test, W_test):
                    U, V, W = U_test, V_test, W_test
                    col = target_rank
                    success = True
                    break
            
            if not success:
                break
            target_rank -= 1

    if not verify_decomp(U, V, W):
        U, V, W = get_schoolbook()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
