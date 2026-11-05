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

    def build_strassen_decomp():
        """Build the rank-50 Strassen-based decomposition"""
        max_rank = 100
        U = np.zeros((n * m, max_rank))
        V = np.zeros((m * p, max_rank))
        W = np.zeros((p * n, max_rank))
        col = 0
        
        def add_strassen_block(i0, j0, k0):
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

        # Add Strassen blocks for 2x2 A columns x 2x2 B columns
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen_block(0, j0, k0)
                
        # Handle column 6 of B (7th column)
        for j in range(m):
            for i in range(n):
                U[i * m + j, col] = 1
                V[j * p + 6, col] = 1
                W[6 * n + i, col] = 1
                col += 1

        return U[:, :col], V[:, :col], W[:, :col]

    def try_reduce_rank(U, V, W):
        """Try to find linear dependencies to reduce rank"""
        rank = U.shape[1]
        if rank <= 1:
            return U, V, W
        
        for _ in range(30):
            if time.time() > deadline - 15:
                break
            
            found_removal = False
            for r in range(rank):
                # Check if column r is redundant by removing it and checking
                U_rest = np.delete(U, r, axis=1)
                V_rest = np.delete(V, r, axis=1)
                W_rest = np.delete(W, r, axis=1)
                
                T = np.zeros((n * m, m * p, p * n))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i * m + j, j * p + k, k * n + i] = 1.0
                
                recon_reduced = np.einsum('ar,br,cr->abc', U_rest, V_rest, W_rest)
                
                if np.allclose(recon_reduced, T, atol=1e-9):
                    # Column r is redundant
                    U, V, W = U_rest, V_rest, W_rest
                    rank -= 1
                    found_removal = True
                    break
            
            if not found_removal:
                break
        
        return U, V, W

    # Build initial decomposition
    U, V, W = build_strassen_decomp()
    
    if not verify_decomp(U, V, W):
        U, V, W = get_schoolbook()
    else:
        # Try to reduce rank
        U, V, W = try_reduce_rank(U, V, W)
        
        # Verify after reduction
        if not verify_decomp(U, V, W):
            U, V, W = build_strassen_decomp()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
