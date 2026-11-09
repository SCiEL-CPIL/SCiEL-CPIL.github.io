import numpy as np
import time
import random

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Build target tensor: T[i*m+j, j*p+k, k*n+i] = 1
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook rank = n*m*p = 96
    rank_schoolbook = n * m * p

    # Initialize with schoolbook
    U_init = np.zeros((n*m, rank_schoolbook))
    V_init = np.zeros((m*p, rank_schoolbook))
    W_init = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_init[i*m+j, col] = 1
                V_init[j*p+k, col] = 1
                W_init[k*n+i, col] = 1
                col += 1

    best_U, best_V, best_W = U_init.copy(), V_init.copy(), W_init.copy()
    best_rank = rank_schoolbook

    def verify(U, V, W):
        """Check if decomposition is exact."""
        recon = np.zeros((n*m, m*p, p*n))
        for r in range(U.shape[1]):
            recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(target, recon, atol=1e-8)

    def snap_half_int(arr):
        return np.round(arr * 2) / 2.0

    def alt_least_squares(U, V, W, target_rank, max_iters=100):
        """Alternating least squares to refine factor matrices."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        # Precompute flattened target
        target_flat = target.flatten()

        for it in range(max_iters):
            if time.time() > deadline - 10:
                break

            # Update U
            # M_VW shape: (m*p*p*n, target_rank)
            # M_VW[j*p*n + k, r] = V[j, r] * W[k, r]
            M_VW = np.einsum('jr,kr->jkr', V, W).reshape(m*p*p*n, target_rank)
            for i in range(n*m):
                if time.time() > deadline - 10:
                    break
                # Target for U[i, :] is T[i, :, :] flattened
                target_i = target[i, :, :].flatten()
                try:
                    sol = np.linalg.lstsq(M_VW, target_i, rcond=None)[0]
                    U[i, :] = sol
                except:
                    pass

            # Update V
            # M_UW shape: (n*m*p*n, target_rank)
            # M_UW[i*p*n + k, r] = U[i, r] * W[k, r]
            M_UW = np.einsum('ir,kr->ikr', U, W).reshape(n*m*p*n, target_rank)
            for j in range(m*p):
                if time.time() > deadline - 10:
                    break
                # Target for V[j, :]
                # T[i*m+j, j*p+k, k*n+i] = 1
                # We need to find V[j, :] such that sum_r V[j,r] * (U[i,r] * W[k,r]) matches T
                # For a fixed j, the non-zero entries of T are where i%m == j//p and k%n == i//m
                # Let's just build the target vector explicitly for this j
                target_j = np.zeros(n*m*p*n)
                j1 = j // p
                for i in range(n*m):
                    i0 = i // m
                    j0 = i % m
                    if j0 == j1:
                        for k in range(p*n):
                            k2 = k // n
                            i2 = k % n
                            if i0 == i2:
                                target_j[i*p*n+k] = 1.0
                try:
                    sol = np.linalg.lstsq(M_UW, target_j, rcond=None)[0]
                    V[j, :] = sol
                except:
                    pass

            # Update W
            # M_UV shape: (n*m*m*p, target_rank)
            # M_UV[i*m*p + j, r] = U[i, r] * V[j, r]
            M_UV = np.einsum('ir,jr->ijr', U, V).reshape(n*m*m*p, target_rank)
            for k in range(p*n):
                if time.time() > deadline - 10:
                    break
                # Target for W[k, :]
                target_k = np.zeros(n*m*m*p)
                k2 = k // n
                for i in range(n*m):
                    i0 = i // m
                    j0 = i % m
                    if i0 == k2:
                        for j in range(m*p):
                            j1 = j // p
                            k1 = j % p
                            if j0 == j1:
                                target_k[i*m*p+j] = 1.0
                try:
                    sol = np.linalg.lstsq(M_UV, target_k, rcond=None)[0]
                    W[k, :] = sol
                except:
                    pass
            
            # Optional: Snap to half-integers every 5 iterations to guide towards rational solutions
            if it % 5 == 4:
                U = snap_half_int(U)
                V = snap_half_int(V)
                W = snap_half_int(W)

        return U, V, W

    # Try to find lower rank decompositions
    random.seed(42)

    # Try ranks from 90 down to 48 (known Strassen-style bound)
    for target_rank in range(90, 47, -2):
        if time.time() > deadline - 15:
            break

        # Multiple random restarts
        for restart in range(10):
            if time.time() > deadline - 15:
                break

            # Random initialization with small values
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5

            # Run ALS
            U, V, W = alt_least_squares(U, V, W, target_rank, max_iters=30)

            # Final snap
            U_s = snap_half_int(U)
            V_s = snap_half_int(V)
            W_s = snap_half_int(W)

            # Check accuracy
            recon = np.zeros((n*m, m*p, p*n))
            for r in range(target_rank):
                recon += np.einsum('a,b,c->abc', U_s[:, r], V_s[:, r], W_s[:, r])

            if np.allclose(target, recon, atol=1e-8):
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_s, V_s, W_s
                    break
            else:
                # Check un-snapped
                recon = np.zeros((n*m, m*p, p*n))
                for r in range(target_rank):
                    recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                if np.allclose(target, recon, atol=1e-8):
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_U, best_V, best_W = U, V, W
                        break

    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]

    return {"U": U_list, "V": V_list, "W": W_list}
