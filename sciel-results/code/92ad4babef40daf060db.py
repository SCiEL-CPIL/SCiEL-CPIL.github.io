"""Matrix multiplication 2x3 x 3x4 - rank-minimal tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    U_dims = n * m  # 6
    V_dims = m * p  # 12
    W_dims = p * n  # 8

    # Build the matrix multiplication tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    T_dense = np.zeros((U_dims, V_dims, W_dims))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_dense[a, b, c] = 1.0

    def compute_error(U, V, W):
        """Compute the squared Frobenius norm of the reconstruction error."""
        T_rec = np.einsum('at,bt,ct->abc', U, V, W)
        diff = T_rec - T_dense
        return np.sum(diff * diff)

    def snap_and_verify(U, V, W, tol=1e-6):
        """Snap to half-integers and verify exactness."""
        U_snap = np.round(U * 2) / 2.0
        V_snap = np.round(V * 2) / 2.0
        W_snap = np.round(W * 2) / 2.0
        
        # Check if snapping was close
        if (np.linalg.norm(U_snap - U) > 1e-3 or 
            np.linalg.norm(V_snap - V) > 1e-3 or 
            np.linalg.norm(W_snap - W) > 1e-3):
            return None

        # Verify exact decomposition with snapped values
        T_rec = np.einsum('at,bt,ct->abc', U_snap, V_snap, W_snap)
        err = np.linalg.norm(T_rec - T_dense)
        if err < 1e-8:
            return U_snap, V_snap, W_snap
        return None

    def to_list(mat):
        """Convert numpy array to list of lists."""
        return [[float(x) for x in row] for row in mat]

    def discrete_local_search(rank, n_restarts=20, max_steps=5000, seed=0):
        """
        Discrete local search over half-integer values {-1, -0.5, 0, 0.5, 1}.
        Starts with random initialization and performs hill climbing.
        """
        rng = random.Random(seed)
        best_err = float('inf')
        best_UVW = None
        
        values = [-1.0, -0.5, 0.0, 0.5, 1.0]

        for restart in range(n_restarts):
            if time.time() > deadline:
                break

            # Random initialization
            U = np.array([[rng.choice(values) for _ in range(rank)] for _ in range(U_dims)])
            V = np.array([[rng.choice(values) for _ in range(rank)] for _ in range(V_dims)])
            W = np.array([[rng.choice(values) for _ in range(rank)] for _ in range(W_dims)])
            
            current_err = compute_error(U, V, W)
            
            # Hill Climbing
            improved = True
            steps = 0
            while improved and steps < max_steps and time.time() < deadline:
                improved = False
                steps += 1
                
                # Try to improve U
                # We iterate through all entries and all possible values
                # To speed up, we can just try the neighbors of the current value
                # But for small dimensions, trying all 5 values is okay if we are careful
                
                # Optimization: Precompute V^T W and U^T W etc?
                # T_rec = sum_t U[:,t] V[:,t] W[:,t]^T
                # Error = || T - sum ... ||^2
                # This is a quadratic function in each entry.
                
                # Let's just do a simple coordinate descent
                # For each t, for each a, try changing U[a,t]
                
                for t in range(rank):
                    if time.time() > deadline:
                        break
                    
                    # Update U column t
                    # Fix V[:,t] and W[:,t]
                    # The term involving U[:,t] is U[:,t] * (V[:,t] * W[:,t]^T)
                    # Let M = V[:,t] * W[:,t]^T (shape V_dims x W_dims)
                    # T_rec_a = U[a,t] * M + Rest
                    # Error_a = || T_a - U[a,t]M - Rest ||^2
                    # This is a quadratic in U[a,t]. The minimum is at a specific value.
                    # Since we are constrained to discrete values, we just check the 5 candidates.
                    
                    M = np.outer(V[:, t], W[:, t])
                    # Rest is the sum of all other terms
                    # This is expensive to compute every time.
                    # Alternative: Just compute full error for candidate changes.
                    # With rank ~19 and dims 6,12,8, full error is cheap.
                    
                    for a in range(U_dims):
                        old_val = U[a, t]
                        for new_val in values:
                            if new_val == old_val:
                                continue
                            U[a, t] = new_val
                            new_err = compute_error(U, V, W)
                            if new_err < current_err:
                                current_err = new_err
                                improved = True
                            else:
                                U[a, t] = old_val
                                # Optional: if we found a better value, we might want to 
                                # continue from there, but simple hill climbing reverts.
                                # Stochastic hill climbing: accept if better, else revert.
                    
                    if time.time() > deadline:
                        break

                    # Update V column t
                    for b in range(V_dims):
                        old_val = V[b, t]
                        for new_val in values:
                            if new_val == old_val:
                                continue
                            V[b, t] = new_val
                            new_err = compute_error(U, V, W)
                            if new_err < current_err:
                                current_err = new_err
                                improved = True
                            else:
                                V[b, t] = old_val

                    if time.time() > deadline:
                        break

                    # Update W column t
                    for c in range(W_dims):
                        old_val = W[c, t]
                        for new_val in values:
                            if new_val == old_val:
                                continue
                            W[c, t] = new_val
                            new_err = compute_error(U, V, W)
                            if new_err < current_err:
                                current_err = new_err
                                improved = True
                            else:
                                W[c, t] = old_val

            if current_err < best_err:
                best_err = current_err
                best_UVW = (U.copy(), V.copy(), W.copy())
            
            if best_err < 1e-8:
                break

        if best_UVW is not None:
            U, V, W = best_UVW
            result = snap_and_verify(U, V, W)
            if result:
                return {'U': to_list(result[0]), 'V': to_list(result[1]), 'W': to_list(result[2])}
        return None

    # Try ranks from the theoretical lower bound (19) up to schoolbook (24)
    # Known rank for 2x3x4 is 19.
    for r in range(19, 25):
        if time.time() > deadline - 10:
            break
        
        # Use multiple seeds for robustness
        found = None
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            found = discrete_local_search(r, n_restarts=5, max_steps=2000, seed=r*100+seed)
            if found:
                break
        
        if found:
            return found

    # Fallback: Schoolbook decomposition (Rank 24)
    rank = n * m * p
    U = [[0] * rank for _ in range(U_dims)]
    V = [[0] * rank for _ in range(V_dims)]
    W = [[0] * rank for _ in range(W_dims)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
