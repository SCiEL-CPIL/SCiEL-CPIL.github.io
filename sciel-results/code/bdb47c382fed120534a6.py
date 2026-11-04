"""
Rank-minimal tensor decomposition for 4x5 @ 5x6 matrix multiplication.

The multiplication tensor T_{i,j,k} = 1 for all i in [0,4), j in [0,5), k in [0,6).
We seek U (20 x R), V (30 x R), W (24 x R) with integer/half-integer entries such that
sum_r U[i*M+j, r] * V[j*P+k, r] * W[k*N+i, r] = 1 for all (i,j,k).

Strategy:
1. Use a fast local search with incremental error updates.
2. Initialize with schoolbook or structured "subgroup" patterns.
3. Optimize to find a rank < 120 decomposition.
"""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin

    n, m, p = DIMENSIONS
    N, M, P = n, m, p
    target_rank = N * M * P  # 120

    def to_list(arr):
        return arr.tolist()

    def schoolbook():
        U = np.zeros((N * M, target_rank), dtype=np.float64)
        V = np.zeros((M * P, target_rank), dtype=np.float64)
        W = np.zeros((P * N, target_rank), dtype=np.float64)
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j, col] = 1.0
                    V[j * P + k, col] = 1.0
                    W[k * N + i, col] = 1.0
                    col += 1
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

    def compute_initial_tensor(U, V, W, R):
        """Compute the tensor values T[i,j,k] = sum_r U[iM+j,r] V[jP+k,r] W[kN+i,r]"""
        T = np.zeros((N, M, P), dtype=np.float64)
        for r in range(R):
            # u_r is vector of length N*M
            # v_r is vector of length M*P
            # w_r is vector of length P*N
            u_r = U[:, r]
            v_r = V[:, r]
            w_r = W[:, r]
            
            # Reshape to matrices
            # U[i, j] = u_r[i*M + j]
            # V[j, k] = v_r[j*P + k]
            # W[k, i] = w_r[k*N + i]
            
            U_mat = u_r.reshape(N, M)
            V_mat = v_r.reshape(M, P)
            W_mat = w_r.reshape(P, N)
            
            # T[i, j, k] += U_mat[i, j] * V_mat[j, k] * W_mat[k, i]
            # This is an outer product sum.
            # We can do this efficiently.
            for i in range(N):
                for j in range(M):
                    # u_val = U_mat[i, j]
                    # If u_val is 0, skip
                    if U_mat[i, j] != 0:
                        # T[i, j, :] += U_mat[i, j] * (V_mat[j, :] * W_mat[:, i])
                        # V_mat[j, :] is length P
                        # W_mat[:, i] is length P
                        # Element-wise product, then add to T[i, j, :]
                        T[i, j, :] += U_mat[i, j] * (V_mat[j, :] * W_mat[:, i])
        return T

    def local_search_fast(U, V, W, R, max_steps=100000):
        """
        Fast local search with incremental tensor updates.
        """
        T = compute_initial_tensor(U, V, W, R)
        
        # Count errors
        err_mask = (np.abs(T - 1.0) > 1e-9)
        err_count = np.sum(err_mask)
        
        if err_count == 0:
            return U, V, W, 0

        # Precompute indices for fast access?
        # We will update T incrementally.
        
        step = 0
        no_improve_steps = 0
        max_no_improve = 5000
        
        while step < max_steps and time.time() < deadline - 2:
            improved = False
            
            # Choose which matrix to update
            # Probabilities proportional to size? Or just random.
            mat_idx = random.choice([0, 1, 2])
            
            if mat_idx == 0: # Update U
                i = random.randrange(N)
                j = random.randrange(M)
                idx_u = i * M + j
                r = random.randrange(R)
                
                old_val = U[idx_u, r]
                # Try new values: 0, 1, -1, 0.5, -0.5
                # Only consider values that are "close" to integer/half-integer
                # Since we want integer/half-integer in the end, we can search in that space directly.
                candidates = [0.0, 0.5, -0.5, 1.0, -1.0]
                # Remove current value
                if old_val in candidates:
                    candidates.remove(old_val)
                
                if not candidates:
                    continue
                    
                new_val = random.choice(candidates)
                delta = new_val - old_val
                
                # Update T
                # T[i, j, k] += delta * V[j*P+k, r] * W[k*N+i, r]
                # V[j*P+k, r] for k in 0..P-1 is V[j*P : j*P+P, r]
                # W[k*N+i, r] for k in 0..P-1 is W[i::N, r] (indices i, i+N, ..., i+(P-1)N)
                
                v_part = V[j*P : j*P+P, r]
                w_part = W[i::N, r] # Length P
                
                # Update T[i, j, :]
                T[i, j, :] += delta * (v_part * w_part)
                
                # Update error count
                # Only T[i, j, :] changed
                old_err = err_mask[i, j, :]
                new_vals = T[i, j, :]
                new_err = (np.abs(new_vals - 1.0) > 1e-9)
                
                old_count = np.sum(old_err)
                new_count = np.sum(new_err)
                
                if new_count < old_count:
                    U[idx_u, r] = new_val
                    err_count += (new_count - old_count)
                    err_mask[i, j, :] = new_err
                    improved = True
                    no_improve_steps = 0
                else:
                    # Revert T
                    T[i, j, :] -= delta * (v_part * w_part)
                    # err_mask remains same
                    
            elif mat_idx == 1: # Update V
                j = random.randrange(M)
                k = random.randrange(P)
                idx_v = j * P + k
                r = random.randrange(R)
                
                old_val = V[idx_v, r]
                candidates = [0.0, 0.5, -0.5, 1.0, -1.0]
                if old_val in candidates:
                    candidates.remove(old_val)
                if not candidates:
                    continue
                new_val = random.choice(candidates)
                delta = new_val - old_val
                
                # T[i, j, k] += U[i*M+j, r] * delta * W[k*N+i, r]
                # U[i*M+j, r] for i in 0..N-1 is U[j::M, r] (indices j, j+M, ..., j+(N-1)M)
                # W[k*N+i, r] for i in 0..N-1 is W[k*N : k*N+N, r]
                
                u_part = U[j::M, r] # Length N
                w_part = W[k*N : k*N+N, r] # Length N
                
                # Update T[:, j, k]
                T[:, j, k] += (u_part * w_part) * delta
                
                old_err = err_mask[:, j, k]
                new_vals = T[:, j, k]
                new_err = (np.abs(new_vals - 1.0) > 1e-9)
                
                old_count = np.sum(old_err)
                new_count = np.sum(new_err)
                
                if new_count < old_count:
                    V[idx_v, r] = new_val
                    err_count += (new_count - old_count)
                    err_mask[:, j, k] = new_err
                    improved = True
                    no_improve_steps = 0
                else:
                    T[:, j, k] -= (u_part * w_part) * delta
                    
            else: # Update W
                k = random.randrange(P)
                i = random.randrange(N)
                idx_w = k * N + i
                r = random.randrange(R)
                
                old_val = W[idx_w, r]
                candidates = [0.0, 0.5, -0.5, 1.0, -1.0]
                if old_val in candidates:
                    candidates.remove(old_val)
                if not candidates:
                    continue
                new_val = random.choice(candidates)
                delta = new_val - old_val
                
                # T[i, j, k] += U[i*M+j, r] * V[j*P+k, r] * delta
                # U[i*M+j, r] for j in 0..M-1 is U[i*M : i*M+M, r]
                # V[j*P+k, r] for j in 0..M-1 is V[k::P, r] (indices k, k+P, ..., k+(M-1)P)
                
                u_part = U[i*M : i*M+M, r] # Length M
                v_part = V[k::P, r] # Length M
                
                # Update T[i, :, k]
                T[i, :, k] += (u_part * v_part) * delta
                
                old_err = err_mask[i, :, k]
                new_vals = T[i, :, k]
                new_err = (np.abs(new_vals - 1.0) > 1e-9)
                
                old_count = np.sum(old_err)
                new_count = np.sum(new_err)
                
                if new_count < old_count:
                    W[idx_w, r] = new_val
                    err_count += (new_count - old_count)
                    err_mask[i, :, k] = new_err
                    improved = True
                    no_improve_steps = 0
                else:
                    T[i, :, k] -= (u_part * v_part) * delta
                    
            if err_count == 0:
                return U, V, W, 0
                
            if improved:
                step += 1
            else:
                no_improve_steps += 1
                if no_improve_steps > max_no_improve:
                    break
                
        return U, V, W, err_count

    best_result = None

    # Try ranks from 119 down to 90
    for R in range(target_rank - 1, max(target_rank - 15, 80), -1):
        if time.time() > deadline - 10:
            break

        for restart in range(5):
            if time.time() > deadline - 5:
                break

            # Initialize with schoolbook-like structure
            U = np.zeros((N * M, R), dtype=np.float64)
            V = np.zeros((M * P, R), dtype=np.float64)
            W = np.zeros((P * N, R), dtype=np.float64)

            # Fill first R columns with schoolbook pattern
            col = 0
            for i in range(N):
                for j in range(M):
                    for k in range(P):
                        if col >= R:
                            break
                        U[i * M + j, col] = 1.0
                        V[j * P + k, col] = 1.0
                        W[k * N + i, col] = 1.0
                        col += 1
                    if col >= R:
                        break
                if col >= R:
                    break

            # Add random noise to break symmetry
            for _ in range(R * 10):
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    idx = random.randrange(N * M)
                    r = random.randrange(R)
                    U[idx, r] = random.choice([0.0, 0.5, -0.5, 1.0, -1.0])
                elif mat == 'V':
                    idx = random.randrange(M * P)
                    r = random.randrange(R)
                    V[idx, r] = random.choice([0.0, 0.5, -0.5, 1.0, -1.0])
                else:
                    idx = random.randrange(P * N)
                    r = random.randrange(R)
                    W[idx, r] = random.choice([0.0, 0.5, -0.5, 1.0, -1.0])

            U, V, W, err_count = local_search_fast(U, V, W, R, max_steps=20000)

            if err_count == 0:
                return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

            if best_result is None or R < best_result[0]:
                best_result = (R, U, V, W, err_count)

    # If no improvement found, return schoolbook
    return schoolbook()

# EVOLVE-BLOCK-END
