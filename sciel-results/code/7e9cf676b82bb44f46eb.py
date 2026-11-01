import time

DIMENSIONS = [2, 2, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS

    # We are constructing a rank 32 decomposition.
    # This is composed of:
    # 4 blocks of M_222 (rank 7) for columns 0-1, 2-3, 4-5, 6-7
    # 1 block of M_221 (rank 4) for column 8
    # Total rank = 4*7 + 4 = 32.

    rank = 32

    U = [[0] * rank for _ in range(n * m)] # 4 x 32
    V = [[0] * rank for _ in range(m * p)] # 18 x 32
    W = [[0] * rank for _ in range(p * n)] # 18 x 32

    col = 0

    def add_strassen(k0, k1):
        nonlocal col
        # M1
        U[0][col] = 1
        U[3][col] = 1
        V[0*p + k0][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        # M2
        U[2][col] = 1
        U[3][col] = 1
        V[0*p + k0][col] = 1
        W[k0*n + 1][col] = 1
        W[k1*n + 1][col] = -1
        col += 1
        # M3
        U[0][col] = 1
        V[0*p + k1][col] = 1
        V[1*p + k1][col] = -1
        W[k1*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        # M4
        U[3][col] = 1
        V[1*p + k0][col] = 1
        V[0*p + k0][col] = -1
        W[k0*n + 0][col] = 1
        W[k0*n + 1][col] = 1
        col += 1
        # M5
        U[0][col] = 1
        U[1][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = -1
        W[k1*n + 0][col] = 1
        col += 1
        # M6
        U[2][col] = 1
        U[0][col] = -1
        V[0*p + k0][col] = 1
        V[0*p + k1][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        # M7
        U[1][col] = 1
        U[3][col] = -1
        V[1*p + k0][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1

        U[1][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1

        U[2][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1

        U[3][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1

    # Cols 0-1
    add_strassen(0, 1)
    # Cols 2-3
    add_strassen(2, 3)
    # Cols 4-5
    add_strassen(4, 5)
    # Cols 6-7
    add_strassen(6, 7)
    # Col 8
    add_trivial(8)

    best_U, best_V, best_W = U, V, W
    best_rank = rank

    # Build the tensor as a flat list of (a,b,c) triples where T=1
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_entries.append((a, b, c))

    import random
    import math

    target_rank = 31
    search_start = time.time()
    max_search_time = 100.0  # Leave 10s margin

    # Pre-compute the tensor as numpy arrays for speed
    try:
        import numpy as np
        use_numpy = True
        tensor_a = np.array([e[0] for e in tensor_entries])
        tensor_b = np.array([e[1] for e in tensor_entries])
        tensor_c = np.array([e[2] for e in tensor_entries])
        tensor_val = np.ones(len(tensor_entries))
        N = len(tensor_entries)
    except ImportError:
        use_numpy = False
        N = len(tensor_entries)

    if use_numpy:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        def compute_residual(U_arr, V_arr, W_arr):
            """Compute sum of squared residuals for all tensor entries."""
            # For each tensor entry (a,b,c), compute sum_r U[a,r]*V[b,r]*W[c,r]
            # U[tensor_a, :] has shape (N, rank)
            # V[tensor_b, :] has shape (N, rank)
            # W[tensor_c, :] has shape (N, rank)
            prod = U_arr[tensor_a, :] * V_arr[tensor_b, :] * W_arr[tensor_c, :]
            vals = prod.sum(axis=1)
            return np.sum((vals - tensor_val) ** 2)

        def compute_gradients(U_arr, V_arr, W_arr):
            """Compute gradients of residual w.r.t. U, V, W."""
            R = U_arr.shape[1]
            # pred[i] = sum_r U[a_i, r] * V[b_i, r] * W[c_i, r]
            prod = U_arr[tensor_a, :] * V_arr[tensor_b, :] * W_arr[tensor_c, :]
            pred = prod.sum(axis=1)
            err = pred - tensor_val  # shape (N,)
            
            # d/dU[a,r] = sum_{i: a_i=a} err[i] * V[b_i, r] * W[c_i, r]
            # Vectorized: for each tensor entry i, contribution to U[a_i, :] is err[i] * V[b_i, :] * W[c_i, :]
            # U_grad[a, r] = sum_{i: a_i=a} err[i] * V[b_i, r] * W[c_i, r]
            # = (err * V_arr[tensor_b, r] * W_arr[tensor_c, r]).sum over i grouped by a_i
            
            # Use np.add.at for scatter-add
            U_grad = np.zeros_like(U_arr)
            V_grad = np.zeros_like(V_arr)
            W_grad = np.zeros_like(W_arr)
            
            # For U: shape (4, R)
            # For each i, add err[i] * V[b_i, :] * W[c_i, :] to U_grad[a_i, :]
            # Vectorized: 
            # contrib_U[i, r] = err[i] * V[b_i, r] * W[c_i, r]
            # Then scatter: U_grad[a_i, r] += contrib_U[i, r]
            
            # Build contribution arrays
            contrib_U = err[:, None] * V_arr[tensor_b, :] * W_arr[tensor_c, :]  # (N, R)
            contrib_V = err[:, None] * U_arr[tensor_a, :] * W_arr[tensor_c, :]  # (N, R)
            contrib_W = err[:, None] * U_arr[tensor_a, :] * V_arr[tensor_b, :]  # (N, R)
            
            # Scatter add
            for r in range(R):
                np.add.at(U_grad, tensor_a, contrib_U[:, r])
                np.add.at(V_grad, tensor_b, contrib_V[:, r])
                np.add.at(W_grad, tensor_c, contrib_W[:, r])
            
            return 2 * U_grad, 2 * V_grad, 2 * W_grad

        # Try dropping each column and polish
        best_residual = float('inf')
        best_solution = None

        for drop_col in range(rank):
            if time.time() - search_start > max_search_time:
                break
            
            U_test = np.delete(U_np, drop_col, axis=1)
            V_test = np.delete(V_np, drop_col, axis=1)
            W_test = np.delete(W_np, drop_col, axis=1)
            
            res = compute_residual(U_test, V_test, W_test)
            if res < best_residual:
                best_residual = res
                best_solution = (U_test.copy(), V_test.copy(), W_test.copy())
            
            if res < 1e-10:
                best_U = U_test.tolist()
                best_V = V_test.tolist()
                best_W = W_test.tolist()
                best_rank = target_rank
                return {"U": best_U, "V": best_V, "W": best_W}

        # If no exact solution found by dropping a column, try gradient-based polish
        if best_solution is not None:
            U_cur, V_cur, W_cur = best_solution
            
            # Try gradient descent with adaptive learning rate
            lr = 0.001
            prev_res = best_residual
            
            for iteration in range(2000):
                if time.time() - search_start > max_search_time:
                    break
                
                U_grad, V_grad, W_grad = compute_gradients(U_cur, V_cur, W_cur)
                
                U_cur -= lr * U_grad
                V_cur -= lr * V_grad
                W_cur -= lr * W_grad
                
                if iteration % 50 == 0:
                    res = compute_residual(U_cur, V_cur, W_cur)
                    if res < best_residual:
                        best_residual = res
                        best_solution = (U_cur.copy(), V_cur.copy(), W_cur.copy())
                        if res < 1e-10:
                            best_U = U_cur.tolist()
                            best_V = V_cur.tolist()
                            best_W = W_cur.tolist()
                            best_rank = target_rank
                            return {"U": best_U, "V": best_V, "W": best_W}
                    
                    # Adaptive learning rate
                    if res < prev_res:
                        lr = min(lr * 1.1, 0.01)
                    else:
                        lr = lr * 0.8
                    
                    prev_res = res
            
            # Final check
            if best_solution is not None and best_residual < 1e-10:
                best_U = best_solution[0].tolist()
                best_V = best_solution[1].tolist()
                best_W = best_solution[2].tolist()
                best_rank = target_rank
                return {"U": best_U, "V": best_V, "W": best_W}

    # If no rank-31 found, return the rank-32 solution
    return {"U": best_U, "V": best_V, "W": best_W}
