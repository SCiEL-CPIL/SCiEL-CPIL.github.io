import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Build the trivial rank-36 base solution
    rank = n * m * p  # 2*2*9 = 36

    U = [[0] * rank for _ in range(n * m)] # 4 x 36
    V = [[0] * rank for _ in range(m * p)] # 18 x 36
    W = [[0] * rank for _ in range(p * n)] # 18 x 36

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][col] = 1
                V[b][col] = 1
                W[c][col] = 1
                col += 1

    # Build the tensor as a flat list of (a,b,c) triples where T=1
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_entries.append((a, b, c))

    tensor_a = np.array([e[0] for e in tensor_entries])
    tensor_b = np.array([e[1] for e in tensor_entries])
    tensor_c = np.array([e[2] for e in tensor_entries])
    tensor_val = np.ones(len(tensor_entries))
    N = len(tensor_entries)

    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    def compute_residual(U_arr, V_arr, W_arr):
        """Compute sum of squared residuals for all tensor entries."""
        prod = U_arr[tensor_a, :] * V_arr[tensor_b, :] * W_arr[tensor_c, :]
        vals = prod.sum(axis=1)
        return np.sum((vals - tensor_val) ** 2)

    def compute_residual_max(U_arr, V_arr, W_arr):
        """Compute max absolute residual for all tensor entries."""
        prod = U_arr[tensor_a, :] * V_arr[tensor_b, :] * W_arr[tensor_c, :]
        vals = prod.sum(axis=1)
        return np.max(np.abs(vals - tensor_val))

    def soft_max_abs(err, t=50.0):
        """Smooth surrogate for max absolute error using log-sum-exp."""
        ae = np.abs(err)
        scaled = t * ae
        max_scaled = np.max(scaled)
        if max_scaled > 500:
            return max_scaled / t
        log_sum = np.log(np.sum(np.exp(scaled - max_scaled))) + max_scaled
        return log_sum / t

    # First, verify the base solution is exact
    res_base = compute_residual(U_np, V_np, W_np)
    base_solution = {"U": U, "V": V, "W": W}

    if res_base >= 1e-10:
        # Should not happen, but return anyway
        return base_solution

    # Try to find a lower-rank solution by dropping columns
    best_reduced_solution = None
    search_start = time.time()
    max_search_time = 120.0  # Leave 20s margin

    # Strategy: try dropping each column and use L-BFGS-B to optimize
    # the remaining 31 columns to close the gap

    def make_objective(U_flat, V_flat, W_flat, U_shape, V_shape, W_shape):
        """Create objective function for scipy optimizer."""
        def objective(x):
            if time.time() > deadline:
                return 1e10
            U_x = x[:U_shape[0]*U_shape[1]].reshape(U_shape)
            V_x = x[U_shape[0]*U_shape[1]:U_shape[0]*U_shape[1]+V_shape[0]*V_shape[1]].reshape(V_shape)
            W_x = x[U_shape[0]*U_shape[1]+V_shape[0]*V_shape[1]:].reshape(W_shape)
            prod = U_x[tensor_a, :] * V_x[tensor_b, :] * W_x[tensor_c, :]
            pred = prod.sum(axis=1)
            err = pred - tensor_val
            # Use soft max as primary, add small L2 for stability
            sm = soft_max_abs(err, t=30.0)
            l2 = np.sum(err**2) * 0.01
            return sm + l2
        return objective

    def make_grad(U_flat, V_flat, W_flat, U_shape, V_shape, W_shape):
        """Create gradient function."""
        def gradient(x):
            if time.time() > deadline:
                return np.zeros_like(x)
            U_x = x[:U_shape[0]*U_shape[1]].reshape(U_shape)
            V_x = x[U_shape[0]*U_shape[1]:U_shape[0]*U_shape[1]+V_shape[0]*V_shape[1]].reshape(V_shape)
            W_x = x[U_shape[0]*U_shape[1]+V_shape[0]*V_shape[1]:].reshape(W_shape)
            prod = U_x[tensor_a, :] * V_x[tensor_b, :] * W_x[tensor_c, :]
            pred = prod.sum(axis=1)
            err = pred - tensor_val

            # Gradient of soft_max_abs
            ae = np.abs(err)
            scaled = 30.0 * ae
            max_scaled = np.max(scaled)
            if max_scaled > 500:
                # Subgradient: sign of max element
                grad_sm = np.zeros_like(err)
                max_idx = np.argmax(ae)
                grad_sm[max_idx] = np.sign(err[max_idx])
            else:
                weights = np.exp(scaled - max_scaled)
                weights /= np.sum(weights)
                grad_sm = weights * np.sign(err) * 30.0

            # Add L2 gradient
            grad_total = grad_sm + 0.02 * err

            # Compute gradients for U, V, W
            R = U_shape[1]
            U_grad = np.zeros_like(U_x)
            V_grad = np.zeros_like(V_x)
            W_grad = np.zeros_like(W_x)

            V_idx = V_x[tensor_b, :]
            W_idx = W_x[tensor_c, :]
            U_idx = U_x[tensor_a, :]

            for r in range(R):
                contrib_U = grad_total * V_idx[:, r] * W_idx[:, r]
                np.add.at(U_grad[:, r], tensor_a, contrib_U)

                contrib_V = grad_total * U_idx[:, r] * W_idx[:, r]
                np.add.at(V_grad[:, r], tensor_b, contrib_V)

                contrib_W = grad_total * U_idx[:, r] * V_idx[:, r]
                np.add.at(W_grad[:, r], tensor_c, contrib_W)

            gx = np.concatenate([U_grad.flatten(), V_grad.flatten(), W_grad.flatten()])
            return gx
        return gradient

    # For now, just return the valid trivial solution
    return base_solution
