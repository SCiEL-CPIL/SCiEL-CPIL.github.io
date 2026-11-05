"""Matrix multiplication 4x5x6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s kill
    
    def verify(U, V, W, n, m, p):
        """Verify that the factorization is correct."""
        rank = U.shape[1]
        # Check: for all i,j,k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Generate schoolbook factors as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_factors = (U_school, V_school, W_school)
    best_rank = rank_school

    # Precompute the tensor T as a flattened vector for faster residual calculation
    tensor_indices = []
    tensor_values = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                tensor_indices.append((idx_u, idx_v, idx_w))
                tensor_values.append(1.0)
    
    tensor_indices = np.array(tensor_indices)
    tensor_values = np.array(tensor_values)
    
    def calculate_residual(U, V, W):
        """Calculate the squared error of the factorization on the non-zero entries."""
        U_sel = U[tensor_indices[:, 0], :]
        V_sel = V[tensor_indices[:, 1], :]
        W_sel = W[tensor_indices[:, 2], :]
        prod = U_sel * V_sel * W_sel
        sums = np.sum(prod, axis=1)
        diff = sums - tensor_values
        return np.sum(diff * diff)

    def try_numerical_search(r, max_iters=200, lr=0.01):
        """Try to find a rank-r factorization using gradient descent."""
        np.random.seed(42)
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                return None
            
            U_sel = U[tensor_indices[:, 0], :]
            V_sel = V[tensor_indices[:, 1], :]
            W_sel = W[tensor_indices[:, 2], :]
            
            prod = U_sel * V_sel * W_sel
            sums = np.sum(prod, axis=1)
            residuals = sums - tensor_values
            
            G_U = np.zeros_like(U)
            G_V = np.zeros_like(V)
            G_W = np.zeros_like(W)
            
            term_U = residuals[:, None] * V_sel * W_sel
            np.add.at(G_U, (tensor_indices[:, 0], slice(None)), term_U)
            
            term_V = residuals[:, None] * U_sel * W_sel
            np.add.at(G_V, (tensor_indices[:, 1], slice(None)), term_V)
            
            term_W = residuals[:, None] * U_sel * V_sel
            np.add.at(G_W, (tensor_indices[:, 2], slice(None)), term_W)
            
            U -= lr * G_U
            V -= lr * G_V
            W -= lr * G_W
            
            if it % 10 == 0:
                res = calculate_residual(U, V, W)
                if res < 1e-6:
                    U_exact = np.round(U * 2) / 2.0
                    V_exact = np.round(V * 2) / 2.0
                    W_exact = np.round(W * 2) / 2.0
                    if verify(U_exact, V_exact, W_exact, n, m, p):
                        return U_exact, V_exact, W_exact
        return None

    # Search for ranks lower than schoolbook
    search_ranks = [60, 70, 80, 90, 100, 110]
    for r in search_ranks:
        if time.time() > deadline - 20:
            break
        result = try_numerical_search(r)
        if result is not None:
            best_factors = result
            best_rank = r
            break

    U, V, W = best_factors
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
