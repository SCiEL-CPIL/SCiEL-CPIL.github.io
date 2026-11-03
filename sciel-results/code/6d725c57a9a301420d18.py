import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = 2, 3, 6
    
    # Precompute indices
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i, j, k))
                
    def schoolbook():
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def verify(U, V, W):
        r = U.shape[1]
        # Check all valid indices
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-6:
                        return False
        # Check some invalid indices to ensure 0
        # We can't check all, but we can check a few random ones or rely on the structure
        # Actually, if the rank is low, the "leakage" to invalid indices might be high.
        # The schoolbook has 0 leakage.
        # A low rank solution might have non-zero leakage.
        # The problem statement says "exactly decompose".
        # So we MUST check leakage.
        # We can check all possible (a,b,cc) that are NOT in the valid set.
        # Total size: 6 * 18 * 12 = 1296.
        # Valid: 36.
        # Invalid: 1260.
        # Checking all is fast.
        valid_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid_set.add((i*m+j, j*p+k, k*n+i))
                    
        for a in range(n*m):
            for b in range(m*p):
                for cc in range(p*n):
                    if (a,b,cc) not in valid_set:
                        val = 0.0
                        for c in range(r):
                            val += U[a, c] * V[b, c] * W[cc, c]
                        if abs(val) > 1e-6:
                            return False
        return True

    def compute_error(U, V, W):
        r = U.shape[1]
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (val - 1.0) ** 2
        # Add penalty for leakage?
        # For now, just the target error.
        return err

    def flatten_params(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

    def unflatten_params(params, rank):
        size_U = n * m * rank
        size_V = m * p * rank
        U = params[:size_U].reshape(n * m, rank)
        V = params[size_U:size_U + size_V].reshape(m * p, rank)
        W = params[size_U + size_V:].reshape(p * n, rank)
        return U, V, W

    def objective(params, rank):
        U, V, W = unflatten_params(params, rank)
        return compute_error(U, V, W)

    def solve_for_W(U, V, r):
        W = np.zeros((p * n, r))
        for i in range(n):
            for k in range(p):
                A = np.zeros((m, r))
                for j in range(m):
                    for c in range(r):
                        A[j, c] = U[i * m + j, c] * V[j * p + k, c]
                b = np.ones(m)
                try:
                    w = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    w = np.linalg.pinv(A) @ b
                W[k * n + i, :] = w
        return W

    def solve_for_U(V, W, r):
        U = np.zeros((n * m, r))
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, r))
                for k in range(p):
                    for c in range(r):
                        A[k, c] = V[j * p + k, c] * W[k * n + i, c]
                b = np.ones(p)
                try:
                    u = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    u = np.linalg.pinv(A) @ b
                U[i * m + j, :] = u
        return U

    def solve_for_V(U, W, r):
        V = np.zeros((m * p, r))
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    for c in range(r):
                        A[i, c] = U[i * m + j, c] * W[k * n + i, c]
                b = np.ones(n)
                try:
                    v = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    v = np.linalg.pinv(A) @ b
                V[j * p + k, :] = v
        return V

    for rank in range(1, best_rank):
        if time.time() > deadline - 20:
            break
            
        found = False
        # Try multiple random initializations
        for init in range(200):
            if time.time() > deadline - 10:
                break
                
            # Random initialization
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # ALS iterations
            for it in range(50):
                W = solve_for_W(U, V, rank)
                U = solve_for_U(V, W, rank)
                V = solve_for_V(U, W, rank)
                
            # Polish with scipy
            params = flatten_params(U, V, W)
            res = minimize(objective, params, args=(rank,), method='L-BFGS-B', options={'maxiter': 100, 'gtol': 1e-8})
            U_pol, V_pol, W_pol = unflatten_params(res.x, rank)
            
            # Round to 0.5
            U_r = np.round(U_pol * 2) / 2
            V_r = np.round(V_pol * 2) / 2
            W_r = np.round(W_pol * 2) / 2
            
            if verify(U_r, V_r, W_r):
                best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                best_rank = rank
                found = True
                break
        if found:
            break
            
    return best_result
