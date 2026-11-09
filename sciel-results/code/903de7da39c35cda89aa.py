"""Exact rank-minimal search for 3x5 * 5x7 matrix multiplication."""
import time
import numpy as np
import random

DIMENSIONS = [3, 5, 7]
DEADLINE = 150.0  # seconds; evaluator kills at 150s, use safety margin

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 5, 7
    schoolbook_rank = n * m * p
    
    def get_schoolbook():
        rank = schoolbook_rank
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        return {"U": U_list, "V": V_list, "W": W_list}

    def verify(U_np, V_np, W_np, rank):
        U_reshaped = U_np.reshape(n, m, rank)
        V_reshaped = V_np.reshape(m, p, rank)
        W_reshaped = W_np.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-6)

    def is_half_integer(arr, tol=1e-6):
        # Check if arr is close to k/2 for integer k
        doubled = 2.0 * arr
        rounded = np.round(doubled)
        return np.allclose(doubled, rounded, atol=tol)

    def snap_to_half_integer(arr):
        return np.round(2.0 * arr) / 2.0

    def solve_for_W(U, V, rank):
        # U: (n*m, rank), V: (m*p, rank)
        # We need W: (p*n, rank)
        # For each k in range(p), i in range(n):
        #   Solve for W[k*n+i, :]
        #   System: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j in range(m)
        #   Let A be m x rank matrix where A[j, r] = U[i*m+j, r] * V[j*p+k, r]
        #   We want A @ w = ones(m)
        W = np.zeros((p * n, rank), dtype=float)
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank), dtype=float)
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                
                try:
                    # Solve A w = 1
                    # Use least squares if rank deficient
                    w, residuals, rank_a, s = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    # Check if solution is valid half-integer
                    if is_half_integer(w):
                        w_snapped = snap_to_half_integer(w)
                        # Verify exactness with snapped values
                        if np.allclose(A @ w_snapped, np.ones(m), atol=1e-6):
                            W[k * n + i, :] = w_snapped
                        else:
                            return None
                    else:
                        return None
                except np.linalg.LinAlgError:
                    return None
        return W

    def solve_for_U(V, W, rank):
        # V: (m*p, rank), W: (p*n, rank)
        # We need U: (n*m, rank)
        # For each i in range(n), j in range(m):
        #   Solve for U[i*m+j, :]
        #   System: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k in range(p)
        #   Let A be p x rank matrix where A[k, r] = V[j*p+k, r] * W[k*n+i, r]
        #   We want A @ u = ones(p)
        U = np.zeros((n * m, rank), dtype=float)
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank), dtype=float)
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                
                try:
                    u, residuals, rank_a, s = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    if is_half_integer(u):
                        u_snapped = snap_to_half_integer(u)
                        if np.allclose(A @ u_snapped, np.ones(p), atol=1e-6):
                            U[i * m + j, :] = u_snapped
                        else:
                            return None
                    else:
                        return None
                except np.linalg.LinAlgError:
                    return None
        return U

    def solve_for_V(U, W, rank):
        # U: (n*m, rank), W: (p*n, rank)
        # We need V: (m*p, rank)
        # For each j in range(m), k in range(p):
        #   Solve for V[j*p+k, :]
        #   System: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i in range(n)
        #   Let A be n x rank matrix where A[i, r] = U[i*m+j, r] * W[k*n+i, r]
        #   We want A @ v = ones(n)
        V = np.zeros((m * p, rank), dtype=float)
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank), dtype=float)
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                
                try:
                    v, residuals, rank_a, s = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    if is_half_integer(v):
                        v_snapped = snap_to_half_integer(v)
                        if np.allclose(A @ v_snapped, np.ones(n), atol=1e-6):
                            V[j * p + k, :] = v_snapped
                        else:
                            return None
                    else:
                        return None
                except np.linalg.LinAlgError:
                    return None
        return V

    def random_init_U(rank):
        U = np.zeros((n * m, rank), dtype=float)
        for r in range(rank):
            # Pick 1 or 2 random non-zeros
            num_nz = random.choice([1, 2])
            indices = random.sample(range(n * m), num_nz)
            for idx in indices:
                U[idx, r] = random.choice([0.5, 1.0, -0.5, -1.0])
        return U

    def random_init_V(rank):
        V = np.zeros((m * p, rank), dtype=float)
        for r in range(rank):
            num_nz = random.choice([1, 2])
            indices = random.sample(range(m * p), num_nz)
            for idx in indices:
                V[idx, r] = random.choice([0.5, 1.0, -0.5, -1.0])
        return V

    # Try to find a rank < 105
    # We will search downwards from 104
    # Limit time to 140s
    search_limit = 140.0
    
    for rank in range(schoolbook_rank - 1, max(50, schoolbook_rank - 10), -1):
        if time.time() - start_time > search_limit:
            break
            
        # Try a few random restarts for this rank
        restarts = 20
        found = False
        for _ in range(restarts):
            if time.time() - start_time > search_limit:
                break
                
            U = random_init_U(rank)
            V = random_init_V(rank)
            
            # Alternating minimization
            for step in range(50):
                if time.time() - start_time > search_limit:
                    break
                
                W = solve_for_W(U, V, rank)
                if W is None:
                    break
                
                U_new = solve_for_U(V, W, rank)
                if U_new is None:
                    break
                U = U_new
                
                V_new = solve_for_V(U, W, rank)
                if V_new is None:
                    break
                V = V_new
                
                # Check if we have a valid solution
                if verify(U, V, W, rank):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        if found:
            break

    # Fallback to schoolbook
    return get_schoolbook()

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
