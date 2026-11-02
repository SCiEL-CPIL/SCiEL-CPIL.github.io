import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = 2, 3, 6
    
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
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        try:
            T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
            if not np.allclose(T, 1.0, atol=1e-6):
                return False
            return True
        except Exception:
            return False

    def solve_for_W(U, V, r):
        W = np.zeros((p * n, r))
        for i in range(n):
            for k in range(p):
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                b = np.ones(m)
                try:
                    w = np.linalg.pinv(A) @ b
                except:
                    w = np.zeros(r)
                W[k * n + i, :] = w
        return W

    def solve_for_U(V, W, r):
        U = np.zeros((n * m, r))
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                b = np.ones(p)
                try:
                    u = np.linalg.pinv(A) @ b
                except:
                    u = np.zeros(r)
                U[i * m + j, :] = u
        return U

    def solve_for_V(U, W, r):
        V = np.zeros((m * p, r))
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                b = np.ones(n)
                try:
                    v = np.linalg.pinv(A) @ b
                except:
                    v = np.zeros(r)
                V[j * p + k, :] = v
        return V

    def compute_error(U, V, W):
        r = U.shape[1]
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        return np.sum((T - 1.0) ** 2)

    def snap_to_half_integers(U, V, W):
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        if verify(U_snap, V_snap, W_snap):
            return U_snap, V_snap, W_snap
        return None

    # Search for lower rank
    for rank in range(18, best_rank):
        if time.time() > deadline - 15:
            break
            
        found = False
        # Try multiple random initializations with different strategies
        for init in range(150):
            if time.time() > deadline - 8:
                break
                
            # Different initialization strategies
            if init % 3 == 0:
                # Standard random
                U = np.random.randn(n * m, rank)
                V = np.random.randn(m * p, rank)
                W = np.random.randn(p * n, rank)
            elif init % 3 == 1:
                # Sparse initialization
                U = np.random.randn(n * m, rank)
                U[np.random.rand(n * m, rank) < 0.5] = 0
                V = np.random.randn(m * p, rank)
                V[np.random.rand(m * p, rank) < 0.5] = 0
                W = np.random.randn(p * n, rank)
                W[np.random.rand(p * n, rank) < 0.5] = 0
            else:
                # Binary initialization
                U = np.random.randint(0, 2, (n * m, rank)) * 2 - 1
                V = np.random.randint(0, 2, (m * p, rank)) * 2 - 1
                W = np.random.randint(0, 2, (p * n, rank)) * 2 - 1
            
            # Iterate with adaptive stopping
            prev_err = float('inf')
            for it in range(200):
                W = solve_for_W(U, V, rank)
                U = solve_for_U(V, W, rank)
                V = solve_for_V(U, W, rank)
                
                err = compute_error(U, V, W)
                
                # Check for convergence
                if err < 1e-8:
                    # Try to snap to half-integers
                    snapped = snap_to_half_integers(U, V, W)
                    if snapped:
                        best_result = {"U": snapped[0].tolist(), "V": snapped[1].tolist(), "W": snapped[2].tolist()}
                        best_rank = rank
                        found = True
                        break
                    if verify(U, V, W):
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        best_rank = rank
                        found = True
                        break
                
                # Early stop if no progress
                if abs(prev_err - err) < 1e-12 and it > 10:
                    break
                prev_err = err
            if found:
                break
        if found:
            break
            
    return best_result
