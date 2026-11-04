"""Matrix multiplication 3x4x6 - Optimized ALS with integer check."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _trivial_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify the decomposition."""
    rank = U.shape[1]
    Ut = U.reshape(n, m, rank)
    Vt = V.reshape(m, p, rank)
    Wt = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
    return np.max(np.abs(T - 1.0)) < tol


def _check_integer_half_integer(U, V, W, tol=1e-4):
    """Check if entries are close to integers or half-integers."""
    def check_matrix(M):
        # Check if M is close to integers
        int_diff = np.abs(M - np.round(M))
        # Check if M is close to half-integers (0.5, 1.5, etc.)
        half_diff = np.abs(M - (np.round(M * 2) / 2.0))
        return np.max(int_diff) < tol or np.max(half_diff) < tol
    
    return check_matrix(U) and check_matrix(V) and check_matrix(W)


def _als_decompose(n, m, p, r, max_sweeps=200, tol=1e-10, seed=42):
    """
    ALS for rank-r decomposition.
    """
    np.random.seed(seed)
    
    # Initialize
    U = np.random.randn(n * m, r) * 0.5
    V = np.random.randn(m * p, r) * 0.5
    W = np.random.randn(p * n, r) * 0.5
    
    Ut = U.reshape(n, m, r)
    Vt = V.reshape(m, p, r)
    Wt = W.reshape(p, n, r)
    
    prev_loss = np.inf
    
    for sweep in range(max_sweeps):
        # --- Update U ---
        new_Ut = np.zeros_like(Ut)
        for i in range(n):
            for j in range(m):
                B = V[j*p:(j+1)*p, :] * W[:, i, :]  # shape (p, r)
                try:
                    Ut_ij = np.linalg.lstsq(B, np.ones(p), rcond=None)[0]
                    new_Ut[i, j, :] = Ut_ij
                except:
                    new_Ut[i, j, :] = Ut[i, j, :]
        U = new_Ut.reshape(n * m, r)
        Ut = U.reshape(n, m, r)
        
        # --- Update V ---
        new_Vt = np.zeros_like(Vt)
        for j in range(m):
            for k in range(p):
                A = U[:, j, :] * W[k, :, :]  # shape (n, r)
                try:
                    Vt_jk = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                    new_Vt[j, k, :] = Vt_jk
                except:
                    new_Vt[j, k, :] = Vt[j, k, :]
        V = new_Vt.reshape(m * p, r)
        Vt = V.reshape(m, p, r)
        
        # --- Update W ---
        new_Wt = np.zeros_like(Wt)
        for k in range(p):
            for i in range(n):
                C = U[i, :, :] * V[:, k, :]  # shape (m, r)
                try:
                    Wt_ki = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                    new_Wt[k, i, :] = Wt_ki
                except:
                    new_Wt[k, i, :] = Wt[k, i, :]
        W = new_Wt.reshape(p * n, r)
        Wt = W.reshape(p, n, r)
        
        # Compute loss
        T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
        loss = np.sum((T - 1.0) ** 2)
        
        if loss < tol:
            break
        
        if abs(prev_loss - loss) < 1e-15:
            break
        
        prev_loss = loss
    
    # Final check
    T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
    if np.max(np.abs(T - 1.0)) < 1e-5:
        # Check integer/half-integer constraint
        if _check_integer_half_integer(U, V, W):
            return U, V, W
        else:
            # Try rounding?
            U_r = np.round(U * 2) / 2.0
            V_r = np.round(V * 2) / 2.0
            W_r = np.round(W * 2) / 2.0
            if _verify(U_r, V_r, W_r, n, m, p, tol=1e-4):
                return U_r, V_r, W_r
            return None
    return None


def construct():
    start_time = time.time()
    deadline = 140.0
    n, m, p = DIMENSIONS
    
    U_trivial, V_trivial, W_trivial = _trivial_decomposition(n, m, p)
    best = {'U': U_trivial, 'V': V_trivial, 'W': W_trivial}
    best_rank = n * m * p
    
    # Try ranks from 71 down to 60 (likely range if any improvement exists)
    # Using fewer seeds and sweeps to stay within time limit
    for r in range(71, 59, -1):
        if time.time() > deadline - 10:
            break
        
        for seed in range(3):
            if time.time() > deadline - 5:
                break
            
            result = _als_decompose(n, m, p, r, max_sweeps=100, tol=1e-10, seed=seed * 100 + 42)
            
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p) and _check_integer_half_integer(U, V, W):
                    best = {'U': U, 'V': V, 'W': W}
                    best_rank = r
                    break
        
        if best_rank < n * m * p:
            break

    return {'U': best['U'].tolist(), 'V': best['V'].tolist(), 'W': best['W'].tolist()}
# EVOLVE-BLOCK-END
