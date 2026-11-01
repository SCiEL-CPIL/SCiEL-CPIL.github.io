import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    n, m, p = 2, 2, 7
    start_time = time.time()
    DEADLINE = 140  # Leave 10s margin for evaluator's 150s limit
    
    # Precompute index arrays for vectorized residual
    # T[i,j,k] = sum_r U[i,r]*V[j*7+k,r]*W[k*2+i,r] = 1
    # 28 entries: i in {0,1}, j in {0,1}, k in {0..6}
    
    def make_residuals(R):
        """Create vectorized residual function for given rank R"""
        # Build the tensor structure
        # For each (i,j,k) entry, we need U[i,:], V[j*7+k,:], W[k*2+i,:]
        idx_i = []
        idx_v = []
        idx_w = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_i.append(i)
                    idx_v.append(j * p + k)
                    idx_w.append(k * n + i)
        idx_i = np.array(idx_i)
        idx_v = np.array(idx_v)
        idx_w = np.array(idx_w)
        
        def residuals(x):
            U = x[:n*m*R].reshape(n*m, R)
            V = x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
            W = x[n*m*R + m*p*R:].reshape(p*n, R)
            
            # Vectorized: for each entry t, compute sum_r U[idx_i[t],r]*V[idx_v[t],r]*W[idx_w[t],r]
            U_sel = U[idx_i, :]  # (28, R)
            V_sel = V[idx_v, :]  # (28, R)
            W_sel = W[idx_w, :]  # (28, R)
            sums = np.sum(U_sel * V_sel * W_sel, axis=1)  # (28,)
            return sums - 1.0
        
        return residuals
    
    # Try rank 24
    R = 24
    residuals_24 = make_residuals(R)
    
    best_x = None
    best_cost = np.inf
    
    # Strategy 1: Start from rank-25 solution, zero out one column
    # First build the rank-25 solution
    rank25 = _build_rank25()
    
    if rank25 is not None and time.time() - start_time < DEADLINE - 30:
        U25, V25, W25 = rank25
        # Zero out the last column to get a rank-24 init
        U0 = U25[:, :R].copy()
        V0 = V25[:, :R].copy()
        W0 = W25[:, :R].copy()
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            sol = least_squares(residuals_24, x0, method='lm', max_nfev=2000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if sol.cost < best_cost:
                best_cost = sol.cost
                best_x = sol.x
        except Exception:
            pass
    
    # Strategy 2: Random initializations
    if time.time() - start_time < DEADLINE - 20:
        for attempt in range(8):
            if time.time() - start_time > DEADLINE - 15:
                break
            np.random.seed(42 + attempt)
            x0 = np.random.randn((n*m + m*p + p*n) * R) * 0.3
            try:
                sol = least_squares(residuals_24, x0, method='lm', max_nfev=1500)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_x = sol.x
            except Exception:
                pass
    
    # Try to use the best rank-24 solution
    if best_x is not None and best_cost < 1e-8:
        U = best_x[:n*m*R].reshape(n*m, R)
        V = best_x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
        W = best_x[n*m*R + m*p*R:].reshape(p*n, R)
        
        # Try rounding to half-integers
        for scale in [2, 4, 8]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            
            # Verify
            res = residuals_24(np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()]))
            if np.max(np.abs(res)) < 1e-6:
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        # If no rounding works, check if entries are already close to half-integers
        # and return as-is if valid
        res = residuals_24(best_x)
        if np.max(np.abs(res)) < 1e-8:
            # Return with entries rounded to 6 decimal places to clean up
            U_c = np.round(U, 6)
            V_c = np.round(V, 6)
            W_c = np.round(W, 6)
            res_c = residuals_24(np.concatenate([U_c.flatten(), V_c.flatten(), W_c.flatten()]))
            if np.max(np.abs(res_c)) < 1e-6:
                return {"U": U_c.tolist(), "V": V_c.tolist(), "W": W_c.tolist()}
    
    # Fallback to rank 25
    if rank25 is None:
        rank25 = _build_rank25()
    
    if rank25 is not None:
        U, V, W = rank25
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Last resort: schoolbook rank 28
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i, idx] = 1
                V[j*p+k, idx] = 1
                W[k*n+i, idx] = 1
                idx += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _build_rank25():
    """Build a verified rank-25 decomposition for 2x2 * 2x7"""
    n, m, p = 2, 2, 7
    rank = 25
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Use Strassen-like blocks for the first 6 columns of the 7-column matrix
    # Each block handles 2 columns of the right matrix using 7 terms (Strassen for 2x2*2x2)
    
    def add_strassen_block(col_start, k_start):
        """Add a 2x2 * 2x2 Strassen block for right-matrix columns k_start, k_start+1"""
        # Term 0: (A00+A11)*(B00+B11)
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1
        V[7*1 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1
        
        # Term 1: (A10+A11)*B00
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1
        W[2*(k_start+1) + 1, col_start+1] -= 1
        
        # Term 2: A00*(B10-B11)
        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1
        V[7*1 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1
        W[2*(k_start+1) + 1, col_start+2] += 1
        
        # Term 3: A11*(B01-B00)
        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1
        V[7*0 + k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1
        W[2*k_start + 1, col_start+3] += 1
        
        # Term 4: (A00+A01)*B11
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 0, col_start+4] += 1
        
        # Term 5: (A10-A00)*(B00+B10)
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1
        V[7*0 + (k_start+1), col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1
        
        # Term 6: (A01-A11)*(B11+B01)
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1
        V[7*1 + (k_start+1), col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1
    
    # Apply Strassen blocks for column pairs (0,1), (2,3), (4,5)
    add_strassen_block(0, 0)
    add_strassen_block(7, 2)
    add_strassen_block(14, 4)
    
    # Handle the last column (k=6) with 4 schoolbook terms
    col_start = 21
    k = 6
    U[0, col_start+0] += 1
    V[7*0 + k, col_start+0] += 1
    W[2*k + 0, col_start+0] += 1

    U[1, col_start+1] += 1
    V[7*1 + k, col_start+1] += 1
    W[2*k + 0, col_start+1] += 1

    U[2, col_start+2] += 1
    V[7*0 + k, col_start+2] += 1
    W[2*k + 1, col_start+2] += 1

    U[3, col_start+3] += 1
    V[7*1 + k, col_start+3] += 1
    W[2*k + 1, col_start+3] += 1
    
    # Verify the decomposition
    def make_residuals(R):
        idx_i = []
        idx_v = []
        idx_w = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_i.append(i)
                    idx_v.append(j * p + k)
                    idx_w.append(k * n + i)
        idx_i = np.array(idx_i)
        idx_v = np.array(idx_v)
        idx_w = np.array(idx_w)
        
        U_sel = U[idx_i, :]
        V_sel = V[idx_v, :]
        W_sel = W[idx_w, :]
        sums = np.sum(U_sel * V_sel * W_sel, axis=1)
        return sums - 1.0
    
    res = make_residuals(rank)
    if np.max(np.abs(res)) > 1e-10:
        return None  # Invalid
    
    return U, V, W
