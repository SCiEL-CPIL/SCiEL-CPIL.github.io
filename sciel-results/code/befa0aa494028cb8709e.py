import time
import numpy as np
from scipy.linalg import lstsq

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    n, m, p = DIMENSIONS
    
    def schoolbook(n, m, p):
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
        return U, V, W
    
    def verify_decomp(U, V, W, n, m, p):
        rank = U.shape[1]
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Laderman 3x3x3 factors
    laderman_23 = [
        ([1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]),
        ([0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0]),
        ([0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1]),
        ([0,1,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0], [0,0,0,0,1,0,0,0,0]),
        ([0,0,0,1,0,0,0,0,0], [0,1,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0]),
        ([0,0,1,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0], [0,0,0,0,1,0,0,0,0]),
        ([0,0,0,0,0,0,1,0,0], [0,0,1,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0]),
        ([0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,0,1], [0,0,0,0,1,0,0,0,0]),
        ([0,0,0,0,0,0,0,1,0], [0,0,0,0,0,1,0,0,0], [0,0,0,0,1,0,0,0,0]),
        ([1,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0]),
        ([0,0,0,0,1,0,0,0,0], [1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]),
        ([0,1,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0], [0,0,0,0,0,1,0,0,0]),
        ([0,0,0,1,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]),
        ([0,0,1,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1]),
        ([0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,1,0,0], [1,0,0,0,0,0,0,0,0]),
        ([0,0,0,0,0,0,1,0,0], [1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]),
        ([1,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,0,1]),
        ([0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1], [0,0,1,0,0,0,0,0,0]),
        ([0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,1,0,0], [1,0,0,0,0,0,0,0,0]),
        ([0,0,0,0,0,0,0,1,0], [0,0,0,0,0,1,0,0,0], [0,0,1,0,0,0,0,0,0]),
        ([0,0,0,0,1,0,0,0,0], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1]),
        ([0,0,1,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,1], [0,0,1,0,0,0,0,0,0]),
        ([0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,1,0,0], [1,0,0,0,0,0,0,0,0]),
    ]
    
    def verify_3x3x3(factors):
        Ua = np.array([f[0] for f in factors], dtype=float)
        Va = np.array([f[1] for f in factors], dtype=float)
        Wa = np.array([f[2] for f in factors], dtype=float)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = np.sum(Ua[:, i*3+j] * Va[:, j*3+k] * Wa[:, k*3+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    laderman_ok = verify_3x3x3(laderman_23)
    
    if not laderman_ok:
        # Fall back to full schoolbook
        U, V, W = schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a better decomposition for the 3x3x4 part
    # Start with schoolbook for 3x3x4
    U4, V4, W4 = schoolbook(3, 3, 4)
    rank4 = 36
    
    # Try to reduce rank by removing one term and solving for W
    best_rank4 = rank4
    best_U4, best_V4, best_W4 = U4, V4, W4
    
    for col_to_remove in range(rank4):
        if time.time() > deadline - 30:
            break
        
        # Remove the column
        mask = [c for c in range(rank4) if c != col_to_remove]
        U4_reduced = U4[:, mask]
        V4_reduced = V4[:, mask]
        
        # Build the linear system for W
        # For each (i,j,k), sum_t U[i*3+j,t] * V[j*4+k,t] * W[k*3+i,t] = 1
        # Unknowns: W[k*3+i, t] for k in 0..3, i in 0..2, t in 0..34
        # That's 12 * 35 = 420 unknowns
        
        num_eqs = 3 * 3 * 4  # 36
        num_unknowns = 12 * 35  # 420
        
        A = np.zeros((num_eqs, num_unknowns))
        b = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(3):
            for j in range(3):
                for k in range(4):
                    for t in range(35):
                        # W[k*3+i, t] is unknown index (k*3+i) * 35 + t
                        unknown_idx = (k*3+i) * 35 + t
                        coeff = U4_reduced[i*3+j, t] * V4_reduced[j*4+k, t]
                        A[eq_idx, unknown_idx] = coeff
                    eq_idx += 1
        
        # Solve the underdetermined system
        W4_solution, residuals, rank, sv = lstsq(A, b, rcond=None)
        
        # Reshape W4_solution
        W4_solution = W4_solution.reshape(12, 35)
        
        # Check if entries are close to integer/half-integer
        # Scale by 2 and check if close to integers
        W4_scaled = W4_solution * 2
        W4_rounded = np.round(W4_scaled)
        if np.all(np.abs(W4_scaled - W4_rounded) < 1e-6):
            W4_candidate = W4_rounded / 2.0
            # Verify
            if verify_decomp(U4_reduced, V4_reduced, W4_candidate, 3, 3, 4):
                best_rank4 = 35
                best_U4, best_V4, best_W4 = U4_reduced, V4_reduced, W4_candidate
                break
    
    # Build the full decomposition
    total_rank = 23 + best_rank4
    U = np.zeros((9, total_rank))
    V = np.zeros((21, total_rank))
    W = np.zeros((21, total_rank))
    
    # Fill Laderman terms
    for r, (u9, v9, w9) in enumerate(laderman_23):
        U[0:9, r] = u9
        V[0:9, r] = v9
        W[0:9, r] = w9
    
    # Fill 3x3x4 terms
    col_offset = 23
    U[col_offset:col_offset+best_rank4, col_offset:col_offset+best_rank4] = best_U4
    V[9:9+12, col_offset:col_offset+best_rank4] = best_V4
    W[9:9+12, col_offset:col_offset+best_rank4] = best_W4
    
    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if verify_decomp(U, V, W, 3, 3, 7):
        return result
    else:
        # Fall back to original rank-59
        U59, V59, W59 = schoolbook(3, 3, 7)
        # But use Laderman for first 23
        U59 = np.zeros((9, 59))
        V59 = np.zeros((21, 59))
        W59 = np.zeros((21, 59))
        for r, (u9, v9, w9) in enumerate(laderman_23):
            U59[0:9, r] = u9
            V59[0:9, r] = v9
            W59[0:9, r] = w9
        col = 23
        for i in range(3):
            for j in range(3):
                for k in range(3, 7):
                    U59[i*3+j, col] = 1
                    V59[j*7+k, col] = 1
                    W59[k*3+i, col] = 1
                    col += 1
        return {"U": U59.tolist(), "V": V59.tolist(), "W": W59.tolist()}
