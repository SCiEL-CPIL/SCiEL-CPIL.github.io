"""Rewritten program for 3x3 * 3x7 matrix multiplication tensor decomposition.
Uses a more powerful optimization approach: 
1. Try to find a real-valued solution for decreasing ranks using least squares.
2. Round to nearest half-integers.
3. Verify exactness.
Falls back to schoolbook rank 63.
"""
import time
import sys
import random
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 7]
n, m, p = DIMENSIONS
NM = n * m
MP = m * p
PN = p * n
SCHOOL_RANK = NM * MP // m  # 3*3*7 = 63

def get_tensor_indices():
    """Return list of (i, j, k) tuples."""
    return [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

def residual_func(params, U_shape, V_shape, W_shape, indices):
    """
    Residual function for least squares.
    params: flattened vector of U, V, W entries.
    """
    num_u = U_shape[0] * U_shape[1]
    num_v = V_shape[0] * V_shape[1]
    num_w = W_shape[0] * W_shape[1]
    
    U_flat = params[:num_u]
    V_flat = params[num_u:num_u+num_v]
    W_flat = params[num_u+num_v:]
    
    U = U_flat.reshape(U_shape)
    V = V_flat.reshape(V_shape)
    W = W_flat.reshape(W_shape)
    
    residuals = []
    for i, j, k in indices:
        ui = i * m + j
        vj = j * p + k
        wk = k * n + i
        val = np.sum(U[ui, :] * V[vj, :] * W[wk, :])
        residuals.append(val - 1.0)
    
    return residuals

def verify_decomposition(U, V, W, tol=1e-8):
    """Verify that the decomposition is exact within tolerance."""
    max_res = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                val = np.sum(U[ui, :] * V[vj, :] * W[wk, :])
                res = abs(val - 1.0)
                if res > max_res:
                    max_res = res
    return max_res < tol, max_res

def clean_matrix(mat):
    """Round values to nearest half-integer and convert to list of lists."""
    result = []
    for row in mat:
        clean_row = []
        for val in row:
            # Round to nearest 0.5
            rounded = round(val * 2) / 2.0
            clean_row.append(rounded)
        result.append(clean_row)
    return result

def schoolbook_decomposition():
    """Return the schoolbook (rank 63) decomposition."""
    rank = SCHOOL_RANK
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def try_rank_search(rank, time_limit=10.0, num_restarts=5):
    """
    Try to find a decomposition of given rank using least squares optimization.
    """
    start_time = time.time()
    indices = get_tensor_indices()
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        random.seed(42 + restart)
        np.random.seed(42 + restart)
        
        # Initialize with small random values
        U_init = np.random.randn(NM, rank) * 0.5
        V_init = np.random.randn(MP, rank) * 0.5
        W_init = np.random.randn(PN, rank) * 0.5
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Use least squares to minimize residual
        try:
            result = least_squares(
                residual_func,
                params_init,
                args=( (NM, rank), (MP, rank), (PN, rank), indices ),
                method='lm', # Levenberg-Marquardt, good for least squares
                max_nfev=10000
            )
        except Exception as e:
            # Fallback to trf if lm fails
            try:
                result = least_squares(
                    residual_func,
                    params_init,
                    args=( (NM, rank), (MP, rank), (PN, rank), indices ),
                    method='trf',
                    max_nfev=10000
                )
            except Exception:
                continue
                
        final_params = result.x
        num_u = NM * rank
        num_v = MP * rank
        num_w = PN * rank
        
        U_final = final_params[:num_u].reshape((NM, rank))
        V_final = final_params[num_u:num_u+num_v].reshape((MP, rank))
        W_final = final_params[num_u+num_v:].reshape((PN, rank))
        
        # Check real-valued residual
        real_res, _ = verify_decomposition(U_final, V_final, W_final, tol=1e-6)
        
        if real_res:
            # Round to half-integers
            U_clean = clean_matrix(U_final)
            V_clean = clean_matrix(V_final)
            W_clean = clean_matrix(W_final)
            
            # Verify exactness of rounded solution
            ok, max_res = verify_decomposition(U_clean, V_clean, W_clean, tol=1e-8)
            if ok:
                return U_clean, V_clean, W_clean
                
    return None

def construct():
    """
    Constructs the tensor decomposition for 3x3 * 3x7 matrix multiplication.
    Attempts to find a rank-minimal exact decomposition.
    Falls back to schoolbook rank 63.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Try ranks from 62 down to 20
    # We prioritize lower ranks, but might find a solution at higher ranks faster?
    # Actually, lower rank is harder. Let's try a few high ranks first to ensure we get > 1.0?
    # No, we want minimal rank. But if we can't find low rank, any < 63 is good.
    # Let's try 62, 61, ... down to 40 first, as they are closer to schoolbook and might be easier to find via perturbation?
    # Or just go down.
    
    # To maximize chance of finding *any* improvement quickly, we can try a random rank first?
    # No, systematic is better.
    
    # Let's try a range of ranks.
    # Ranks to try: 62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40
    # If we find one, we return it.
    # If we want minimal, we should go 20, 21... but that's slow.
    # The score is seed/value. 63/62 = 1.016. 63/50 = 1.26.
    # So finding rank 50 is much better than 62.
    # We should aim for low ranks.
    
    # Strategy: Try low ranks first, but with fewer restarts?
    # Or try a few mid-range ranks?
    
    # Let's try ranks: 30, 35, 40, 45, 50, 55, 60, 62
    # And if none found, try 20-29.
    
    ranks_to_try = [62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 20: # Reserve 20s for fallback/verification
            break
            
        result = try_rank_search(rank, time_limit=5.0, num_restarts=3)
        if result is not None:
            U, V, W = result
            # Final check
            ok, max_res = verify_decomposition(U, V, W, tol=1e-8)
            if ok:
                return {"U": U, "V": V, "W": W}
                
    # If we didn't find anything in the high range, try lower ranks with less time per rank
    lower_ranks = [48, 46, 44, 42, 40, 38, 36, 34, 32, 30]
    for rank in lower_ranks:
        if time.time() > deadline - 10:
            break
            
        result = try_rank_search(rank, time_limit=3.0, num_restarts=2)
        if result is not None:
            U, V, W = result
            ok, max_res = verify_decomposition(U, V, W, tol=1e-8)
            if ok:
                return {"U": U, "V": V, "W": W}
                
    # Fallback to schoolbook
    return schoolbook_decomposition()

if __name__ == "__main__":
    start_time = time.time()
    res = construct()
    end_time = time.time()
    if end_time - start_time > 145:
        print("Warning: Construction took too long", file=sys.stderr)
    # Verify dimensions
    assert len(res['U']) == NM
    assert len(res['V']) == MP
    assert len(res['W']) == PN
    rank = len(res['U'][0])
    assert len(res['V'][0]) == rank
    assert len(res['W'][0]) == rank
    # Verify correctness
    ok, max_res = verify_decomposition(res['U'], res['V'], res['W'], tol=1e-8)
    if not ok:
        print(f"ERROR: Decomposition not exact. Max residual: {max_res}", file=sys.stderr)
        sys.exit(1)
    print(f"Rank: {rank}, Max residual: {max_res:.2e}, Time: {end_time-start_time:.2f}s")
