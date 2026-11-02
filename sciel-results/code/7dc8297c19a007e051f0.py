import time
import numpy as np

DIMENSIONS = [3, 3, 8]

def _validate(U, V, W, n=3, m=3, p=8):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                total = 0
                for r in range(rank):
                    total += U[a][r] * V[b][r] * W[c][r]
                if total != 1:
                    return False
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if isinstance(val, float):
                    if not val.is_integer() and abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
    return True

def _build_schoolbook(n=3, m=3, p=8):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_fixed_UV(r, n=3, m=3, p=8, deadline=None, num_attempts=200):
    """Try to find a rank-r decomposition by fixing U,V and solving for W."""
    if deadline is not None and time.time() > deadline:
        return None
    
    for attempt in range(num_attempts):
        if deadline is not None and time.time() > deadline:
            break
        
        np.random.seed(attempt)
        
        # Random half-integer initialization
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        for r_idx in range(r):
            U[:, r_idx] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=n * m)
            V[:, r_idx] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=m * p)
        
        # Build linear system for W
        num_constraints = n * m * p
        num_unknowns = p * n * r
        
        A = np.zeros((num_constraints, num_unknowns))
        b = np.ones(num_constraints)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b_idx = j * p + k
                    c = k * n + i
                    for r_idx in range(r):
                        A[idx, c * r + r_idx] = U[a, r_idx] * V[b_idx, r_idx]
                    idx += 1
        
        # Solve least squares
        try:
            W_flat, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        W = W_flat.reshape(p * n, r)
        W_rounded = np.round(W * 2) / 2.0
        
        # Check error
        err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b_idx = j * p + k
                    c = k * n + i
                    total = 0
                    for r_idx in range(r):
                        total += U[a, r_idx] * V[b_idx, r_idx] * W_rounded[c, r_idx]
                    err += (total - 1) ** 2
        
        if err < 1e-10:
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W_rounded]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    n, m, p = 3, 3, 8
    
    # Start with schoolbook
    result = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    lower_bound = max(n * p, m * p, n * m)
    
    for r in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        res = _try_rank_fixed_UV(r, n, m, p, deadline)
        if res is not None:
            if _validate(res["U"], res["V"], res["W"], n, m, p):
                best_rank = r
                result = res
    
    return result

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    valid = _validate(result["U"], result["V"], result["W"])
    print(f"Valid: {valid}")
