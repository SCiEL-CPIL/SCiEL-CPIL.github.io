import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    total_entries = n * m * p  # 18
    
    start_time = time.time()
    deadline = 140  # safety margin
    
    # Precompute the tensor entries: T[u, v, w] = 1 if (i,j,k) valid
    # u = i*m + j, v = j*p + k, w = k*n + i
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                tensor_entries.append((u, v, w))
    
    # Schoolbook decomposition (rank 18)
    rank_school = total_entries
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    for col, (u, v, w) in enumerate(tensor_entries):
        U_school[u, col] = 1
        V_school[v, col] = 1
        W_school[w, col] = 1
    
    def verify_solution(U, V, W, tol=1e-9):
        """Check if U, V, W give exact decomposition with half-integer entries."""
        # Check half-integer constraint
        if not np.allclose(U * 2, np.round(U * 2), atol=tol):
            return False
        if not np.allclose(V * 2, np.round(V * 2), atol=tol):
            return False
        if not np.allclose(W * 2, np.round(W * 2), atol=tol):
            return False
        
        # Check exact decomposition
        for (u, v, w) in tensor_entries:
            s = 0.0
            for r in range(U.shape[1]):
                s += U[u, r] * V[v, r] * W[w, r]
            if not np.isclose(s, 1.0, atol=tol):
                return False
        return True
    
    def als_decompose(U_init, V_init, W_init, R, max_iters=300, tol=1e-12):
        """Alternating least squares for tensor decomposition."""
        U = U_init.copy().astype(np.float64)
        V = V_init.copy().astype(np.float64)
        W = W_init.copy().astype(np.float64)
        
        prev_err = np.inf
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            # Update U rows
            for u in range(nm):
                # Find all (v, w) such that there exists j,k with u=i*m+j, v=j*p+k, w=k*n+i
                B_rows = []
                for (uu, vv, ww) in tensor_entries:
                    if uu == u:
                        B_rows.append(V[vv, :] * W[ww, :])
                if B_rows:
                    B = np.array(B_rows)
                    target = np.ones(len(B_rows))
                    U_u, _, _, _ = np.linalg.lstsq(B, target, rcond=None)
                    U[u, :] = U_u
            
            # Update V rows
            for v in range(mp):
                C_rows = []
                for (uu, vv, ww) in tensor_entries:
                    if vv == v:
                        C_rows.append(U[uu, :] * W[ww, :])
                if C_rows:
                    C = np.array(C_rows)
                    target = np.ones(len(C_rows))
                    V_v, _, _, _ = np.linalg.lstsq(C, target, rcond=None)
                    V[v, :] = V_v
            
            # Update W rows
            for w in range(pn):
                D_rows = []
                for (uu, vv, ww) in tensor_entries:
                    if ww == w:
                        D_rows.append(U[uu, :] * V[vv, :])
                if D_rows:
                    D = np.array(D_rows)
                    target = np.ones(len(D_rows))
                    W_w, _, _, _ = np.linalg.lstsq(D, target, rcond=None)
                    W[w, :] = W_w
            
            # Calculate error
            err = 0.0
            for (u, v, w) in tensor_entries:
                s = 0.0
                for r in range(R):
                    s += U[u, r] * V[v, r] * W[w, r]
                err += (s - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        U, V, W = best_U, best_V, best_W
        
        # Round to half-integers and verify
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        
        if verify_solution(U_round, V_round, W_round):
            return U_round, V_round, W_round
            
        # Try slightly different rounding (snap to nearest 0.5)
        U_snap = np.floor(U * 2 + 0.5) / 2
        V_snap = np.floor(V * 2 + 0.5) / 2
        W_snap = np.floor(W * 2 + 0.5) / 2
        if verify_solution(U_snap, V_snap, W_snap):
            return U_snap, V_snap, W_snap
            
        return None
    
    def try_rank(R, num_restarts=20, time_limit=10):
        """Try to find a rank-R decomposition with multiple random restarts."""
        start = time.time()
        for restart in range(num_restarts):
            if time.time() - start > time_limit or time.time() > deadline:
                break
            np.random.seed(restart * 1000 + R * 7)
            U_init = np.random.randn(nm, R) * 0.5
            V_init = np.random.randn(mp, R) * 0.5
            W_init = np.random.randn(pn, R) * 0.5
            res = als_decompose(U_init, V_init, W_init, R, max_iters=200)
            if res is not None:
                return res
        return None
    
    # Strategy: Try ranks from low to high
    # Known bounds: rank(2x3x3) is between 11 and 18
    # The 2x2x3 tensor has rank 11, and 1x3x3 has rank 3
    # So 2x3x3 rank is at most 11+3=14, but could be lower
    
    # Try rank 14 first (upper bound from sub-tensor argument)
    for target_rank in [14, 13, 12, 11]:
        if time.time() > deadline - 5:
            break
        res = try_rank(target_rank, num_restarts=30, time_limit=20)
        if res is not None:
            U, V, W = res
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Try rank 15 and 16
    for target_rank in [15, 16]:
        if time.time() > deadline - 5:
            break
        res = try_rank(target_rank, num_restarts=20, time_limit=15)
        if res is not None:
            U, V, W = res
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Try rank 17
    if time.time() < deadline - 5:
        res = try_rank(17, num_restarts=10, time_limit=10)
        if res is not None:
            U, V, W = res
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
