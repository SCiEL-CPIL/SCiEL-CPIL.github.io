import time
import numpy as np
from itertools import product

_START = time.time()
DEADLINE_S = 140.0

def _time_left():
    return DEADLINE_S - (time.time() - _START)

def _strassen_2x2_2():
    """U, V, W (lists) for 2x2 * 2x2 with rank 7 (Strassen)."""
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
    U[2][1]=1; U[3][1]=1; V[0][1]=1; W[1][1]=1; W[3][1]=-1
    U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[2][2]=1; W[3][2]=1
    U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[1][3]=1
    U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[2][4]=1
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_np(Ua, Va, Wa, n, m, p, tol=1e-6):
    """Vectorized verification using numpy."""
    rank = Ua.shape[1]
    for i in range(n):
        for j in range(m):
            u_row = Ua[i*m + j]
            for k in range(p):
                v_row = Va[j*p + k]
                w_row = Wa[k*n + i]
                # sum_c u_c * v_c * w_c
                s = np.dot(u_row, v_row * w_row)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _build_strassen_embedding(n, m, p, strassen_blocks):
    covered = set()
    for (i0, j0, k0) in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    n_school = n*m*p - len(covered)
    rank = 7 * len(strassen_blocks) + n_school
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for (i0, j0, k0) in strassen_blocks:
        sU, sV, sW = _strassen_2x2_2()
        for c in range(7):
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj)][col+c] = sU[di*2+dj][c]
            for dj in range(2):
                for dk in range(2):
                    V[(j0+dj)*p + (k0+dk)][col+c] = sV[dj*2+dk][c]
            for dk in range(2):
                for di in range(2):
                    W[(k0+dk)*n + (i0+di)][col+c] = sW[dk*2+di][c]
        col += 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    return U, V, W

def _try_reduce_als(Ua, Va, Wa, n, m, p, target_rank, time_limit=8.0, seed=0):
    """Try to find a rank-`target_rank` decomposition using alternating least squares.
    Inputs are numpy arrays of the original rank. Returns (U,V,W) as numpy arrays if valid, else None."""
    if _time_left() < time_limit + 2:
        return None
    start = time.time()
    nm, mp, pn = n*m, m*p, p*n
    
    rng = np.random.default_rng(seed)
    
    # Initialize: pick target_rank columns from the original decomposition.
    # To improve success rate, we can try different subsets or random perturbations.
    # For now, we use the first target_rank columns as a baseline, but we might
    # want to try a few random subsets if time permits.
    
    # Strategy: Use the first target_rank columns.
    Uo = Ua[:, :target_rank].copy()
    Vo = Va[:, :target_rank].copy()
    Wo = Wa[:, :target_rank].copy()
    
    # ALS Loop
    for it in range(200):
        if (time.time() - start) > time_limit - 0.2:
            break
        
        # Update U
        # For each (i, j), solve A x = b where A is p x target_rank
        for i in range(n):
            for j in range(m):
                # A[k, c] = V[j, k, c] * W[k, i, c]
                # V slice for j is Va[j*p : (j+1)*p, :] -> shape (p, rank)
                # W slice for i is Wa[i : p*n : n, :] ? No, W index is k*n + i.
                # So for fixed i, we need W[k*n+i] for k=0..p-1.
                # This is a strided view.
                V_slice = Va[j*p:(j+1)*p, :] # (p, rank)
                # W_slice needs to be (p, rank) where row k is W[k*n+i]
                # Indices: 0*n+i, 1*n+i, ..., (p-1)*n+i
                w_indices = np.arange(p) * n + i
                W_slice = Wa[w_indices, :] # (p, rank)
                
                A = V_slice * W_slice # (p, rank)
                b = np.ones(p)
                try:
                    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                    Uo[i*m + j] = sol
                except np.linalg.LinAlgError:
                    # Fallback to simple projection or keep old value
                    pass
        
        # Update V
        # For each (j, k), solve A x = b where A is n x target_rank
        for j in range(m):
            for k in range(p):
                # A[i, c] = U[i, j, c] * W[k, i, c]
                # U slice for j is U[i*m+j] for i=0..n-1.
                # Indices: 0*m+j, 1*m+j, ..., (n-1)*m+j
                u_indices = np.arange(n) * m + j
                U_slice = Ua[u_indices, :] # (n, rank)
                # Wait, we are updating V, so we use current Uo and Wo.
                U_slice = Uo[u_indices, :]
                
                # W slice for k is W[k*n+i] for i=0..n-1.
                # Indices: k*n+0, k*n+1, ..., k*n+n-1
                w_indices = np.arange(n) + k * n
                W_slice = Wo[w_indices, :] # (n, rank)
                
                A = U_slice * W_slice # (n, rank)
                b = np.ones(n)
                try:
                    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                    Vo[j*p + k] = sol
                except np.linalg.LinAlgError:
                    pass
        
        # Update W
        # For each (k, i), solve A x = b where A is m x target_rank
        for k in range(p):
            for i in range(n):
                # A[j, c] = U[i, j, c] * V[j, k, c]
                # U slice for i is U[i*m+j] for j=0..m-1.
                # Indices: i*m+0, i*m+1, ..., i*m+m-1
                u_indices = np.arange(m) + i * m
                U_slice = Uo[u_indices, :] # (m, rank)
                
                # V slice for k is V[j*p+k] for j=0..m-1.
                # Indices: 0*p+k, 1*p+k, ..., (m-1)*p+k
                v_indices = np.arange(m) * p + k
                V_slice = Vo[v_indices, :] # (m, rank)
                
                A = U_slice * V_slice # (m, rank)
                b = np.ones(m)
                try:
                    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                    Wo[k*n + i] = sol
                except np.linalg.LinAlgError:
                    pass
        
        # Check error
        err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = Uo[i*m + j]
                for k in range(p):
                    v_row = Vo[j*p + k]
                    w_row = Wo[k*n + i]
                    val = np.dot(u_row, v_row * w_row)
                    err += (val - 1.0)**2
        if err < 1e-12:
            return Uo, Vo, Wo
            
    return None

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n

    def schoolbook():
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

    # Deterministic rank-68 construction: 4 disjoint 2x2x2 Strassen blocks
    # Non-overlapping blocks in i=0,1; j=0,1; k=0..7
    strassen_blocks_valid = [(0,0,0), (0,0,2), (0,0,4), (0,0,6)]
    U, V, W = _build_strassen_embedding(n, m, p, strassen_blocks_valid)
    
    # Check validity
    if _verify(U, V, W, n, m, p):
        best_rank = len(U[0])
        best = (U, V, W)
    else:
        # Fallback to schoolbook
        best = schoolbook()
        best_rank = n*m*p
        best = (best["U"], best["V"], best["W"])

    # Convert best to numpy for ALS
    Ua = np.array(best[0], dtype=float)
    Va = np.array(best[1], dtype=float)
    Wa = np.array(best[2], dtype=float)
    
    # Try to reduce rank
    found_reduction = False
    for target in range(best_rank - 1, best_rank - 11, -1):
        if _time_left() < 20.0:
            break
        
        # Try a few seeds
        for seed in range(5):
            if _time_left() < 15.0:
                break
            res = _try_reduce_als(Ua, Va, Wa, n, m, p, target, time_limit=8.0, seed=seed)
            if res is not None:
                rU, rV, rW = res
                # Verify the reduced solution
                if _verify_np(rU, rV, rW, n, m, p):
                    best = (rU.tolist(), rV.tolist(), rW.tolist())
                    best_rank = target
                    found_reduction = True
                    break
        
        if found_reduction:
            # Update numpy arrays to the new best for subsequent reductions?
            # The ALS in _try_reduce_als initializes from the input arrays.
            # If we found a rank R-1 solution, we should use it as the base for R-2.
            Ua = np.array(best[0], dtype=float)
            Va = np.array(best[1], dtype=float)
            Wa = np.array(best[2], dtype=float)
            # Reset flag to allow further reduction in next loop iteration?
            # The loop structure `for target in ...` will decrement target.
            # But `found_reduction` is currently used to break the seed loop.
            # We need to continue the outer loop.
            # The variable `found_reduction` is local to the iteration? No, it's defined outside.
            # If we find a reduction, we update best. The next `target` is lower.
            # We should probably reset `found_reduction` or just let the loop continue.
            # The `break` only exits the seed loop. The outer loop continues.
            pass

    # Final safety check
    if _time_left() < 5.0:
        pass

    return {"U": best[0], "V": best[1], "W": best[2]}

DIMENSIONS = [3, 3, 8]
