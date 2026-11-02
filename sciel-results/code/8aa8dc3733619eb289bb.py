"""Construct a rank-minimal tensor decomposition for 3x3 * 3x4 matrix multiplication using Strassen blocks + rank reduction."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _verify(n, m, p, U, V, W, tol=1e-9):
    """Verify tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k)"""
    r = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(Ua[i*m+j, rr] * Va[j*p+k, rr] * Wa[k*n+i, rr] for rr in range(r))
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_fast(n, m, p, Ua, Va, Wa, tol=1e-9):
    """Vectorized verification using einsum."""
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    r = Ua.shape[1]
    # Reshape to 3D
    U3 = Ua.reshape(n, m, r)
    V3 = Va.reshape(m, p, r)
    W3 = Wa.reshape(p, n, r)
    # Compute tensor product
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=tol)

def _build_rank34_strassen():
    """Build rank-34 decomposition: 2 Strassen blocks (2x2x2) + schoolbook for remaining parts.
    
    Strategy:
    - Split the 3x3x4 multiplication into:
      * Top-left 2x2x2 block (i=0,1; j=0,1; k=0,1): Strassen rank 7
      * Top-right 2x2x2 block (i=0,1; j=0,1; k=2,3): Strassen rank 7
      * Bottom row (i=2; j=0,1,2; k=0,1,2,3): schoolbook rank 12
      * Right column (i=0,1; j=2; k=0,1,2,3): schoolbook rank 8
    Total: 7 + 7 + 12 + 8 = 34
    """
    n, m, p = 3, 3, 4
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a standard multiplication term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1

    # Helper to add a Strassen block for 2x2x2
    # local_i, local_j, local_k correspond to offsets in global i, j, k
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # 7 Strassen terms for 2x2x2 matrix multiplication
        
        # Term 1: M1 = (A11+A22) * (B11+B22)
        # A: (i_off, j_off) + (i_off+1, j_off+1)
        # B: (j_off, k_off) + (j_off+1, k_off+1)
        # C: C11 += M1, C22 += M1
        U[i_off * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = 1       # C11
        W[(k_off+1) * n + (i_off+1)][col] = 1 # C22
        col += 1

        # Term 2: M2 = (A21+A22) * B11
        # A: (i_off+1, j_off) + (i_off+1, j_off+1)
        # B: (j_off, k_off)
        # C: C21 += M2, C22 -= M2
        U[(i_off+1) * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        W[k_off * n + (i_off+1)][col] = 1   # C21
        W[(k_off+1) * n + (i_off+1)][col] = -1 # C22
        col += 1

        # Term 3: M3 = A11 * (B12-B22)
        # A: (i_off, j_off)
        # B: (j_off, k_off+1) - (j_off+1, k_off+1)
        # C: C12 += M3, C22 += M3
        U[i_off * m + j_off][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = -1
        W[(k_off+1) * n + i_off][col] = 1   # C12
        W[(k_off+1) * n + (i_off+1)][col] = 1 # C22
        col += 1

        # Term 4: M4 = A22 * (B21-B11)
        # A: (i_off+1, j_off+1)
        # B: (j_off+1, k_off) - (j_off, k_off)
        # C: C11 += M4, C21 += M4
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[(j_off+1) * p + k_off][col] = 1
        V[j_off * p + k_off][col] = -1
        W[k_off * n + i_off][col] = 1       # C11
        W[k_off * n + (i_off+1)][col] = 1   # C21
        col += 1

        # Term 5: M5 = (A11+A12) * B22
        # A: (i_off, j_off) + (i_off, j_off+1)
        # B: (j_off+1, k_off+1)
        # C: C11 -= M5, C12 += M5
        U[i_off * m + j_off][col] = 1
        U[i_off * m + (j_off+1)][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = -1     # C11
        W[(k_off+1) * n + i_off][col] = 1  # C12
        col += 1

        # Term 6: M6 = (A21-A11) * (B11+B12)
        # A: (i_off+1, j_off) - (i_off, j_off)
        # B: (j_off, k_off) + (j_off, k_off+1)
        # C: C22 += M6
        U[(i_off+1) * m + j_off][col] = 1
        U[i_off * m + j_off][col] = -1
        V[j_off * p + k_off][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1 # C22
        col += 1

        # Term 7: M7 = (A12-A22) * (B21+B22)
        # A: (i_off, j_off+1) - (i_off+1, j_off+1)
        # B: (j_off+1, k_off) + (j_off+1, k_off+1)
        # C: C11 += M7
        U[i_off * m + (j_off+1)][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = -1
        V[(j_off+1) * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = 1       # C11
        col += 1

    # 1. Top-left 2x2x2 block (i=0,1; j=0,1; k=0,1): Strassen rank 7
    add_strassen(0, 0, 0)
    
    # 2. Top-right 2x2x2 block (i=0,1; j=0,1; k=2,3): Strassen rank 7
    add_strassen(0, 0, 2)
    
    # 3. Bottom row (i=2; j=0,1,2; k=0,1,2,3): schoolbook rank 12
    for j in range(3):
        for k in range(4):
            add_term(2 * m + j, j * p + k, k * n + 2)
    
    # 4. Right column (i=0,1; j=2; k=0,1,2,3): schoolbook rank 8
    for i in range(2):
        for k in range(4):
            add_term(i * m + 2, 2 * p + k, k * n + i)
    
    return {"U": U, "V": V, "W": W}

def _rank_reduce(n, m, p, U_list, V_list, W_list, target_rank, time_budget, seed=42):
    """Try to reduce rank via random basis transformations and column elimination.
    
    Strategy: 
    1. Start with rank-34 decomposition
    2. Apply random invertible transformations to U, V, W
    3. Try to identify near-zero columns and eliminate them
    4. Use greedy elimination: try removing each column and check if remaining still spans the tensor
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    
    U0 = np.array(U_list, dtype=float)
    V0 = np.array(V_list, dtype=float)
    W0 = np.array(W_list, dtype=float)
    r0 = U0.shape[1]
    
    best_rank = r0
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    def try_elimination(U, V, W, time_limit):
        """Greedy: try removing columns one by one, check if tensor is still spanned."""
        r = U.shape[1]
        active = list(range(r))
        
        # Compute the full tensor first
        n, m, p = 3, 3, 4
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        
        def compute_tensor(indices):
            idx = np.array(indices)
            return np.einsum('ijr,jkr,kir->ijk', U3[:, :, idx], V3[:, :, idx], W3[:, :, idx])
        
        T_full = compute_tensor(active)
        
        # Try removing each column
        removed = set()
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for col_idx in list(active):
                if col_idx in removed:
                    continue
                remaining = [c for c in active if c != col_idx]
                if len(remaining) < target_rank:
                    break
                T_rem = compute_tensor(remaining)
                if np.allclose(T_rem, T_full, atol=1e-8):
                    active = remaining
                    removed.add(col_idx)
                    changed = True
                    break
        
        return active
    
    # Try several random transformations
    n_transforms = 50
    for t in range(n_transforms):
        if time.time() > time_budget:
            break
        
        # Random invertible transformations
        Pu = np_rng.random((r0, r0)) + np.eye(r0) * 2
        Pv = np_rng.random((r0, r0)) + np.eye(r0) * 2
        Pw = np_rng.random((r0, r0)) + np.eye(r0) * 2
        
        U_t = U0 @ Pu
        V_t = V0 @ Pv
        W_t = W0 @ Pw
        
        # Verify it's still valid
        if not _verify_fast(n, m, p, U_t, V_t, W_t, tol=1e-7):
            continue
        
        # Try elimination
        active = try_elimination(U_t, V_t, W_t, time_budget - 0.5)
        
        if len(active) < best_rank:
            best_rank = len(active)
            idx = np.array(active)
            best_U = U_t[:, idx]
            best_V = V_t[:, idx]
            best_W = W_t[:, idx]
            
            if best_rank <= target_rank:
                break
    
    # Convert back to lists
    def to_list(arr):
        return [[float(v) for v in row] for row in arr]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W), "rank": best_rank}

def _build_schoolbook():
    """Build schoolbook rank-36 decomposition."""
    n, m, p = 3, 3, 4
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

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = 3, 3, 4
    
    # First get the rank-34 baseline
    try:
        baseline = _build_rank34_strassen()
        if _verify(n, m, p, baseline["U"], baseline["V"], baseline["W"]):
            best_result = baseline
            best_rank = 34
        else:
            best_result = _build_schoolbook()
            best_rank = 36
    except Exception:
        best_result = _build_schoolbook()
        best_rank = 36
    
    # Try rank reduction if we have time
    if time.time() < deadline - 5 and best_rank > 24:
        try:
            time_budget = deadline - 2
            reduced = _rank_reduce(n, m, p, 
                                    best_result["U"], best_result["V"], best_result["W"],
                                    target_rank=24, time_budget=time_budget)
            if reduced["rank"] < best_rank:
                # Verify the reduced result
                if _verify(n, m, p, reduced["U"], reduced["V"], reduced["W"]):
                    best_result = reduced
                    best_rank = reduced["rank"]
        except Exception:
            pass
    
    # Final verification
    if not _verify(n, m, p, best_result["U"], best_result["V"], best_result["W"]):
        return _build_schoolbook()
    
    return best_result
# EVOLVE-BLOCK-END
