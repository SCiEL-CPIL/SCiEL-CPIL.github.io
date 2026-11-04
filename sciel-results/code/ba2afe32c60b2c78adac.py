"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _exact_check(U, V, W, n, m, p):
    """Exact verification using Fraction arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    uc = Fraction(str(U[i*m+j][c])) if isinstance(U[i*m+j][c], (int, float)) else U[i*m+j][c]
                    vc = Fraction(str(V[j*p+k][c])) if isinstance(V[j*p+k][c], (int, float)) else V[j*p+k][c]
                    wc = Fraction(str(W[k*n+i][c])) if isinstance(W[k*n+i][c], (int, float)) else W[k*n+i][c]
                    s += uc * vc * wc
                if s != 1:
                    return False
    return True


def _fast_check(U, V, W, n, m, p, tol=1e-9):
    """Fast approximate check using numpy."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _build_strassen_4x4x4():
    """Build the 49-term Strassen decomposition for 4x4x4."""
    # Strassen 2x2 terms: (u_vec, v_vec, w_vec) where each is length-4
    S2 = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
        ([0,0,0,1], [0,0,1,-1], [-1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [1,1,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [-1,0,0,-1]),
    ]
    terms = []
    for a in range(7):
        for b in range(7):
            ua, va, wa = S2[a]
            ub, vb, wb = S2[b]
            U4 = [0]*16; V4 = [0]*16; W4 = [0]*16
            for I in range(4):
                for J in range(4):
                    U4[I*4+J] = ua[(I>>1)*2+(J>>1)] * ub[(I&1)*2+(J&1)]
            for J in range(4):
                for K in range(4):
                    V4[J*4+K] = va[(J>>1)*2+(K>>1)] * vb[(J&1)*2+(K&1)]
            for K in range(4):
                for I in range(4):
                    W4[K*4+I] = wa[(K>>1)*2+(I>>1)] * wb[(K&1)*2+(I&1)]
            terms.append((U4, V4, W4))
    return terms


def _make_strassen_4x4x7():
    """Rank-98 decomposition: 49 terms for cols 0-3, 49 for cols 4-6."""
    terms44 = _build_strassen_4x4x4()
    U_list, V_list, W_list = [], [], []
    for U4, V4, W4 in terms44:
        # Cols 0-3
        V1 = [0]*28; W1 = [0]*28
        for J in range(4):
            for K in range(4):
                V1[J*7+K] = V4[J*4+K]
        for K in range(4):
            for I in range(4):
                W1[K*4+I] = W4[K*4+I]
        U_list.append(list(U4)); V_list.append(V1); W_list.append(W1)
        # Cols 4-6
        V2 = [0]*28; W2 = [0]*28
        for J in range(4):
            for K in range(3):
                V2[J*7+(K+4)] = V4[J*4+K]
        for K in range(3):
            for I in range(4):
                W2[(K+4)*4+I] = W4[K*4+I]
        U_list.append(list(U4)); V_list.append(V2); W_list.append(W2)
    
    rank = len(U_list)
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(28)]
    W = [[0]*rank for _ in range(28)]
    for c in range(rank):
        for i in range(16): U[i][c] = U_list[c][i]
        for i in range(28): V[i][c] = V_list[c][i]
        for i in range(28): W[i][c] = W_list[c][i]
    return U, V, W


def _make_schoolbook(n, m, p):
    """Trivial rank-n*m*p decomposition."""
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W


def _greedy_peel(U, V, W, n, m, p, deadline):
    """Greedy tensor peeling: iteratively remove terms that are linearly dependent
    in the mode-1 flattening of the residual tensor.
    
    The multiplication tensor T has T[i,j,k]=1 iff the (i,j,k) entry of C=A*B is
    determined. We represent the residual as a 3D array and try to peel off
    rank-1 components that are already spanned by remaining terms.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    keep = [True] * rank
    best_rank = rank
    
    # Compute the full tensor for reference
    T_target = np.ones((n, m, p))
    
    def compute_residual(keep_mask):
        """Compute sum of rank-1 tensors for kept terms."""
        res = np.zeros((n, m, p))
        for c in range(rank):
            if not keep_mask[c]:
                continue
            for i in range(n):
                for j in range(m):
                    uc = U_arr[i*m+j, c]
                    if uc == 0:
                        continue
                    for k in range(p):
                        vc = V_arr[j*p+k, c]
                        if vc == 0:
                            continue
                        wc = W_arr[k*n+i, c]
                        if wc == 0:
                            continue
                        res[i,j,k] += uc * vc * wc
        return res
    
    # Try removing each term, starting with most "redundant" (lowest contribution)
    # Score each term by how much it contributes to the tensor
    contributions = []
    for c in range(rank):
        contrib = 0.0
        for i in range(n):
            for j in range(m):
                uc = U_arr[i*m+j, c]
                if uc == 0: continue
                for k in range(p):
                    vc = V_arr[j*p+k, c]
                    if vc == 0: continue
                    wc = W_arr[k*n+i, c]
                    if wc == 0: continue
                    contrib += abs(uc * vc * wc)
        contributions.append(contrib)
    
    # Sort by contribution (smallest first - most likely redundant)
    order = sorted(range(rank), key=lambda c: contributions[c])
    
    for idx in order:
        if time.time() > deadline - 40:
            break
        if not keep[idx]:
            continue
        keep[idx] = False
        # Quick check: recompute residual and verify
        residual = compute_residual(keep)
        if np.allclose(residual, T_target, atol=1e-9):
            best_rank = sum(keep)
        else:
            keep[idx] = True
    
    if best_rank < rank:
        keep_indices = [c for c in range(rank) if keep[c]]
        new_U = U_arr[:, keep_indices].tolist()
        new_V = V_arr[:, keep_indices].tolist()
        new_W = W_arr[:, keep_indices].tolist()
        return new_U, new_V, new_W, best_rank
    return U, V, W, rank


def _structured_search(n, m, p, deadline):
    """Try structured decompositions with lower rank using block structure.
    
    Key idea: A 4x4x7 multiplication can be decomposed by splitting the 7 columns
    into blocks and using efficient algorithms for each block.
    
    Strategy: Use the fact that 4x4 matrix multiplication has rank at most 49 (Strassen),
    and for the 3-column block, we can potentially do better.
    
    We try:
    1. Rank 49+49=98 (baseline Strassen split)
    2. Try to find a rank-49 decomposition for the 4x4x4 part that also helps the 3-col part
    3. Use bilinear algorithms with shared structure
    """
    # Baseline: Strassen 4x4x4 split
    U, V, W = _make_strassen_4x4x7()
    if not _fast_check(U, V, W, n, m, p):
        U, V, W = _make_schoolbook(n, m, p)
        return U, V, W
    
    # Try greedy peeling
    U, V, W, r = _greedy_peel(U, V, W, n, m, p, deadline)
    
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # First try structured search
    U, V, W = _structured_search(n, m, p, deadline)
    
    # Final verification with exact arithmetic
    if not _exact_check(U, V, W, n, m, p):
        # Fall back to schoolbook
        U, V, W = _make_schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
