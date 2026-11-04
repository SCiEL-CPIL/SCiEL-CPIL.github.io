"""
Rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication.

Strategy:
1. Build the 10x8x20 multiplication tensor.
2. Use greedy rank-1 extraction with exact rational arithmetic (fractions) to
   find a minimal-rank decomposition.
3. The key insight: for matrix multiplication M_{n,m,p}, the tensor has a
   well-known structure. For 2x4*4x5, we can exploit the bilinear structure.

The tensor T[a,b,c] = 1 iff a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k.

For n=2, m=4, p=5:
  a = i*4 + j  (i in {0,1}, j in {0,1,2,3})
  b = j*5 + k  (j in {0,1,2,3}, k in {0,1,2,3,4})
  c = k*2 + i  (k in {0,1,2,3,4}, i in {0,1})

So the 40 nonzero entries are:
  (i*4+j, j*5+k, k*2+i) = 1

We need to find rank-r decomposition: T = sum_{s=1}^r U_s ⊗ V_s ⊗ W_s
where U_s in R^8, V_s in R^20, W_s in R^10.

The schoolbook rank is 40. We try to do better.

Key structural observation: The tensor is a sum of 40 rank-1 terms, but many
can be combined. For example, if we fix i, the terms for that i form a
2x4x5 sub-tensor (well, actually a subset).

Actually, let me think more carefully. The tensor can be written as:
T = sum_{i=0}^{1} sum_{j=0}^{3} sum_{k=0}^{4} e_{i*4+j} ⊗ e_{j*5+k} ⊗ e_{k*2+i}

This is the standard schoolbook decomposition with rank 40.

To reduce rank, we need to find linear dependencies. The tensor rank of
M_{n,m,p} is known for small cases. For M_{2,4,5}:
- The trivial bound is 40.
- Strassen-type algorithms can reduce this.

Let me try a different approach: use the fact that the tensor is a "matrix
multiplication tensor" and apply known lower bounds and constructions.

For M_{2,4,5}, we can view it as:
C = A * B where A is 2x4, B is 4x5, C is 2x5.

Split A into [A1 A2] where A1, A2 are 2x2.
Split B into [B1; B2] where B1, B2 are 2x5.

Then C = A1*B1 + A2*B2.

For 2x2 * 2x5: we can use rank 8 (schoolbook) or potentially less.
Actually, 2x2*2x5 has tensor rank... let me think.

M_{2,2,5}: schoolbook rank is 20. But we can do better.
Actually, for M_{2,2,p}, the rank is known to be at most 2p - 2 for p >= 2?
No, that's not right either.

Let me just try the greedy approach with rational arithmetic. If we can find
a rank-32 or lower decomposition, that's an improvement.

Actually, let me try a smarter approach. The 2x4*4x5 problem can be seen as:
- Two 2x2*2x5 multiplications (for the two column-blocks of A)
- Or four 2x1*1x5 multiplications (for each column of A)

The best known approach for small matrix multiplication uses Strassen-like
recursion. Let me try to construct an explicit low-rank decomposition.

For 2x2 * 2x2, Strassen gives rank 7.
For 2x2 * 2x5, we can use:
  C = A*B where A is 2x2, B is 2x5, C is 2x5.
  Split B into [B1 B2 B3 B4 B5] (five 2x1 columns).
  C = A*B1 + A*B2 + ... + A*B5
  Each A*Bj is 2x2 * 2x1 = 2x1, which has rank 2 (schoolbook).
  So total rank = 5*2 = 10? No, that's not how it works.

Actually, 2x2 * 2x1: A is 2x2, B is 2x1, C is 2x1.
C[0] = A[0,0]*B[0] + A[0,1]*B[1]
C[1] = A[1,0]*B[0] + A[1,1]*B[1]
This is rank 4 schoolbook, but can we do better? The tensor is:
T[a,b,c] where a in {0,1,2,3} (2x2), b in {0,1} (2x1), c in {0,1} (2x1)
T[0,0,0]=1, T[1,1,0]=1, T[2,0,1]=1, T[3,1,1]=1
This is actually two independent 1x1*1x1 multiplications (for row 0 and row 1 of C),
each with rank 2. So total rank is 4. Can we do rank 3? Probably not.

Let me just implement the greedy rational approach and see what rank we get.
"""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START

def _schoolbook_decomposition(n, m, p):
    """Standard O(nmp) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if the tensor decomposition is exact."""
    nm = n * m
    mp = m * p
    np_ = n * p
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)

    # Check the n*m*p "one" positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a = i * m + j
                idx_b = j * p + k
                idx_c = k * n + i
                val = np.sum(U[idx_a, :] * V[idx_b, :] * W[idx_c, :])
                if abs(val - 1.0) > tol:
                    return False

    # Check that all other positions are zero
    one_positions = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                one_positions.add((i * m + j, j * p + k, k * n + i))

    for a in range(nm):
        for b in range(mp):
            for c in range(np_):
                if (a, b, c) in one_positions:
                    continue
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val) > tol:
                    return False
    return True


def _check_half_integer(U, V, W, tol=1e-9):
    """Check if entries are half-integers."""
    for mat in [U, V, W]:
        mat = np.asarray(mat, dtype=float)
        if not np.allclose(mat * 2, np.round(mat * 2), atol=tol):
            return False
    return True


def _snap_to_half_integer(U, V, W):
    """Snap entries to nearest half-integer."""
    U_s = np.round(np.asarray(U, dtype=float) * 2) / 2.0
    V_s = np.round(np.asarray(V, dtype=float) * 2) / 2.0
    W_s = np.round(np.asarray(W, dtype=float) * 2) / 2.0
    return U_s, V_s, W_s


def _build_target_tensor(n, m, p):
    """Build the target matmul tensor T[a,b,c]."""
    nm = n * m
    mp = m * p
    np_ = n * p
    T = np.zeros((nm, mp, np_), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _greedy_rational_decomposition(n, m, p, max_rank=40, time_budget=None, start_time=None):
    """
    Greedy rank-1 extraction with exact rational arithmetic.
    
    At each step, find the best rank-1 approximation of the residual tensor,
    subtract it, and repeat until the residual is zero or we hit max_rank.
    """
    if start_time is None:
        start_time = time.time()
    
    nm = n * m
    mp = m * p
    np_ = n * p
    
    # Build tensor as Fraction for exact arithmetic
    T = [[ [Fraction(0) for _ in range(np_)] for _ in range(mp)] for _ in range(nm)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a][b][c] = Fraction(1)
    
    U_cols = []  # list of vectors in R^nm
    V_cols = []  # list of vectors in R^mp
    W_cols = []  # list of vectors in R^np_
    
    for rank_step in range(max_rank):
        if time_budget is not None and time.time() > time_budget:
            break
        
        # Check if residual is zero
        residual_norm = 0
        for a in range(nm):
            for b in range(mp):
                for c in range(np_):
                    if T[a][b][c] != 0:
                        residual_norm += 1
                        break
                else:
                    continue
                break
            else:
                continue
            break
        
        if residual_norm == 0:
            break
        
        # Find best rank-1 approximation
        # We want to find u in R^nm, v in R^mp, w in R^np_ such that
        # sum_{a,b,c} (T[a][b][c] - u[a]*v[b]*w[c])^2 is minimized
        # 
        # Greedy approach: try all possible "patterns" based on the support
        # of the residual tensor.
        
        # Get support of residual
        support = []
        for a in range(nm):
            for b in range(mp):
                for c in range(np_):
                    if T[a][b][c] != 0:
                        support.append((a, b, c, T[a][b][c]))
        
        if not support:
            break
        
        # Try to find a good rank-1 approximation
        # Strategy: pick a "pivot" entry and try to match it
        # Then solve for the remaining entries
        
        best_err = None
        best_u, best_v, best_w = None, None, None
        
        # Try each support entry as pivot
        for pivot_idx in range(min(len(support), 50)):
            pa, pb, pc, pval = support[pivot_idx]
            
            # Set u[pa] = 1, v[pb] = 1, w[pc] = pval (wlog normalization)
            # Then for each (a,b,c) in support:
            #   T[a][b][c] ≈ u[a] * v[b] * w[c]
            #   => u[a] = T[a][b][c] / (v[b] * w[c])
            
            # This is underdetermined, so we need a different approach.
            # Use ALS-like: fix two factors, solve for the third.
            
            # Initialize
            u = [Fraction(0)] * nm
            v = [Fraction(0)] * mp
            w = [Fraction(0)] * np_
            
            # Set pivot
            u[pa] = Fraction(1)
            v[pb] = Fraction(1)
            w[pc] = pval
            
            # Iterate
            converged = False
            for it in range(20):
                # Update u: for each a, u[a] = sum_{b,c: T[a][b][c]!=0} T[a][b][c] * v[b] * w[c] / (sum_{b',c'} v[b']^2 * w[c']^2)
                # Simplified: u[a] = (sum_{b,c} T[a][b][c]*v[b]*w[c]) / (sum_{b,c} v[b]^2*w[c]^2)
                denom = sum(v[b]**2 * w[c]**2 for b in range(mp) for c in range(np_))
                if denom == 0:
                    break
                for a in range(nm):
                    num = sum(T[a][b][c] * v[b] * w[c] for b in range(mp) for c in range(np_) if T[a][b][c] != 0)
                    u[a] = num / denom
                
                # Update v
                denom = sum(u[a]**2 * w[c]**2 for a in range(nm) for c in range(np_))
                if denom == 0:
                    break
                for b in range(mp):
                    num = sum(T[a][b][c] * u[a] * w[c] for a in range(nm) for c in range(np_) if T[a][b][c] != 0)
                    v[b] = num / denom
                
                # Update w
                denom = sum(u[a]**2 * v[b]**2 for a in range(nm) for b in range(mp))
                if denom == 0:
                    break
                for c in range(np_):
                    num = sum(T[a][b][c] * u[a] * v[b] for a in range(nm) for b in range(mp) if T[a][b][c] != 0)
                    w[c] = num / denom
            
            # Compute error
            err = 0
            for a in range(nm):
                for b in range(mp):
                    for c in range(np_):
                        recon = u[a] * v[b] * w[c]
                        diff = T[a][b][c] - recon
                        err += diff * diff
            
            if best_err is None or err < best_err:
                best_err = err
                best_u, best_v, best_w = u[:], v[:], w[:]
        
        if best_u is None:
            break
        
        # Subtract the rank-1 term
        for a in range(nm):
            for b in range(mp):
                for c in range(np_):
                    T[a][b][c] -= best_u[a] * best_v[b] * best_w[c]
        
        U_cols.append(best_u)
        V_cols.append(best_v)
        W_cols.append(best_w)
        
        # Check if we've converged
        all_zero = all(T[a][b][c] == 0 for a in range(nm) for b in range(mp) for c in range(np_))
        if all_zero:
            break
    
    if not U_cols:
        return None
    
    # Convert to numpy
    rank = len(U_cols)
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((np_, rank), dtype=float)
    for s in range(rank):
        for a in range(nm):
            U[a, s] = float(U_cols[s][a])
        for b in range(mp):
            V[b, s] = float(V_cols[s][b])
        for c in range(np_):
            W[c, s] = float(W_cols[s][c])
    
    return U, V, W


def _try_explicit_constructions(n, m, p, time_budget=None, start_time=None):
    """
    Try explicit constructions based on known low-rank algorithms.
    
    For 2x4 * 4x5:
    - Schoolbook: rank 40
    - Block approach: split into 2x2*2x5 blocks
    
    For 2x2 * 2x5:
    - Schoolbook: rank 20
    - Can we do better?
    
    Actually, let's try a direct construction.
    """
    if start_time is None:
        start_time = time.time()
    
    nm = n * m  # 8
    mp = m * p  # 20
    np_ = n * p  # 10
    
    # Try rank 32: use Strassen for the 2x2*2x2 parts
    # C = A1*B1 + A2*B2 where A1,A2 are 2x2, B1,B2 are 2x5
    # For 2x2*2x5, use rank 8 (not 10) by exploiting structure
    # Actually, 2x2*2x5 schoolbook is 2*2*5 = 20
    # But we can view 2x2*2x5 as two 2x2*2x1 multiplications, each rank 4, total 8?
    # No, 2x2*2x1 has rank 4 schoolbook, but the two are independent.
    
    # Let me try rank 30, 28, etc. with a structured initialization
    for target_rank in [32, 30, 28, 26, 24]:
        if time_budget is not None and time.time() > time_budget - 5:
            break
        
        # Structured initialization
        U = np.zeros((nm, target_rank), dtype=float)
        V = np.zeros((mp, target_rank), dtype=float)
        W = np.zeros((np_, target_rank), dtype=float)
        
        col = 0
        # For each (i,j,k), we need U[i*4+j, col] * V[j*5+k, col] * W[k*2+i, col] = 1
        # and all other entries should cancel out.
        
        # Simple approach: just use schoolbook but with fewer terms by combining
        # This is hard to do explicitly, so let's use the greedy approach instead.
        pass
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook_decomposition(n, m, p)
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    best_rank = n * m * p
    
    # Try greedy rational decomposition
    try:
        result = _greedy_rational_decomposition(n, m, p, max_rank=best_rank, 
                                                  time_budget=deadline - 10, 
                                                  start_time=start_time)
        if result is not None:
            U, V, W = result
            if _check_decomposition(U, V, W, n, m, p):
                if _check_half_integer(U, V, W):
                    best_U, best_V, best_W = U, V, W
                    best_rank = U.shape[1]
                else:
                    U_s, V_s, W_s = _snap_to_half_integer(U, V, W)
                    if _check_decomposition(U_s, V_s, W_s, n, m, p):
                        best_U, best_V, best_W = U_s, V_s, W_s
                        best_rank = U_s.shape[1]
    except Exception:
        pass
    
    # Final verification
    if not _check_decomposition(best_U, best_V, best_W, n, m, p):
        # Fallback to schoolbook
        best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
