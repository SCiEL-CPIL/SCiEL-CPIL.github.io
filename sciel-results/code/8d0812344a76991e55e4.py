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


def _strassen_2x2x2():
    """
    Return the 7 rank-1 terms for 2x2 * 2x2 matrix multiplication (Strassen).
    
    A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[i, j], [k, l]]
    
    Strassen's 7 products:
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C = [[M1+M4-M5+M7, M3+M5],
         [M2+M4, M1-M2+M3+M6]]
    
    We need to express each term as U ⊗ V ⊗ W vectors.
    For 2x2*2x2: n=2, m=2, p=2
    A indices: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    B indices: e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    C indices: i=C[0,0], j=C[0,1], k=C[1,0], l=C[1,1]
    
    U vector is in R^{n*m} = R^4: [a, b, c, d]
    V vector is in R^{m*p} = R^4: [e, f, g, h]
    W vector is in R^{p*n} = R^4: [i, j, k, l] (using W[k*n+i,r] convention)
    
    For each Strassen product, we need U_s, V_s, W_s such that
    sum_s U_s[a] * V_s[b] * W_s[c] gives the correct C entry.
    """
    # Each term: (u_vec, v_vec, w_vec) where these are 4-dim vectors
    # M1 = (a+d)(e+h) contributes to i, k, l
    # M2 = (c+d)e contributes to k, l
    # M3 = a(f-h) contributes to i, j
    # M4 = d(g-e) contributes to i, k
    # M5 = (a+b)h contributes to i, j
    # M6 = (c-a)(e+f) contributes to l
    # M7 = (b-d)(g+h) contributes to i
    
    terms = []
    
    # M1 = (a+d)(e+h)
    # U: [1, 0, 0, 1] (a+d)
    # V: [1, 0, 0, 1] (e+h)
    # W: [1, 0, 1, 0] (i + k) ... wait, need to check
    # C[0,0] = M1+M4-M5+M7, C[0,1] = M3+M5, C[1,0] = M2+M4, C[1,1] = M1-M2+M3+M6
    # So M1 contributes +1 to C[0,0] and +1 to C[1,1]
    # W for C indices: [i, j, k, l] = [C[0,0], C[0,1], C[1,0], C[1,1]]
    # M1: W = [1, 0, 0, 1]
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    
    # M2 = (c+d)e
    # U: [0, 0, 1, 1] (c+d)
    # V: [1, 0, 0, 0] (e)
    # M2 contributes to C[1,0] and -C[1,1]
    # W: [0, 0, 1, -1]
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
    
    # M3 = a(f-h)
    # U: [1, 0, 0, 0] (a)
    # V: [0, 1, 0, -1] (f-h)
    # M3 contributes to C[0,1] and +C[1,1]
    # W: [0, 1, 0, 1]
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,1]))
    
    # M4 = d(g-e)
    # U: [0, 0, 0, 1] (d)
    # V: [0, 0, 1, 0] - wait, g-e means [0,0,1,0] for g and [1,0,0,0] for e
    # V: [-1, 0, 1, 0] (g-e)
    # M4 contributes to C[0,0] and +C[1,0]
    # W: [1, 0, 1, 0]
    terms.append(([0,0,0,1], [-1,0,1,0], [1,0,1,0]))
    
    # M5 = (a+b)h
    # U: [1, 1, 0, 0] (a+b)
    # V: [0, 0, 0, 1] (h)
    # M5 contributes -C[0,0] and +C[0,1]
    # W: [-1, 1, 0, 0]
    terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
    
    # M6 = (c-a)(e+f)
    # U: [-1, 0, 1, 0] (c-a)
    # V: [1, 1, 0, 0] (e+f)
    # M6 contributes +C[1,1]
    # W: [0, 0, 0, 1]
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    
    # M7 = (b-d)(g+h)
    # U: [0, 1, 0, -1] (b-d)
    # V: [0, 0, 1, 1] (g+h)
    # M7 contributes +C[0,0]
    # W: [1, 0, 0, 0]
    terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    
    return terms


def _build_strassen_based_decomposition(n, m, p):
    """
    Build a low-rank decomposition for 2x4 * 4x5 using Strassen-like blocks.
    
    Strategy: 
    - Split the 4 middle dimension into two 2-dim blocks
    - A = [A1 | A2] (each 2x2), B = [B1; B2] (each 2x5)
    - C = A1*B1 + A2*B2
    - For each 2x2 * 2x5, split B into 5 columns of 2x1
    - Each 2x2 * 2x1 can be done with rank 4 (schoolbook) but we can combine
    - Actually, let's use a different split.
    
    Alternative: View 2x4*4x5 as a single tensor and apply Strassen-like
    transformations on subsets of indices.
    
    Let me try a direct construction with explicit coefficients.
    """
    nm = n * m  # 8
    mp = m * p  # 20
    np_ = n * p  # 10
    
    # Use the block structure:
    # A rows: i=0,1; A cols: j=0,1,2,3
    # B rows: j=0,1,2,3; B cols: k=0,1,2,3,4
    # C rows: k=0,1,2,3,4; C cols: i=0,1
    # 
    # C[k,i] = sum_j A[i,j] * B[j,k]
    #
    # Split j into {0,1} and {2,3}:
    # C[k,i] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    #
    # For the first part (j in {0,1}): 2x2 * 2x5
    # For the second part (j in {2,3}): 2x2 * 2x5
    #
    # For 2x2 * 2x5: A is 2x2, B is 2x5, C is 2x5
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #
    # This is like 5 independent 2x2*2x1 problems (one per k).
    # Each 2x2*2x1: C[i] = A[i,0]*B[0] + A[i,1]*B[1], rank 4 schoolbook.
    # But we can use Strassen-like tricks.
    #
    # Actually, for 2x2*2x1, the tensor is:
    # T[a,b,c] where a in {0,1,2,3} (A indices), b in {0,1} (B indices), c in {0,1} (C indices)
    # T[0,0,0]=1, T[1,1,0]=1, T[2,0,1]=1, T[3,1,1]=1
    # This is two independent 1x1*1x1*1x1 tensors, each rank 2.
    # Total rank 4, and I believe this is minimal.
    #
    # So 2x2*2x5 has rank 5*4 = 20 schoolbook.
    # And 2x4*4x5 has rank 2*20 = 40 schoolbook.
    #
    # To do better, we need to find cross-term cancellations.
    # 
    # Let me try a different approach: use the fact that the full tensor
    # has a specific structure and find a rank-r decomposition with r < 40.
    
    # Direct construction attempt with rank 36:
    # Use 36 terms, each being a rank-1 tensor with half-integer coefficients.
    
    # Actually, let me try to find the minimal rank by solving a system of equations.
    # For rank r, we need:
    # For each (i,j,k): sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1
    # For all other (a,b,c): sum_s U[a,s] * V[b,s] * W[c,s] = 0
    #
    # This is a system of polynomial equations in the U, V, W entries.
    # For small r, we can try to solve it.
    
    # Let me try a structured approach: assume U, V, W have a specific pattern
    # based on the Strassen decomposition of sub-blocks.
    
    # For now, return None and let the greedy method handle it.
    return None


def _direct_low_rank_search(n, m, p, max_rank=36, time_budget=None, start_time=None):
    """
    Search for a low-rank decomposition using a structured approach.
    
    Key idea: The 2x4*4x5 multiplication tensor can be decomposed using
    Strassen-like formulas applied to 2x2*2x2 sub-blocks.
    
    Let's try to construct a rank-r decomposition by solving for the
    coefficients directly, assuming a specific structure.
    """
    if start_time is None:
        start_time = time.time()
    
    nm = n * m  # 8
    mp = m * p  # 20
    np_ = n * p  # 10
    
    # Build the target tensor
    T = np.zeros((nm, mp, np_), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try ranks from low to high
    for target_rank in range(20, max_rank + 1):
        if time_budget is not None and time.time() > time_budget - 5:
            break
        
        # Use random initialization with restarts
        best_result = None
        for restart in range(3):
            if time_budget is not None and time.time() > time_budget - 5:
                break
            
            # Initialize with random half-integer values
            np.random.seed(42 + restart * 1000 + target_rank)
            U = (np.random.randint(-4, 5, size=(nm, target_rank)) / 2.0)
            V = (np.random.randint(-4, 5, size=(mp, target_rank)) / 2.0)
            W = (np.random.randint(-4, 5, size=(np_, target_rank)) / 2.0)
            
            # ALS iterations to minimize the error
            for it in range(50):
                if time_budget is not None and time.time() > time_budget - 10:
                    break
                
                # Update U
                for a in range(nm):
                    # U[a,:] = (T[a,:,:] * (V*W)) / (V^2 * W^2)
                    num = np.sum(T[a, :, :] * np.outer(V, W).T, axis=(0,1))
                    denom = np.sum(np.outer(V, V) * np.outer(W, W))
                    if denom > 1e-12:
                        U[a, :] = num / denom
                    else:
                        U[a, :] = 0
                
                # Update V
                for b in range(mp):
                    num = np.sum(T[:, b, :] * np.outer(U, W), axis=(0,1))
                    denom = np.sum(np.outer(U, U) * np.outer(W, W))
                    if denom > 1e-12:
                        V[b, :] = num / denom
                    else:
                        V[b, :] = 0
                
                # Update W
                for c in range(np_):
                    num = np.sum(T[:, :, c] * np.outer(U, V), axis=(0,1))
                    denom = np.sum(np.outer(U, U) * np.outer(V, V))
                    if denom > 1e-12:
                        W[c, :] = num / denom
                    else:
                        W[c, :] = 0
            
            # Snap to half-integers
            U_s = np.round(U * 2) / 2.0
            V_s = np.round(V * 2) / 2.0
            W_s = np.round(W * 2) / 2.0
            
            # Check
            if _check_decomposition(U_s, V_s, W_s, n, m, p):
                best_result = (U_s, V_s, W_s)
                break
            
            # Also check without snapping
            if _check_decomposition(U, V, W, n, m, p, tol=1e-6):
                if _check_half_integer(U, V, W, tol=1e-6):
                    best_result = (U, V, W)
                    break
        
        if best_result is not None:
            return best_result
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook_decomposition(n, m, p)
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    best_rank = n * m * p
    
    # Try direct low-rank search first (faster than greedy rational)
    try:
        result = _direct_low_rank_search(n, m, p, max_rank=36, 
                                          time_budget=deadline - 20, 
                                          start_time=start_time)
        if result is not None:
            U, V, W = result
            if _check_decomposition(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = U.shape[1]
    except Exception:
        pass
    
    # If we haven't improved, try greedy rational decomposition
    if best_rank == n * m * p:
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
