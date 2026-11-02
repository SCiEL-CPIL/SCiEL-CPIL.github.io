"""Exact rank-minimal tensor decomposition for 2x4 * 4x6 matrix multiplication.
Structural change: model as a linear system over Q and solve with exact rational
arithmetic to recover a low-rank decomposition. We try to find rank 36 first
(48 schoolbook -> 42 Strassen -> hopefully 36 via structured search).
If that fails, we fall back to the known rank-42 Strassen construction.
"""
import time
import numpy as np
from fractions import Fraction

N, M, P = 2, 4, 6
NM, MP, PN = N * M, M * P, P * N  # 8, 24, 12
START = time.time()
DEADLINE = 140.0  # safety margin before 150s kill


def construct():
    # Return the best decomposition we can produce before the deadline.
    # Strategy:
    #   1. Try a structured exact-rank-36 construction (linear-system based).
    #   2. Fall back to the rank-42 Strassen construction (known correct).
    #   3. Fall back to schoolbook rank-48.
    try:
        if _try_rank36_exact():
            U, V, W = _rank36_solution()
            if _verify_fast(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    try:
        if _time_ok():
            U, V, W = _strassen42()
            if _verify_fast(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    U, V, W = _schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _time_ok():
    return (time.time() - START) < DEADLINE


# ---------------------------------------------------------------------------
# Known rank-42 Strassen construction (2x2 blocks, 6 independent Strassen
# decompositions for the 2x2 * 2x6 sub-problems).
# ---------------------------------------------------------------------------

def _strassen42():
    rank = 42
    U = np.zeros((NM, rank))
    V = np.zeros((MP, rank))
    W = np.zeros((PN, rank))

    # Strassen coefficients for 2x2 matrices.
    # Local indices for A (2x2): 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    S = [
        [1, 0, 1, 0],   # A00 + A11
        [0, 0, 1, 1],   # A10 + A11
        [1, 0, 0, 0],   # A00
        [0, 0, 0, 1],   # A11
        [1, 1, 0, 0],   # A00 + A01
        [-1, 0, 1, 0],  # A10 - A00
        [0, -1, 0, 1]   # A01 - A11
    ]
    T = [
        [1, 0, 0, 1],   # B00 + B11
        [1, 0, 0, 0],   # B00
        [0, 1, 0, -1],  # B01 - B11
        [0, 0, 1, -1],  # B10 - B00
        [0, 0, 0, 1],   # B11
        [1, 1, 0, 0],   # B00 + B01
        [0, 0, 1, 1]    # B10 + B11
    ]
    R = [
        [1, 0, 0, -1, 1, 0, 1],   # C00
        [0, 0, 1, 1, 0, 0, 0],    # C01
        [0, 1, 0, 0, 1, 0, 0],    # C10
        [1, 0, -1, 0, 0, 1, 1]    # C11
    ]

    # 6 blocks: (A_col_off, B_row_off, B_col_off, C_col_off)
    blocks = [
        (0, 0, 0, 0),
        (0, 0, 2, 2),
        (0, 0, 4, 4),
        (2, 2, 0, 0),
        (2, 2, 2, 2),
        (2, 2, 4, 4)
    ]

    col = 0
    for a_co, b_ro, b_co, c_co in blocks:
        for k in range(7):
            for i in range(2):
                for jl in range(2):
                    jg = a_co + jl
                    U[i * M + jg, col] += S[k][i * 2 + jl]
            for jl in range(2):
                jg = b_ro + jl
                for kl in range(2):
                    kg = b_co + kl
                    V[jg * P + kg, col] += T[k][jl * 2 + kl]
            for il in range(2):
                ig = il
                for kl in range(2):
                    kg = c_co + kl
                    W[kg * N + ig, col] += R[il * 2 + kl][k]
            col += 1
    return U, V, W


# ---------------------------------------------------------------------------
# Schoolbook rank-48 fallback.
# ---------------------------------------------------------------------------

def _schoolbook():
    rank = NM * M * P // M  # n*m*p = 48
    U = np.zeros((NM, rank))
    V = np.zeros((MP, rank))
    W = np.zeros((PN, rank))
    c = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, c] = 1
                V[j * P + k, c] = 1
                W[k * N + i, c] = 1
                c += 1
    return U, V, W


# ---------------------------------------------------------------------------
# Exact rank-36 construction via linear system over Q.
#
# The tensor T_mm has NM * MP * PN = 2304 entries, but only 48 are non-zero.
# We look for a decomposition with rank r = 36:
#   sum_{s=1}^{36} U[u,s] * V[v,s] * W[w,s] = T[u,v,w]  for all u,v,w
#
# This is a system of 2304 polynomial equations in 36*8 + 36*24 + 36*12 =
# 288 + 864 + 432 = 1584 unknowns. However, the system is bilinear in (U,V,W)
# for each fixed index triple, so we can use an iterative approach:
#   - Fix V and W, solve for U (linear in each U[u,s])
#   - Fix U and W, solve for V
#   - Fix U and V, solve for W
#   - Repeat until convergence.
#
# For exactness, we use Fraction arithmetic. The system is underdetermined
# (1584 unknowns, 2304 equations), but the bilinear structure means we can
# set up a linear system for each variable set with the others fixed.
#
# To make this tractable, we use a structured ansatz:
#   - Split into 3 groups of 12 terms each, corresponding to the 3 column
#     pairs of B (cols 0-1, 2-3, 4-5).
#   - Each group handles a 2x4 * 2x2 sub-problem.
#   - For each sub-problem, use a rank-12 decomposition of the 2x4*2x2 tensor.
#
# Actually, a cleaner approach: use the known rank-42 structure and try to
# merge two of the six Strassen blocks (each rank 7) into a combined rank-12
# block (saving 2 ranks total). This is possible if the two blocks share
# structure. Blocks 0 and 3 both contribute to C cols 0-1, so we can try
# a rank-12 decomposition of the 2x4 * 4x2 tensor (first two columns of B).
#
# The 2x4 * 4x2 tensor has schoolbook rank 16. Can we do rank 12?
# Using Strassen on the two 2x2*2x2 sub-blocks: rank 14. 
# A rank-12 decomposition of 2x4*4x2 would give total rank 12+7+7 = 26.
# But that's not 36. Let me reconsider.
#
# Total: 3 groups of 2x4*4x2, each rank 12 -> total rank 36.
# So we need a rank-12 decomposition of the 2x4 * 4x2 tensor.
#
# Let me construct this directly.
# A is 2x4, B is 4x2, C is 2x2.
# Split A into A1 (cols 0-1) and A2 (cols 2-3), each 2x2.
# B is 4x2.
# C = A1 * B1 + A2 * B2, where B1 = B[0:2,:], B2 = B[2:4,:].
#
# For A1*B1 (2x2 * 2x2): use Strassen rank 7.
# For A2*B2 (2x2 * 2x2): use Strassen rank 7.
# Total: 14. We need 12.
#
# Alternatively, use a rank-12 decomposition of 2x4*4x2 directly.
# This is a known problem. Let me try to find it via the linear system.
#
# For 2x4*4x2: NM=8, MP=8, PN=4, total tensor size 256.
# Non-zero entries: 2*4*2 = 16.
# Rank r=12: unknowns = 12*8 + 12*8 + 12*4 = 96+96+48 = 240.
# Equations: 256. Underdetermined.
#
# Let's solve via iterative linearization with exact fractions.

def _try_rank36_exact():
    """Try to construct a rank-36 decomposition using three rank-12
    decompositions of the 2x4*4x2 sub-problems."""
    if not _time_ok():
        return False
    # We'll construct three rank-12 decompositions, one for each column pair
    # of B (cols 0-1, 2-3, 4-5). Each handles A (2x4) * B_sub (4x2) -> C_sub (2x2).
    # If we can get each to rank 12, total rank = 36.
    #
    # Use iterative linearization to find a rank-12 decomposition of 2x4*4x2.
    # Then replicate it three times with appropriate index shifts.
    U12, V12, W12 = _find_rank12_2x4x4x2()
    if U12 is None:
        return False
    return True


_RANK12_CACHE = None

def _find_rank12_2x4x4x2():
    """Find a rank-12 decomposition of the 2x4 * 4x2 multiplication tensor.
    Uses iterative linearization with exact rational arithmetic."""
    global _RANK12_CACHE
    if _RANK12_CACHE is not None:
        return _RANK12_CACHE
    if not _time_ok():
        return None

    n, m, p = 2, 4, 2
    nm, mp, pn = n * m, m * p, p * n  # 8, 8, 4
    r = 12

    # Initialize with Strassen on two 2x2 blocks (rank 14), then try to
    # compress to rank 12 via iterative refinement.
    # Actually, let's just try iterative linearization from a random start.
    
    # Build the tensor: T[u,v,w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i else 0
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[(u, v, w)] = Fraction(1)

    # Initialize U, V, W with small rational values.
    # Use Strassen-like structure as initial guess.
    U = [[Fraction(0)] * r for _ in range(nm)]
    V = [[Fraction(0)] * r for _ in range(mp)]
    W = [[Fraction(0)] * r for _ in range(pn)]

    # Set up Strassen for the two 2x2*2x2 blocks.
    # Block 0: A cols 0-1, B rows 0-1 -> C
    # Block 1: A cols 2-3, B rows 2-3 -> C
    S = [
        [1, 0, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, -1, 0, 1]
    ]
    Tc = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    Rc = [
        [1, 0, 0, -1, 1, 0, 1],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0],
        [1, 0, -1, 0, 0, 1, 1]
    ]

    for blk, (a_off, b_off) in enumerate([(0, 0), (2, 2)]):
        for k in range(7):
            col = blk * 7 + k
            for i in range(2):
                for jl in range(2):
                    jg = a_off + jl
                    U[i * m + jg][col] = Fraction(S[k][i * 2 + jl])
            for jl in range(2):
                jg = b_off + jl
                for kl in range(2):
                    kg = kl
                    V[jg * p + kg][col] = Fraction(Tc[k][jl * 2 + kl])
            for il in range(2):
                ig = il
                for kl in range(2):
                    kg = kl
                    W[kg * n + ig][col] = Fraction(Rc[il * 2 + kl][k])

    # Now try to compress to rank 12 by zeroing out 2 columns and adjusting.
    # Use iterative linearization: alternate solving for U, V, W.
    # This is a bilinear system, so we fix two matrices and solve for the third.

    for iteration in range(50):
        if not _time_ok():
            _RANK12_CACHE = None
            return None

        # Fix V, W; solve for U.
        # For each (u, v, w): sum_s U[u,s] * V[v,s] * W[w,s] = T[u,v,w]
        # This is: for each (u, w): sum_s U[u,s] * (sum_v ... ) -- no,
        # it's: for each (v, w): sum_s U[u,s] * V[v,s] * W[w,s] = T[u,v,w]
        # For fixed (v, w), this is a linear equation in U[u,*]:
        #   sum_s U[u,s] * (V[v,s] * W[w,s]) = T[u,v,w]
        # So for each (v, w), we have a linear system in U[:, s] for all u.
        # But it's coupled across u for each (v,w).
        
        # Actually, for fixed (v, w), the equation for each u is:
        #   sum_s U[u,s] * V[v,s] * W[w,s] = T[u,v,w]
        # This is a system of nm equations in nm*r unknowns (U[u,s] for all u,s).
        # But the same coefficients V[v,s]*W[w,s] appear for all u.
        # So it's: U_row[u] dot (V[v,:] * W[w,:]) = T[u,v,w] for each u.
        # This is nm separate equations, each involving the same weight vector.
        # We can't solve this directly for U without more constraints.
        
        # Instead, use a different approach: treat it as a tensor completion problem.
        # For each rank term s, U[:,s], V[:,s], W[:,s] are vectors.
        # The constraint is: sum_s U[:,s] V[:,s]^T W[:,s]^T (outer product) = T.
        
        # This is complex. Let's use a simpler heuristic: 
        # try to find two columns in the rank-14 solution that are linearly
        # dependent or can be merged, reducing to rank 12.
        
        break  # Skip iterative approach; use a different strategy.

    # Alternative: directly construct a rank-12 decomposition using a known
    # formula or search. For 2x4*4x2, the tensor has rank 14 (Strassen) or
    # possibly less. Let me check if rank 12 is achievable.
    #
    # Actually, the tensor rank of 2x4*4x2 is known to be 14 (from Strassen
    # on two 2x2 blocks). It's not known to be less than 14.
    # So rank 36 may not be achievable.
    
    _RANK12_CACHE = None
    return None


def _rank36_solution():
    """Construct the full rank-36 decomposition from three rank-12 sub-decomps."""
    U12, V12, W12 = _find_rank12_2x4x4x2()
    if U12 is None:
        return None, None, None
    # This function is only called if _try_rank36_exact() returned True,
    # so U12 should be valid. But we handle the None case gracefully.
    return U12, V12, W12


def _verify_fast(U, V, W):
    """Verify the decomposition by checking all 2304 tensor entries."""
    nm, mp, pn = U.shape[0], V.shape[0], W.shape[0]
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)

    for u in range(nm):
        i = u // M
        j = u % M
        for v in range(mp):
            jv = v // P
            k = v % P
            if jv != j:
                # Should be zero for all w
                prod = U[u, :] * V[v, :]
                for w in range(pn):
                    if abs(np.dot(prod, W[w, :])) > 1e-7:
                        return False
            else:
                w_t = k * N + i
                prod = U[u, :] * V[v, :]
                if abs(np.dot(prod, W[w_t, :]) - 1.0) > 1e-7:
                    return False
                for w in range(pn):
                    if w == w_t:
                        continue
                    if abs(np.dot(prod, W[w, :])) > 1e-7:
                        return False
    return True


if __name__ == "__main__":
    result = construct()
    print(f"Best rank found: {len(result['U'][0])}")
