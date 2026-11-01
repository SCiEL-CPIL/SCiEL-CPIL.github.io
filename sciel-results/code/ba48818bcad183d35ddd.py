"""Matrix multiplication 3x3 by 3x8: rank-minimal exact tensor decomposition.

Strategy:
  * The multiplication tensor T[i,j,k] = 1 for all (i,j,k) in {0..2}x{0..2}x{0..7}.
  * We seek a rank-r decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    where U is (n*m, r), V is (m*p, r), W is (p*n, r).
  * For a fixed rank component c, the contribution to (i,j,k) is u_c(i,j) * v_c(j,k) * w_c(i,k).
    This must be consistent: for each c, the pattern (u_c, v_c, w_c) must multiply out to 1
    at positions where the schoolbook term exists and 0 elsewhere... but actually the tensor is
    all-ones on the valid index set, so we just need sum over c of the products to equal 1
    for every (i,j,k).

  Key insight: If we can find rank-1 terms where u_c(i,j)*v_c(j,k)*w_c(i,k) is either 0 or 1
  for all (i,j,k), and the sum equals 1 everywhere, we're done. But that's hard.

  Alternative: Use the fact that for the all-ones tensor on a grid, we can use additive
  decompositions. For example, if we set W[k*n+i, c] = 1 for all c, then we need
  sum_c U[i*m+j, c] * V[j*p+k, c] = 1 for all (i,j,k).

  This reduces to: find matrices U (9 x r) and V (24 x r) such that for all i,j,k:
    sum_c U[i*m+j, c] * V[j*p+k, c] = 1

  This is equivalent to: the matrix product structure where row i*m+j of U times row j*p+k of V
  (element-wise, summed) equals 1.

  A simple construction: set U[i*m+j, c] = 1/sqrt(r) and V[j*p+k, c] = sqrt(r) for c=0..r-1.
  Then sum_c = r * (1/sqrt(r)) * sqrt(r) = r. Not 1.

  Better: U[i*m+j, c] = 1, V[j*p+k, c] = 1/r for all c. Then sum = r * 1/r = 1.
  So rank r works with W all ones. But we want minimal r.

  The constraint is that for each j, the row U[i*m+j, :] and V[j*p+k, :] must satisfy
  dot(U[i*m+j, :], V[j*p+k, :]) = 1 for all i,k.

  So for each j, we need a set of r vectors u_j[i] = U[i*m+j, :] (for i=0,1,2) and
  v_j[k] = V[j*p+k, :] (for k=0..7) such that u_j[i] . v_j[k] = 1 for all i,k.

  This means all u_j[i] are orthogonal to (v_j[k] - v_j[0]) for all k, and u_j[i] . v_j[0] = 1.

  So u_j[i] lies in the 1-dimensional space spanned by some vector perpendicular to all
  (v_j[k] - v_j[0]). If v_j[0] = ... = v_j[7] (all equal), then u_j[i] just needs to have
  dot product 1 with that common vector.

  So the minimal construction: for each j, pick a vector w_j in R^r with ||w_j||^2 = 1 (or whatever),
  set V[j*p+k, :] = w_j for all k, and set U[i*m+j, :] = w_j for all i. Then
  U[i*m+j, :] . V[j*p+k, :] = w_j . w_j = ||w_j||^2.

  For this to equal 1, we need ||w_j||^2 = 1. So we need r vectors w_0, w_1, w_2 in R^r
  each with norm 1, and they can be anything (even the same vector).

  If we use the same vector w for all j, then r=1 works! U[i*m+j, 0] = 1, V[j*p+k, 0] = 1,
  W[k*n+i, 0] = 1. Then sum_c U*V*W = 1*1*1 = 1. 

  Wait, let me re-examine the tensor decomposition formula. The standard form is:
  T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]

  For 3x3x8: T[i,j,k] = 1 for all i in {0,1,2}, j in {0,1,2}, k in {0,..,7}.

  With U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for a single c:
  T[i,j,k] = 1*1*1 = 1. 

  So rank 1 should work! But that contradicts known lower bounds. Let me re-read the task.

  "output factor follows the official trace convention W[k*n+i,r]"

  The decomposition is: C_{i,k} = sum_j A_{i,j} B_{j,k}
  = sum_j sum_c U[i*m+j, c] V[j*p+k, c] * ... 

  Actually, the standard Strassen-style decomposition is:
  C_{i,k} = sum_c (sum_j U[i*m+j, c] A_{i,j}) * (sum_k V[j*p+k, c] B_{j,k}) ... no.

  The correct form: define intermediate values:
  M_c = (sum_j U[i*m+j, c] A_{i,j}) * (sum_k ... ) -- this isn't right either.

  The tensor form: T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}
  where the full computation is: C_{i,k} = sum_{j} A_{i,j} B_{j,k} = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k}

  So T_{i,j,k} must equal 1 for all valid (i,j,k) and 0 otherwise. But in the flattened
  representation, the tensor is in R^{9} x R^{24} x R^{24} (since p*n = 8*3 = 24).

  The entry T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i is 1 if the indices are
  consistent (i.e., the j in a matches the j in b, and the i in a matches the i in c,
  and the k in b matches the k in c), else 0.

  For our case, every (i,j,k) triple gives a unique (a,b,c), and T[a,b,c] = 1 for these
  3*3*8 = 72 entries, and 0 for all other 9*24*24 - 72 entries.

  So it's NOT the all-ones tensor. My earlier analysis was wrong.

  So we need: for each of the 72 valid (i,j,k) triples, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1,
  and for all other (a,b,c) triples, the sum = 0.

  This is a much harder constraint. The rank-1 idea doesn't work because it would give
  non-zero values at invalid positions.

  Let's go back to the seed approach but improve it.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 8]
START = time.time()
DEADLINE = 145.0


def _build_schoolbook():
    n, m, p = DIMENSIONS
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


def _validate_exact(U, V, W, n, m, p):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    U2 = (2 * U).astype(int)
    V2 = (2 * V).astype(int)
    W2 = (2 * W).astype(int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = U2[i * m + j, :] * V2[j * p + k, :] * W2[k * n + i, :]
                # The tensor entry should be 1, so sum over c should be 1
                # But each term is U*V*W, and we're checking element-wise
                # Actually the decomposition is: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                # So we need sum_c = 1, not element-wise equality
                s = np.sum(lhs)
                if s != 4:  # 4 = 2^2 * 1, since we doubled each factor
                    return False
    # Also check that invalid entries are 0
    for a in range(n * m):
        for b in range(m * p):
            # Check if this (a,b) pair corresponds to valid (i,j,k)
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            if j != j2:
                continue
            for c_idx in range(p * n):
                k2 = c_idx // n
                i2 = c_idx % n
                if k2 != k or i2 != i:
                    continue
                # This is a valid position, already checked
    # Need to check all positions where the tensor should be 0
    for a in range(n * m):
        for b in range(m * p):
            for c_idx in range(p * n):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c_idx // n
                i2 = c_idx % n
                if j == j2 and k == k2 and i == i2:
                    continue  # valid, already checked
                # Invalid position: sum_c U[a,c]*V[b,c]*W[c_idx,c] should be 0
                s = np.sum(U2[a, :] * V2[b, :] * W2[c_idx, :])
                if s != 0:
                    return False
    return True


def _to_list(M):
    result = []
    for row in M:
        r = []
        for v in row:
            fv = float(v)
            if fv == int(fv):
                r.append(int(fv))
            else:
                r.append(fv)
        result.append(r)
    return result


def _try_rank(r, n, m, p, seed=0):
    """Try to find a rank-r decomposition using structured search."""
    rng = np.random.default_rng(seed)
    values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    for it in range(2000):
        if time.time() > DEADLINE:
            return None
        
        U = rng.choice(values, size=(n * m, r))
        V = rng.choice(values, size=(m * p, r))
        W = rng.choice(values, size=(p * n, r))
        
        if _validate_exact(U, V, W, n, m, p):
            return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    best = _build_schoolbook()
    best_rank = n * m * p

    lower = max(n * p, m * p, n * m)
    
    for r in range(lower, best_rank):
        if time.time() > DEADLINE - 5:
            break
        
        seed_base = r * 1000
        res = _try_rank(r, n, m, p, seed=seed_base)
        if res is not None:
            best = res
            best_rank = r
            break
    
    return best


if __name__ == "__main__":
    res = construct()
    print("rank:", len(res["U"][0]))
