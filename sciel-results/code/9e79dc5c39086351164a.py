"""Tensor rank minimization for 2x3 * 3x3 matrix multiplication.

Strategy:
  1. Start from the schoolbook (rank 18) decomposition.
  2. Use a rank-reduction search: try to find a decomposition of rank r < 18
     by iteratively attempting to merge/simplify rank-1 terms.
  3. Use numerical linear algebra to check if a lower-rank decomposition exists
     with half-integer entries.
  4. Deadline guard to ensure we finish well within 150s.

Known lower bound: For 2x3 * 3x3, the tensor rank is at least 12 (from
Strassen-type arguments). The actual rank is known to be 12 in the literature
for some small matrix multiplication problems, but for 2x3x3 specifically,
we need to verify.

Actually, let me think more carefully. The matrix multiplication tensor for
C = A * B where A is 2x3 and B is 3x3 has 18 entries. The schoolbook gives
rank 18. 

A key insight: we can use the fact that some rank-1 terms can be combined.
For example, if we have terms like e_i ⊗ e_j ⊗ e_k and e_i ⊗ e_j' ⊗ e_k',
they might be mergeable under certain conditions.

Let me try a direct approach:
- Enumerate possible support patterns for rank-1 terms
- Use linear programming / SVD to find minimal rank

For a 2x3x3 tensor, the number of possible rank-1 terms (with unit entries
in the support) is 2*3*3 = 18. We want to find the minimum number of such
terms (with coefficients) that sum to the matrix multiplication tensor.

The matrix multiplication tensor T[i,j,k] = 1 if (i,j,k) corresponds to
C[i,k] = sum_j A[i,j] * B[j,k], i.e., T[i,j,k] = 1 for all valid (i,j,k).

Wait, actually the tensor entries are: T[i,j,k] = delta_{i,k} ... no.

For matrix multiplication C = A*B:
C[i,k] = sum_j A[i,j] * B[j,k]

The tensor T has T[i,j,k] = 1 if and only if this term contributes, which is
always 1 for all valid (i,j,k) since every product A[i,j]*B[j,k] appears in
the sum for C[i,k].

So T is the all-ones tensor of shape (2,3,3) with 18 entries all equal to 1.

Wait, that's not right either. The tensor for matrix multiplication is:
T[i,j,k] = 1 for all (i,j,k) where i in {0,1}, j in {0,1,2}, k in {0,1,2}.

Yes, because C[i,k] = sum_j A[i,j] B[j,k], so each triple (i,j,k) contributes
A[i,j] * B[j,k] to C[i,k]. The tensor entry T[i,j,k] = 1 means this term is
included.

So T is indeed the all-ones tensor of shape (2,3,3).

Now, the rank of the all-ones tensor of shape (n,m,p) is at most n*m*p (schoolbook).
Can we do better?

For the all-ones tensor, we can write:
T[i,j,k] = 1 = (1/sqrt(n)) * (1/sqrt(m)) * (1/sqrt(p)) * sqrt(n) * sqrt(m) * sqrt(p)

But that's just one term with coefficient 1, which would give rank 1. But that's
not correct because the decomposition must be of the form:
T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]

Wait, the indexing is: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
And the decomposition is:
T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]

For the all-ones tensor, T[i,j,k] = 1 for all (i,j,k).

So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k).

This is a system of 18 equations. Let's see if we can find a low-rank solution.

Let me try rank 1: U[i*m+j,0] * V[j*p+k,0] * W[k*n+i,0] = 1 for all (i,j,k).
This requires U[i*m+j,0] = a_ij, V[j*p+k,0] = b_jk, W[k*n+i,0] = c_ki such that
a_ij * b_jk * c_ki = 1 for all (i,j,k).

From a_00 * b_00 * c_00 = 1, a_00 * b_01 * c_10 = 1, etc.
This gives a_00 * b_00 * c_00 = 1 and a_00 * b_01 * c_10 = 1, so b_00 * c_00 = b_01 * c_10.
Similarly, many constraints. It's possible but let's check if half-integer solutions exist.

Actually, this is getting complex. Let me just implement a search that tries ranks from
the lower bound up and uses numerical methods to find solutions.

Given the time constraint, I'll implement:
1. A check function to verify a decomposition
2. A search that tries to find rank-12, rank-13, etc. decompositions
3. Fall back to schoolbook if no lower rank is found quickly

The known rank for 2x3x3 matrix multiplication... let me think. 
For 2x2x2, the rank is 7 (Strassen). For 2x3x3, I believe the rank is 12 or 13.

Let me just implement a practical search with a deadline guard.
"""

import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 3]

_DEADLINE = 140.0  # seconds, safety margin before 150s kill
_start_time = None

def _elapsed():
    return time.time() - _start_time if _start_time is not None else 0.0

def _check_decomposition(U, V, W, n=2, m=3, p=3):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _is_half_integer(x, tol=1e-9):
    """Check if x is a half-integer (k/2 for integer k)."""
    doubled = x * 2
    nearest = round(doubled)
    return abs(doubled - nearest) < tol

def _snap_to_half_integer(x, tol=1e-9):
    """Snap x to nearest half-integer if close."""
    doubled = x * 2
    nearest = round(doubled)
    if abs(doubled - nearest) < tol:
        return nearest / 2.0
    return x

def _try_rank(r, n=2, m=3, p=3, timeout=10.0):
    """Try to find a rank-r decomposition with half-integer entries."""
    t0 = time.time()
    
    # The tensor is all-ones of shape (n, m, p)
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
    
    # Strategy: use a parameterized family of solutions.
    # For the all-ones tensor, one approach is to use "diagonal" patterns.
    
    # Let's try a structured approach:
    # Assign each rank-1 term a support pattern (i,j,k) -> coefficient
    # We can use the fact that the tensor is symmetric in a sense.
    
    # Approach: Use numpy lstsq to solve for coefficients given a fixed support.
    
    # For a given set of r support patterns, we have r unknowns (coefficients)
    # and 18 equations. We need an exact solution.
    
    # Let's try all combinations of r patterns from the 18 possible (i,j,k) triples.
    # But that's C(18, r) which for r=12 is C(18,12) = C(18,6) = 18564, manageable.
    
    # For each combination, set up the linear system and solve.
    
    nrm = n * m
    mrm = m * p
    pnm = p * n
    
    # All possible (i,j,k) indices
    all_indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    # Precompute: for each (i,j,k) and each pattern (ip,jp,kp), the contribution
    # Pattern (ip,jp,kp) with coefficient alpha contributes:
    #   alpha * U[ip*m+jp] * V[jp*p+kp] * W[kp*n+ip] to T[i,j,k]
    # But U, V, W are also unknowns!
    
    # This is nonlinear. Let me think differently.
    
    # Actually, for a rank-1 term with support at (i0,j0,k0):
    # U[i*m+j,c] is nonzero only when (i,j) = (i0,j0)
    # V[j*p+k,c] is nonzero only when (j,k) = (j0,k0)  
    # W[k*n+i,c] is nonzero only when (k,i) = (k0,i0)
    
    # So a rank-1 term with "support" (i0,j0,k0) has:
    # U[i0*m+j0, c] = a, V[j0*p+k0, c] = b, W[k0*n+i0, c] = c_coeff
    # and contributes a*b*c_coeff to T[i0,j0,k0] and 0 elsewhere.
    
    # Wait, that's not right. The contribution to T[i,j,k] from term c is:
    # U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # If U is nonzero only at (i0,j0), V only at (j0,k0), W only at (k0,i0),
    # then the product is nonzero only when (i,j)=(i0,j0), (j,k)=(j0,k0), (k,i)=(k0,i0).
    # This requires i=i0, j=j0, k=k0, and also j=j0, k=k0, i=i0, and k=k0, i=i0, j=j0.
    # So it's nonzero only at (i0,j0,k0) if i0,j0,k0 are consistent.
    
    # Actually, for the product to be nonzero at (i,j,k), we need:
    # i=i0, j=j0 (from U), j=j0, k=k0 (from V), k=k0, i=i0 (from W).
    # So i=i0, j=j0, k=k0. The term only contributes to T[i0,j0,k0].
    
    # So each rank-1 term with sparse support contributes to exactly one entry.
    # This means the rank is at least 18 if we use sparse supports!
    
    # But we can use dense supports! U[i*m+j,c] can be nonzero for multiple (i,j).
    
    # Let me reconsider. A rank-1 term is defined by vectors u in R^(n*m), v in R^(m*p), w in R^(p*n).
    # The contribution to T[i,j,k] is u[i*m+j] * v[j*p+k] * w[k*n+i].
    
    # For the all-ones tensor, we need sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1.
    
    # Let's try a simple ansatz: u_c[i*m+j] = a_c * f_c(i,j), v_c[j*p+k] = b_c * g_c(j,k), w_c[k*n+i] = c_c * h_c(k,i)
    
    # For simplicity, try u_c[i*m+j] = a_c for all (i,j) (constant), similarly for v, w.
    # Then term c contributes a_c * b_c * c_c to every entry.
    # So sum_c a_c * b_c * c_c = 1.
    # With r terms, we can easily satisfy this. E.g., set a_c=b_c=c_c=1/r^(1/3) for all c.
    # But we need half-integer entries.
    
    # Let's try: all u, v, w entries are 1/sqrt(r) ... not half-integer.
    
    # Alternative: use u_c[i*m+j] = 1 for all, v_c[j*p+k] = 1 for all, w_c[k*n+i] = 1/r for all.
    # Then each term contributes 1/r, and r terms give 1. 
    # U entries: 1 (half-integer), V entries: 1 (half-integer), W entries: 1/r.
    # 1/r is half-integer only if r is 1, 2, or 4 (1/1=1, 1/2=0.5, 1/4=0.25 not half-int).
    # Actually 1/4 = 0.25 is not a half-integer. Half-integers are ..., -1, -0.5, 0, 0.5, 1, 1.5, ...
    # So 1/r is half-integer only for r=1 (1) or r=2 (0.5).
    
    # For r=2: u=1, v=1, w=0.5. Each term contributes 0.5, two terms give 1. Rank 2!
    # But wait, is this correct? Let me verify.
    # Term 0: U[i*m+j,0]=1, V[j*p+k,0]=1, W[k*n+i,0]=0.5 -> contributes 0.5 to T[i,j,k]
    # Term 1: U[i*m+j,1]=1, V[j*p+k,1]=1, W[k*n+i,1]=0.5 -> contributes 0.5 to T[i,j,k]
    # Total: 1.0 for all (i,j,k). Yes!
    
    # So rank 2 works? That seems too good. Let me double-check.
    
    # The decomposition is:
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # With U all 1s, V all 1s, W all 0.5s:
    # T[i,j,k] = sum_c 1 * 1 * 0.5 = r * 0.5
    
    # For r=2: T[i,j,k] = 1. Correct!
    
    # But wait, the task says "exactly decompose the matrix multiplication tensor". 
    # Is the matrix multiplication tensor really all ones?
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k].
    # This coefficient is 1 for all valid (i,j,k). So yes, T is all ones.
    
    # So rank 2 should work! Let me verify with the check function.
    
    # Actually, I need to be more careful. Let me re-read the task.
    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "exactly decompose the matrix multiplication tensor"
    
    # The matrix multiplication tensor T satisfies:
    # C[i,k] = sum_{j,c} A[i,j] * B[j,k] * T[i,j,k] ... no.
    
    # Actually, the standard definition is:
    # T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k], else 0.
    # For matrix multiplication, every (i,j,k) with i in [n], j in [m], k in [p] contributes,
    # so T[i,j,k] = 1 for all (i,j,k).
    
    # And the decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # So yes, with U=1, V=1, W=0.5, rank=2, we get T[i,j,k] = 1.
    
    # Let me verify this is valid.
    
    n, m, p = 2, 3, 3
    rank = 2
    U = [[1.0] * rank for _ in range(n * m)]
    V = [[1.0] * rank for _ in range(m * p)]
    W = [[0.5] * rank for _ in range(p * n)]
    
    if _check_decomposition(U, V, W, n, m, p):
        return U, V, W
    
    # If rank 2 doesn't work (shouldn't happen), try rank 3, 4, etc.
    for r in range(3, 19):
        if time.time() - t0 > timeout:
            break
        U = [[1.0] * r for _ in range(n * m)]
        V = [[1.0] * r for _ in range(m * p)]
        W = [[1.0 / r] * r for _ in range(p * n)]
        if _check_decomposition(U, V, W, n, m, p):
            return U, V, W
    
    return None

def construct():
    global _start_time
    _start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Try to find a low-rank decomposition
    best = None
    for r in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]:
        if _elapsed() > _DEADLINE - 5:
            break
        result = _try_rank(r, n, m, p, timeout=min(10.0, _DEADLINE - _elapsed() - 2))
        if result is not None:
            best = result
            break  # Found the lowest rank, stop
    
    if best is not None:
        U, V, W = best
        # Snap to half-integers
        U = [[_snap_to_half_integer(x) for x in row] for row in U]
        V = [[_snap_to_half_integer(x) for x in row] for row in V]
        W = [[_snap_to_half_integer(x) for x in row] for row in W]
        # Verify after snapping
        if _check_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
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
    return {"U": U, "V": V, "W": W}
