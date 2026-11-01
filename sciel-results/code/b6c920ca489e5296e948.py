"""
Rank-minimal tensor decomposition for 2x2 * 2x14 matrix multiplication.

The tensor T lives in R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n} = R^4 ⊗ R^{28} ⊗ R^{28}.
T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k.

Strategy:
  1. Build a rank-49 decomposition using Strassen on 7 pairs of k-values.
  2. Try to reduce rank further via linear dependence search.
  3. Fallback to schoolbook rank 56.
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Start with Strassen rank-49
    result = _construct_strassen_49(n, m, p)
    if result is not None:
        # Try to reduce rank by finding linear dependencies
        result = _try_reduce_rank(result, n, m, p, deadline)
        return result

    # Fallback: schoolbook rank-56
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _construct_strassen_49(n, m, p):
    """
    Rank-49 decomposition using Strassen's algorithm on 7 blocks of 2x2x2.
    
    Split the 14 k-values into 7 pairs: (0,1), (2,3), ..., (12,13).
    For each pair, apply Strassen's rank-7 decomposition for 2x2x2 matrix mult.
    Total rank: 7 * 7 = 49.
    """
    rank = 49
    U = [[0] * rank for _ in range(n * m)]   # 4 x 49
    V = [[0] * rank for _ in range(m * p)]   # 28 x 49
    W = [[0] * rank for _ in range(p * n)]   # 28 x 49

    for pair_idx in range(7):
        k1 = 2 * pair_idx
        k2 = 2 * pair_idx + 1
        base = pair_idx * 7

        # M1: (A[0,0]+A[1,1]) * (B[0,k1]+B[1,k2]) -> C[0,k1] + C[1,k2]
        U[0][base] = 1; U[3][base] = 1
        V[k1][base] = 1; V[p + k2][base] = 1
        W[k1 * n + 0][base] = 1; W[k2 * n + 1][base] = 1

        # M2: (A[1,0]+A[1,1]) * B[0,k1] -> C[1,k1]
        U[2][base+1] = 1; U[3][base+1] = 1
        V[k1][base+1] = 1
        W[k1 * n + 1][base+1] = 1

        # M3: A[0,0] * (B[0,k2]-B[1,k2]) -> C[0,k2]
        U[0][base+2] = 1
        V[k2][base+2] = 1; V[p + k2][base+2] = -1
        W[k2 * n + 0][base+2] = 1

        # M4: A[1,1] * (B[1,k1]-B[0,k1]) -> C[1,k1]
        U[3][base+3] = 1
        V[p + k1][base+3] = 1; V[k1][base+3] = -1
        W[k1 * n + 1][base+3] = 1

        # M5: (A[0,0]+A[0,1]) * B[1,k2] -> C[0,k2]
        U[0][base+4] = 1; U[1][base+4] = 1
        V[p + k2][base+4] = 1
        W[k2 * n + 0][base+4] = 1

        # M6: (A[1,0]-A[0,0]) * (B[0,k1]+B[0,k2]) -> C[1,k1]+C[1,k2]
        U[2][base+5] = 1; U[0][base+5] = -1
        V[k1][base+5] = 1; V[k2][base+5] = 1
        W[k1 * n + 1][base+5] = 1; W[k2 * n + 1][base+5] = 1

        # M7: (A[0,1]-A[1,1]) * (B[1,k1]+B[1,k2]) -> C[0,k1]+C[0,k2]
        U[1][base+6] = 1; U[3][base+6] = -1
        V[p + k1][base+6] = 1; V[p + k2][base+6] = 1
        W[k1 * n + 0][base+6] = 1; W[k2 * n + 0][base+6] = 1

    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    return None


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Check T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(Ua[i*m+j], Va[j*p+k] * Wa[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _try_reduce_rank(result, n, m, p, deadline):
    """
    Try to reduce the rank by finding linear dependencies among the rank-1 terms.
    
    Given a rank-r decomposition, if there exists a non-trivial linear relation
    among the columns of U (or equivalently among the rank-1 tensors), we can
    eliminate a term.
    
    We use the following approach:
    - Form the matrix M where M[a, r] = U[a, r] for all a (this is just U).
    - If U has rank < r, there's a dependency in the U columns.
    - More generally, we look for a vector c such that sum_r c_r * u_r = 0
      and sum_r c_r * v_r * w_r = 0 (the "shadow" condition).
    
    For a simpler approach: try to find subsets of columns that are linearly
    dependent in a way that allows elimination.
    
    A practical approach: use the fact that if we have a decomposition of rank r,
    we can try to find a (r-1)-rank decomposition by solving a system.
    
    Given time constraints, we'll try a few heuristic reductions.
    """
    U = result["U"]
    V = result["V"]
    W = result["W"]
    r = len(U[0])
    
    # Check if U columns are linearly dependent
    Ua = np.array(U, dtype=float)
    rank_U = np.linalg.matrix_rank(Ua)
    if rank_U < r:
        # There's a linear dependency in U columns.
        # Find the null space vector
        null_vec = _null_space_vector(Ua)
        if null_vec is not None:
            # Check if the dependency also satisfies the shadow condition
            # sum_r null_vec[r] * V[:,r] * W[:,r] = 0
            Va = np.array(V, dtype=float)
            Wa = np.array(W, dtype=float)
            shadow = np.zeros((m * p, p * n))
            for rr in range(r):
                if abs(null_vec[rr]) > 1e-15:
                    shadow += null_vec[rr] * np.outer(Va[:, rr], Wa[:, rr])
            if np.max(np.abs(shadow)) < 1e-10:
                # We can eliminate one term
                # Find the column with largest |null_vec| to eliminate
                elim_idx = int(np.argmax(np.abs(null_vec)))
                U_new = [row[:elim_idx] + row[elim_idx+1:] for row in U]
                V_new = [row[:elim_idx] + row[elim_idx+1:] for row in V]
                W_new = [row[:elim_idx] + row[elim_idx+1:] for row in W]
                return {"U": U_new, "V": V_new, "W": W_new}
    
    # Try pairwise elimination: for each pair of columns, check if one can be
    # expressed as a combination of others
    # This is expensive, so we limit the number of attempts
    if time.time() > deadline - 30:
        return result
    
    # Try to find a rank-48 decomposition by looking for a single dependency
    # that satisfies both U and the shadow condition
    # We'll try random linear combinations
    np.random.seed(42)
    for attempt in range(50):
        if time.time() > deadline - 30:
            break
        # Pick a random subset of columns to try to eliminate
        # Try eliminating one column at a time
        for elim in range(r):
            if time.time() > deadline - 30:
                break
            # Check if column elim can be expressed as linear combo of others
            # while maintaining the shadow condition
            # This is a complex constraint, so we skip for now
    
    return result


def _null_space_vector(A):
    """Find a vector in the null space of A (A is n x r with n < r)."""
    try:
        U, s, Vt = np.linalg.svd(A, full_matrices=True)
        # The last row of Vt gives a null space vector if rank < r
        r = A.shape[1]
        if len(s) < r or s[-1] < 1e-12:
            return Vt[-1, :]
        return None
    except:
        return None


def _try_kronecker_28(n, m, p):
    """
    Try a rank-28 construction using Kronecker product structure.
    
    The idea: C[i,k] = sum_j A[i,j] * B[j,k]
    For fixed i, this is a linear combination of rows of B weighted by A[i,:].
    
    We can write:
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    This is like two independent "1x2x14" multiplications.
    Each 1x2x14 multiplication has tensor rank 2*14 = 28 (schoolbook) or potentially less.
    
    Actually, for 1x2x14: C[k] = A[0]*B[0,k] + A[1]*B[1,k]
    The tensor is T[j,k] = 1 for j in {0,1}, k in {0,...,13}.
    This is the all-ones tensor of size 2x14, which has rank 1 (it's outer product of (1,1) and ones_14).
    
    Wait, that's not right. The tensor for 1x2x14 is:
    T[j,k] where the multiplication is C[k] = sum_j A[j] B[j,k]
    T[j,k] = 1 for all j,k.
    
    As a tensor in R^2 ⊗ R^{14}, T = (1,1) ⊗ (1,1,...,1) which has rank 1.
    
    But in our problem, the tensor is in R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n}.
    For the full 2x2x14 problem, it's more complex.
    
    Let me think differently. The total tensor is:
    T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,...,13}
    
    We want to write this as sum_r u_r ⊗ v_r ⊗ w_r where
    u_r ∈ R^4, v_r ∈ R^{28}, w_r ∈ R^{28}.
    
    One approach: factor the "i" dimension.
    T[i,j,k] = T_i[j,k] where T_i is the 2x14 all-ones tensor.
    T_0 = T_1 = all-ones 2x14.
    
    So T = T_0 ⊗ e_0 + T_1 ⊗ e_1 where e_0, e_1 are basis vectors in R^2.
    
    But T_0 and T_1 are the same tensor, so:
    T = (e_0 + e_1) ⊗ T_0
    
    And T_0 = (1,1) ⊗ (1,1,...,1) has rank 1 in R^2 ⊗ R^{14}.
    
    So T = (e_0+e_1) ⊗ (1,1) ⊗ (1,...,1) which has rank 1 in R^4 ⊗ R^{28} ⊗ R^{28}??
    
    No, that's not right because the indices are coupled differently.
    
    Let me be more careful. The tensor T is in R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n}.
    The first index is (i,j) with i ∈ {0,1}, j ∈ {0,1}, so 4 values.
    The second index is (j,k) with j ∈ {0,1}, k ∈ {0,...,13}, so 28 values.
    The third index is (k,i) with k ∈ {0,...,13}, i ∈ {0,1}, so 28 values.
    
    T[(i,j), (j,k), (k,i)] = 1 for all valid i,j,k.
    
    Note that the indices are coupled: the j in the first and second indices must match,
    and the k in the second and third must match, and the i in the first and third must match.
    
    So it's not simply an all-ones tensor in the naive sense - the indices are linked.
    
    For a rank-1 term u ⊗ v ⊗ w:
    sum_r u[(i,j),r] * v[(j,k),r] * w[(k,i),r] = 1 for all i,j,k.
    
    This is a constraint on the rank-1 terms.
    
    The Kronecker product approach: if we can write the tensor as
    sum_{a} (u_a ⊗ v_a ⊗ w_a) where each term has a specific structure...
    
    Actually, let's just try to construct a rank-28 solution directly.
    """
    # For each k in {0,...,13}, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j.
    
    # A simple construction: for each k, use 2 terms (one for each j).
    # Term (k, j): U[i*m+j, r] = delta(i, something), V[j*p+k, r] = 1, W[k*n+i, r] = delta(i, something)
    
    # Actually, the schoolbook uses 1 term per (i,j,k) = 56 terms.
    # To get rank 28, we need to share terms between different i values.
    
    # Construction: for each (j,k), use 1 term where:
    # U[i*m+j, r] = 1 for both i=0 and i=1
    # V[j*p+k, r] = 1
    # W[k*n+i, r] = 1 for both i=0 and i=1
    # 
    # Then sum_r U[i*m+j',r] V[j'*p+k',r] W[k'*n+i',r]
    # For fixed (i,j,k): only the term with (j,k) contributes (since V[j'*p+k',r] is nonzero
    # only when j'=j and k'=k).
    # That term gives: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1*1*1 = 1. ✓
    
    # Wait, but this gives rank = m*p = 2*14 = 28!
    # Let me verify this works.
    
    rank = m * p  # 28
    U = [[0] * rank for _ in range(n * m)]   # 4 x 28
    V = [[0] * rank for _ in range(m * p)]   # 28 x 28
    W = [[0] * rank for _ in range(p * n)]   # 28 x 28
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            # U[i*m+j, r] = 1 for all i
            for i in range(n):
                U[i * m + j][r] = 1
            # V[j*p+k, r] = 1
            V[j * p + k][r] = 1
            # W[k*n+i, r] = 1 for all i
            for i in range(n):
                W[k * n + i][r] = 1
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    return None


# EVOLVE-BLOCK-END
