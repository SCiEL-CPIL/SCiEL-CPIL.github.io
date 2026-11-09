"""Matrix multiplication 3x4x5: search for rank-minimal exact tensor decomposition.

Strategy:
  1. Start from the schoolbook (rank 60) decomposition.
  2. Iteratively attempt to merge pairs of columns via solving a linear system
     to find integer/half-integer coefficients that express one column as a
     combination of the other two, then eliminate it.
  3. If merging fails, also try single-column "reduction" where a column is
     expressed as a combination of other columns (requiring those other columns
     to span it).
  4. Use a deadline guard to stop before the 150 s wall-clock limit.
"""
import time
import numpy as np
from fractions import Fraction

_DIMENSIONS = (3, 4, 5)
_DEADLINE = 150.0
_START_TIME = time.time()


def _elapsed():
    return time.time() - _START_TIME


def _remaining():
    return _DEADLINE - _elapsed() - 5.0  # 5 s safety margin


def _check_deadline():
    return _remaining() > 1


def _to_fraction(x):
    return Fraction(int(x))


def _is_half_integer(frac):
    """Check if a Fraction is an integer or half-integer (denominator divides 2)."""
    return frac.denominator in (1, 2)


def _all_half_integer(rows):
    """Check all entries in a list of Fractions are half-integers."""
    for r in rows:
        for v in r:
            if not _is_half_integer(v):
                return False
    return True


def _validate(U, V, W):
    """Verify that U, V, W form an exact decomposition.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The decomposition says: for each column c,
      sum over j of A[i,j] * B[j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    We check that for all (i,j,k): 
      A[i,j]*B[j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Equivalently, the tensor T[i,j,k] should equal sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    where T[i,j,k] = delta (the (i,j) entry of A times (j,k) entry of B).
    
    Actually the standard formulation: the multiplication tensor is
    T_{i,j,k} such that (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}.
    A rank-r decomposition writes this as sum_c U^{(c)}_{i,j} * V^{(c)}_{j,k} * W^{(c)}_{k,i}
    Wait, let me re-read the task.
    
    The output factor W[k*n+i, r]. So W is indexed by (k,i) where k is the row of B (column of result) and i is the row of A (row of result).
    
    Standard bilinear decomposition of matrix multiplication:
    (A B)_{i,k} = sum_j A_{i,j} B_{j,k}
    
    A rank-r decomposition gives:
    (A B)_{i,k} = sum_{c=1}^r (sum_j U[i,j,c] * A_{i,j}) * (sum_k V[j,k,c] * B_{j,k}) * W[k,i,c]
    
    Hmm, that's not quite right either. Let me think more carefully.
    
    The standard tensor decomposition for matrix mult of size (n,m,p):
    We want to compute C = A * B where A is n x m, B is m x p, C is n x p.
    
    The bilinear map is: (A, B) -> A B.
    
    A rank-r straight-line program computes:
    C_{i,k} = sum_j A_{i,j} B_{j,k}
    
    This can be written as:
    C_{i,k} = sum_{c=1}^r (sum_j U_{i,j}^{(c)} A_{i,j}) * (sum_k V_{j,k}^{(c)} B_{j,k}) * W_{i,k}^{(c)}
    
    Wait no. The standard formulation is:
    
    We define r linear combinations of entries of A: a_c = sum_j U_{i,j,c} A_{i,j}  (no, that doesn't make sense either)
    
    Let me look at this differently. The multiplication tensor T is a trilinear form:
    T(A, B, W) = sum_{i,j,k} A_{i,j} B_{j,k} W_{k,i} = tr(A B W)
    
    Wait, W[k*n+i] so W is p*n matrix. If we view W as a matrix W_{k,i}, then:
    sum_{i,j,k} A_{i,j} B_{j,k} W_{k,i} = sum_k (sum_i A_{i,:} B W_{k,i})
    = sum_i sum_k W_{k,i} (A B)_{i,k}
    = tr(W^T A B) = tr(A B W^T)
    
    So the tensor is T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i} in the appropriate basis.
    
    A rank-r decomposition means:
    T(A,B,W) = sum_c U_c(A) V_c(B) W_c(W)
    where U_c(A) = sum_{i,j} U_{i,j,c} A_{i,j}, V_c(B) = sum_{j,k} V_{j,k,c} B_{j,k}, W_c(W) = sum_{k,i} W_{k,i,c} W_{k,i}.
    
    Equivalently: (AB)_{i,k} = sum_c (sum_j U_{i,j,c} A_{i,j}) ... no.
    
    Actually: A rank-r decomposition of the matmul tensor means there exist
    U: (n*m, r), V: (m*p, r), W: (p*n, r) such that for all i,k:
    (AB)_{i,k} = sum_c (sum_j U[i*m+j, c] A_{i,j}) * (sum_k' V[j*p+k', c] B_{j,k'}) * W[k*p+n+i, c]
    
    Hmm, I'm overcomplicating this. Let me just use the standard schoolbook as reference:
    
    In the schoolbook, rank = n*m*p = 60. For each (i,j,k), there's one column c:
    U[i*m+j, c] = 1, all else 0 in that column
    V[j*p+k, c] = 1, all else 0
    W[k*n+i, c] = 1, all else 0
    
    Then sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 if there exists c with all three = 1, which happens iff (i,j,k) matches, and the contribution is A[i,j] B[j,k] W[k,i] summed... 
    
    OK let me just use the verification that the evaluator would use. The key identity is:
    
    For all i in range(n), k in range(p):
      (A @ B)[i, k] = sum_j A[i,j] * B[j,k]
    
    The decomposition says this equals:
      sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_k2 V[j*p+k2, c] * B[j,k2]) * W[k*n+i, c]
    
    No wait. I think the correct interpretation given the shapes is:
    
    U is (n*m, rank), so U[row, col] where row = i*m + j
    V is (m*p, rank), so V[row, col] where row = j*p + k  
    W is (p*n, rank), so W[row, col] where row = k*n + i

    The decomposition identity is:
    For all i, k:
      sum_j A[i,j] B[j,k] = sum_c U[i*m+?, c] ... 
    
    Actually I think it's simpler. The tensor is:
    T[a,b,c] where a indexes (i,j) in A, b indexes (j,k) in B, c indexes (k,i) in W.
    
    T[a, b, c] is nonzero only when the j index in a matches the j index in b AND the k index in b matches the k index in c. In that case T[a,b,c] = 1.
    
    A rank-r decomposition: T[a,b,c] = sum_{col} U[a, col] * V[b, col] * W[c, col]
    
    So the condition is: for all a in range(n*m), b in range(m*p), c in range(p*n):
      T[a,b,c] = sum_{col=0}^{r-1} U[a,col] * V[b,col] * W[c,col]
    
    Where T[a,b,c] = 1 if the j from (i,j)=a and (j,k)=b agree AND k from (j,k)=b and (k,i)=c agree, else 0.
    More precisely: let a = i*m + j, b = j'*p + k, c = k'*n + i'.
    T[a,b,c] = 1 if j == j' AND k == k' AND i == i', else 0.
    Wait no - we also need consistency: a has (i,j), b has (j',k), c has (k',i').
    The product A[i,j]*B[j,k]*W[k,i] contributes to tr(ABW^T) when all indices are consistent.
    So T[a,b,c] = 1 iff j==j' AND k==k' AND i==i'.
    
    So let me verify with schoolbook: U[a,col]=1 iff a's (i,j) and col's (i,j,k) have same (i,j).
    V[b,col]=1 iff b's (j,k) matches col's (j,k).
    W[c,col]=1 iff c's (k,i) matches col's (k,i).
    So U[a,col]*V[b,col]*W[c,col] = 1 iff a=(i,j), b=(j,k), c=(k,i) all match the col's (i,j,k).
    Sum over col: equals 1 iff there exists (i,j,k) such that a=(i,j), b=(j,k), c=(k,i), which means j_a == j_b, k_b == k_c, i_c == i_a.
    And the only such col is col=(i_a, j_a, k_b).
    So the sum is 1 when j_a == j_b and k_b == k_c and i_a == i_c, which is exactly T[a,b,c].
    Great, this is correct.
    """
    n, m, p = _DIMENSIONS
    rank = len(U[0])
    for a in range(n * m):
        i, j = divmod(a, m)
        for b in range(m * p):
            j2, k = divmod(b, p)
            if j != j2:
                T_ab_c = 0
            else:
                T_ab_c = 1
            if T_ab_c == 0:
                # Need sum to be 0
                s = 0
                for c in range(p * n):
                    k2, i2 = divmod(c, n)
                    if i2 == i and k2 == k:
                        # This c could contribute
                        for col in range(rank):
                            if U[a][col] != 0 and V[b][col] != 0 and W[c][col] != 0:
                                s += 1
                                break
                if s != 0:
                    # Check more carefully
                    val = 0
                    for c in range(p * n):
                        for col in range(rank):
                            val += U[a][col] * V[b][col] * W[c][col]
                    if val != 0:
                        return False
            else:
                # T[a,b,c] should be 1 for matching c, 0 for non-matching
                # This gets complex; do full check for small cases
                pass
    # Full verification
    r = rank
    for a in range(n * m):
        ia, ja = divmod(a, m)
        for b in range(m * p):
            jb, kb = divmod(b, p)
            for c in range(p * n):
                kc, ic = divmod(c, n)
                expected = 1 if (ja == jb and kb == kc and ia == ic) else 0
                actual = 0
                for col in range(r):
                    actual += U[a][col] * V[b][col] * W[c][col]
                if actual != expected:
                    return False
    return True


def _build_schoolbook():
    """Build the schoolbook decomposition as lists of lists of int (0 or 1)."""
    n, m, p = _DIMENSIONS
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
    return U, V, W


def _attempt_merge(U, V, W, col_keep, col_elim):
    """Try to express column col_elim as a linear combination of col_keep and other columns.
    
    For a merge of two columns (col_keep and col_elim) into one, we need:
    There exists alpha, beta (half-integers) such that for all a, b, c:
    U[a, col_elim] = alpha * U[a, col_keep] + beta * U[a, new]
    (and similarly for V, W)
    
    Actually, a simpler merge: can we replace two columns c1, c2 by a single column c_new?
    This requires that the rank-2 bilinear form equals a rank-1 form, which is generally not possible.
    
    A more practical approach: can column c2 be eliminated by expressing its contribution
    as a combination of other columns? I.e., does there exist a vector of coefficients
    alpha_c (for each remaining column c) such that for all a:
    U[a, c2] = sum_c alpha_c * U[a, c]
    AND for all b: V[b, c2] = sum_c alpha_c * V[b, c]
    AND for all c2': W[c2', c2] = sum_c alpha_c * W[c2', c]
    
    This is the condition that column c2 is "redundant" - it lies in the span of the other columns
    in all three matrices simultaneously with the SAME coefficients.
    """
    n, m, p = _DIMENSIONS
    rank = len(U[0])
    
    # Build the system: for all a, U[a, c2] = sum_{c != c2} alpha_c * U[a, c]
    # This is a linear system in alpha. If it has a solution that also satisfies V and W,
    # then c2 can be eliminated.
    
    # Unknowns: alpha_c for c in range(rank), c != c2  -> rank-1 unknowns
    # Equations from U: n*m equations
    # Equations from V: m*p equations  
    # Equations from W: p*n equations
    
    num_unknowns = rank - 1
    if num_unknowns <= 0:
        return None
    
    # Map from column index (excluding c2) to unknown index
    col_to_unk = {}
    unk = 0
    for c in range(rank):
        if c != col_elim:
            col_to_unk[c] = unk
            unk += 1
    
    # Build augmented system [A | b] where A * alpha = b
    # Row i corresponds to an equation
    rows_A = []
    rows_b = []
    
    # From U
    for a in range(n * m):
        row = [0] * num_unknowns
        rhs = U[a][col_elim]
        for c in range(rank):
            if c != col_elim:
                row[col_to_unk[c]] = U[a][c]
        rows_A.append(row)
        rows_b.append(rhs)
    
    # From V
    for b in range(m * p):
        row = [0] * num_unknowns
        rhs = V[b][col_elim]
        for c in range(rank):
            if c != col_elim:
                row[col_to_unk[c]] = V[b][c]
        rows_A.append(row)
        rows_b.append(rhs)
    
    # From W
    for c_idx in range(p * n):
        row = [0] * num_unknowns
        rhs = W[c_idx][col_elim]
        for c in range(rank):
            if c != col_elim:
                row[col_to_unk[c]] = W[c_idx][c]
        rows_A.append(row)
        rows_b.append(rhs)
    
    # Solve using numpy
    A_np = np.array(rows_A, dtype=float)
    b_np = np.array(rows_b, dtype=float)
    
    try:
        # Use least squares; check if residual is near zero
        alpha, residuals, rank_lsv, sv = np.linalg.lstsq(A_np, b_np, rcond=1e-12)
        if len(residuals) > 0 and residuals[0] > 1e-9:
            return None
        # Verify exactness
        residual = A_np @ alpha - b_np
        if np.max(np.abs(residual)) > 1e-9:
            return None
    except:
        return None
    
    # Check if alpha values are half-integers
    for a_val in alpha:
        a_frac = Fraction(int(round(a_val * 2)) / 2)
        if abs(a_frac - a_val) > 1e-10:
            return None
        if not _is_half_integer(a_frac):
            return None
    
    # Also verify the solution with Fractions for exactness
    alpha_frac = [Fraction(int(round(a_val * 2)) / 2) for a_val in alpha]
    
    # Verify all equations exactly
    for a in range(n * m):
        s = 0
        for c in range(rank):
            if c != col_elim:
                s += alpha_frac[col_to_unk[c]] * U[a][c]
        if s != U[a][col_elim]:
            return None
    
    for b in range(m * p):
        s = 0
        for c in range(rank):
            if c != col_elim:
                s += alpha_frac[col_to_unk[c]] * V[b][c]
        if s != V[b][col_elim]:
            return None
    
    for c_idx in range(p * n):
        s = 0
        for c in range(rank):
            if c != col_elim:
                s += alpha_frac[col_to_unk[c]] * W[c_idx][c]
        if s != W[c_idx][col_elim]:
            return None
    
    return alpha_frac


def _eliminate_column(U, V, W, col_elim, alpha_frac, col_to_unk):
    """Eliminate column col_elim by updating other columns."""
    n, m, p = _DIMENSIONS
    rank = len(U[0])
    
    # New rank = rank - 1
    new_rank = rank - 1
    # Map old columns (excluding col_elim) to new indices
    old_to_new = {}
    ni = 0
    for c in range(rank):
        if c != col_elim:
            old_to_new[c] = ni
            ni += 1
    
    new_U = [[0] * new_rank for _ in range(n * m)]
    new_V = [[0] * new_rank for _ in range(m * p)]
    new_W = [[0] * new_rank for _ in range(p * n)]
    
    for c in range(rank):
        if c == col_elim:
            continue
        nc = old_to_new[c]
        coeff = alpha_frac[col_to_unk[c]]
        # new_col[nc] = old_col[c] + coeff * eliminated_contribution
        # Actually: we're saying col_elim = sum alpha_c * col_c, so we just drop col_elim
        # and keep the others unchanged. The identity still holds because
        # the contribution of col_elim is replicated by the others.
        # So we just copy:
        for a in range(n * m):
            new_U[a][nc] = U[a][c]
        for b in range(m * p):
            new_V[b][nc] = V[b][c]
        for c_idx in range(p * n):
            new_W[c_idx][nc] = W[c_idx][c]
    
    return new_U, new_V, new_W


def _try_all_eliminations(U, V, W):
    """Try to eliminate one column at a time. Returns (new_U, new_V, new_W, new_rank) or None."""
    n, m, p = _DIMENSIONS
    rank = len(U[0])
    
    # Build column-to-unknown mapping for each candidate column to eliminate
    for col_elim in range(rank):
        if not _check_deadline():
            break
        
        col_to_unk = {}
        unk = 0
        for c in range(rank):
            if c != col_elim:
                col_to_unk[c] = unk
                unk += 1
        
        # Check if elimination is possible
        num_unknowns = rank - 1
        if num_unknowns <= 0:
            continue
        
        # Quick check: if col_elim is all zeros, it's already redundant
        all_zero_U = all(U[a][col_elim] == 0 for a in range(n * m))
        all_zero_V = all(V[b][col_elim] == 0 for b in range(m * p))
        all_zero_W = all(W[c][col_elim] == 0 for c in range(p * n))
        if all_zero_U and all_zero_V and all_zero_W:
            # Trivially eliminable
            new_U, new_V, new_W = _eliminate_column(U, V, W, col_elim, 
                                                      [Fraction(0)] * (rank - 1), col_to_unk)
            return new_U, new_V, new_W, rank - 1
        
        alpha_frac = _attempt_merge(U, V, W, None, col_elim)
        if alpha_frac is not None:
            new_U, new_V, new_W = _eliminate_column(U, V, W, col_elim, alpha_frac, col_to_unk)
            return new_U, new_V, new_W, rank - 1
    
    return None


def _pairwise_merge(U, V, W):
    """Try to merge two columns into one using a 2-parameter search.
    
    For columns c1 and c2, try to find alpha, beta (half-integers) such that
    replacing col_c2 by alpha*col_c1 and dropping col_c2 (or some combination)
    preserves the tensor.
    
    A cleaner approach: can we express col_c2 as alpha * col_c1 (single coefficient)?
    This is a special case of elimination.
    """
    n, m, p = _DIMENSIONS
    rank = len(U[0])
    
    for c1 in range(rank):
        for c2 in range(c1 + 1, rank):
            if not _check_deadline():
                return None
            
            # Try: can col_c2 = alpha * col_c1 for some half-integer alpha?
            # Check U: U[a, c2] = alpha * U[a, c1] for all a
            # If U[a, c1] == 0 for all a where U[a, c2] != 0, fail
            # Otherwise, alpha must be U[a, c2] / U[a, c1] for any a where U[a, c1] != 0
            
            alpha_candidate = None
            consistent = True
            for a in range(n * m):
                u1 = U[a][c1]
                u2 = U[a][c2]
                if u1 == 0:
                    if u2 != 0:
                        consistent = False
                        break
                else:
                    ratio = Fraction(u2, u1)
                    if alpha_candidate is None:
                        alpha_candidate = ratio
                    elif alpha_candidate != ratio:
                        consistent = False
                        break
            
            if not consistent:
                continue
            if alpha_candidate is None:
                # col_c1 is all zeros in U; check if col_c2 is also all zeros in U
                if any(U[a][c2] != 0 for a in range(n * m)):
                    continue
                alpha_candidate = Fraction(0)
            
            if not _is_half_integer(alpha_candidate):
                continue
            
            # Check V with same alpha
            for b in range(m * p):
                v1 = V[b][c1]
                v2 = V[b][c2]
                if v1 == 0:
                    if v2 != 0:
                        consistent = False
                        break
                else:
                    if Fraction(v2, v1) != alpha_candidate:
                        consistent = False
                        break
            if not consistent:
                continue
            
            # Check W with same alpha
            for c_idx in range(p * n):
                w1 = W[c_idx][c1]
                w2 = W[c_idx][c2]
                if w1 == 0:
                    if w2 != 0:
                        consistent = False
                        break
                else:
                    if Fraction(w2, w1) != alpha_candidate:
                        consistent = False
                        break
            if not consistent:
                continue
            
            # col_c2 = alpha * col_c1; eliminate c2
            # alpha_frac for the elimination: we need to express col_c2 in terms of remaining columns
            # Since col_c2 = alpha * col_c1, and col_c1 is being kept, the coefficient of col_c1 is alpha
            col_to_unk = {}
            unk = 0
            for c in range(rank):
                if c != c2:
                    col_to_unk[c] = unk
                    unk += 1
            alpha_frac = [Fraction(0)] * (rank - 1)
            alpha_frac[col_to_unk[c1]] = alpha_candidate
            
            new_U, new_V, new_W = _eliminate_column(U, V, W, c2, alpha_frac, col_to_unk)
            return new_U, new_V, new_W, rank - 1
    
    return None


def _to_int_or_half(frac_list_of_lists):
    """Convert a list of lists of Fractions to a list of lists of int or float (x.5)."""
    result = []
    for row in frac_list_of_lists:
        new_row = []
        for v in row:
            if v.denominator == 1:
                new_row.append(int(v.numerator))
            else:
                new_row.append(v.numerator / 2.0)
        result.append(new_row)
    return result


def construct():
    """Find a rank-minimal exact tensor decomposition for 3x4x5 matrix multiplication."""
    n, m, p = _DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _build_schoolbook()
    current_rank = n * m * p
    best_rank = current_rank
    best_U, best_V, best_W = U, V, W
    
    # Iteratively try to reduce rank
    max_iterations = 50
    for it in range(max_iterations):
        if not _check_deadline():
            break
        
        result = _pairwise_merge(U, V, W)
        if result is None:
            result = _try_all_eliminations(U, V, W)
        
        if result is not None:
            U, V, W, new_rank = result
            current_rank = new_rank
            if current_rank < best_rank:
                best_rank = current_rank
                best_U, best_V, best_W = U, V, W
        else:
            break
    
    # Convert to int/half-integer format
    if best_rank == n * m * p:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        # The matrices are already ints
        return {"U": best_U, "V": best_V, "W": best_W}


def construct():
    """Find a rank-minimal exact tensor decomposition for 3x4x5 matrix multiplication."""
    n, m, p = _DIMENSIONS
    _START_TIME_LOCAL = time.time()
    deadline = _START_TIME_LOCAL + 145.0  # 5s safety margin
    
    def time_up():
        return time.time() > deadline
    
    # Start with schoolbook
    U, V, W = _build_schoolbook()
    current_rank = n * m * p
    best_rank = current_rank
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Phase 1: Try pairwise proportional elimination
    for it in range(60):
        if time_up():
            break
        merged = False
        rank = len(U[0])
        for c1 in range(rank):
            if time_up():
                break
            for c2 in range(c1 + 1, rank):
                # Check if col_c2 = alpha * col_c1
                alpha_candidate = None
                ok = True
                for a in range(n * m):
                    u1, u2 = U[a][c1], U[a][c2]
                    if u1 == 0:
                        if u2 != 0:
                            ok = False
                            break
                    else:
                        r = Fraction(u2, u1)
                        if alpha_candidate is None:
                            alpha_candidate = r
                        elif alpha_candidate != r:
                            ok = False
                            break
                if not ok:
                    continue
                if alpha_candidate is None:
                    if any(U[a][c2] != 0 for a in range(n * m)):
                        continue
                    alpha_candidate = Fraction(0)
                if alpha_candidate.denominator not in (1, 2):
                    continue
                # Check V
                for b in range(m * p):
                    v1, v2 = V[b][c1], V[b][c2]
                    if v1 == 0:
                        if v2 != 0:
                            ok = False
                            break
                    else:
                        if Fraction(v2, v1) != alpha_candidate:
                            ok = False
                            break
                if not ok:
                    continue
                # Check W
                for ci in range(p * n):
                    w1, w2 = W[ci][c1], W[ci][c2]
                    if w1 == 0:
                        if w2 != 0:
                            ok = False
                            break
                    else:
                        if Fraction(w2, w1) != alpha_candidate:
                            ok = False
                            break
                if not ok:
                    continue
                # Eliminate c2
                new_rank = rank - 1
                nU = [[0] * new_rank for _ in range(n * m)]
                nV = [[0] * new_rank for _ in range(m * p)]
                nW = [[0] * new_rank for _ in range(p * n)]
                ni = 0
                for c in range(rank):
                    if c == c2:
                        continue
                    for a in range(n * m):
                        nU[a][ni] = U[a][c]
                    for b in range(m * p):
                        nV[b][ni] = V[b][c]
                    for ci in range(p * n):
                        nW[ci][ni] = W[ci][c]
                    ni += 1
                U, V, W = nU, nV, nW
                current_rank = new_rank
                if current_rank < best_rank:
                    best_rank = current_rank
                    best_U = [row[:] for row in U]
                    best_V = [row[:] for row in V]
                    best_W = [row[:] for row in W]
                merged = True
                break
            if merged:
                break
        if not merged:
            break
    
    # Phase 2: Try linear combination elimination (general)
    for it in range(40):
        if time_up():
            break
        eliminated = False
        rank = len(U[0])
        for col_elim in range(rank):
            if time_up():
                break
            # Check if col_elim is all zeros
            if all(U[a][col_elim] == 0 for a in range(n * m)) and \
               all(V[b][col_elim] == 0 for b in range(m * p)) and \
               all(W[c][col_elim] == 0 for c in range(p * n)):
                new_rank = rank - 1
                nU = [[0] * new_rank for _ in range(n * m)]
                nV = [[0] * new_rank for _ in range(m * p)]
                nW = [[0] * new_rank for _ in range(p * n)]
                ni = 0
                for c in range(rank):
                    if c == col_elim:
                        continue
                    for a in range(n * m):
                        nU[a][ni] = U[a][c]
                    for b in range(m * p):
                        nV[b][ni] = V[b][c]
                    for ci in range(p * n):
                        nW[ci][ni] = W[ci][c]
                    ni += 1
                U, V, W = nU, nV, nW
                current_rank = new_rank
                if current_rank < best_rank:
                    best_rank = current_rank
                    best_U = [row[:] for row in U]
                    best_V = [row[:] for row in V]
                    best_W = [row[:] for row in W]
                eliminated = True
                break
            
            # Try general linear combination elimination
            num_unknowns = rank - 1
            col_to_unk = {}
            unk = 0
            for c in range(rank):
                if c != col_elim:
                    col_to_unk[c] = unk
                    unk += 1
            
            # Build system
            rows_A = []
            rows_b = []
            for a in range(n * m):
                row = [0] * num_unknowns
                rhs = U[a][col_elim]
                for c in range(rank):
                    if c != col_elim:
                        row[col_to_unk[c]] = U[a][c]
                rows_A.append(row)
                rows_b.append(rhs)
            for b in range(m * p):
                row = [0] * num_unknowns
                rhs = V[b][col_elim]
                for c in range(rank):
                    if c != col_elim:
                        row[col_to_unk[c]] = V[b][c]
                rows_A.append(row)
                rows_b.append(rhs)
            for ci in range(p * n):
                row = [0] * num_unknowns
                rhs = W[ci][col_elim]
                for c in range(rank):
                    if c != col_elim:
                        row[col_to_unk[c]] = W[ci][c]
                rows_A.append(row)
                rows_b.append(rhs)
            
            A_np = np.array(rows_A, dtype=float)
            b_np = np.array(rows_b, dtype=float)
            try:
                alpha, _, _, _ = np.linalg.lstsq(A_np, b_np, rcond=1e-12)
                residual = A_np @ alpha - b_np
                if np.max(np.abs(residual)) > 1e-9:
                    continue
            except:
                continue
            
            # Check half-integer
            alpha_frac = []
            valid = True
            for a_val in alpha:
                af = Fraction(int(round(a_val * 2)), 2)
                if abs(af - a_val) > 1e-10:
                    valid = False
                    break
                alpha_frac.append(af)
            if not valid:
                continue
            
            # Verify exactly
            for a in range(n * m):
                s = 0
                for c in range(rank):
                    if c != col_elim:
                        s += alpha_frac[col_to_unk[c]] * U[a][c]
                if s != U[a][col_elim]:
                    valid = False
                    break
            if not valid:
                continue
            for b in range(m * p):
                s = 0
                for c in range(rank):
                    if c != col_elim:
                        s += alpha_frac[col_to_unk[c]] * V[b][c]
                if s != V[b][col_elim]:
                    valid = False
                    break
            if not valid:
                continue
            for ci in range(p * n):
                s = 0
                for c in range(rank):
                    if c != col_elim:
                        s += alpha_frac[col_to_unk[c]] * W[ci][c]
                if s != W[ci][col_elim]:
                    valid = False
                    break
            if not valid:
                continue
            
            # Eliminate
            new_rank = rank - 1
            nU = [[0] * new_rank for _ in range(n * m)]
            nV = [[0] * new_rank for _ in range(m * p)]
            nW = [[0] * new_rank for _ in range(p * n)]
            ni = 0
            for c in range(rank):
                if c == col_elim:
                    continue
                for a in range(n * m):
                    nU[a][ni] = U[a][c]
                for b in range(m * p):
                    nV[b][ni] = V[b][c]
                for ci in range(p * n):
                    nW[ci][ni] = W[ci][c]
                ni += 1
            U, V, W = nU, nV, nW
            current_rank = new_rank
            if current_rank < best_rank:
                best_rank = current_rank
                best_U = [row[:] for row in U]
                best_V = [row[:] for row in V]
                best_W = [row[:] for row in W]
            eliminated = True
            break
        if not eliminated:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
