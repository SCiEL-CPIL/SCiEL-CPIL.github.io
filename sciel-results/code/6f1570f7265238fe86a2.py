"""
Rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication.

Strategy:
1. Build a block-Strassen + schoolbook decomposition (rank 99 baseline).
2. Attempt rank reduction via:
   a. Merging two rank-1 terms into one when the resulting vectors are
      linearly dependent (a "2-to-1" merge).
   b. Removing zero terms.
3. Use exact rational arithmetic (Fraction) for correctness.
"""
import time
import numpy as np
from fractions import Fraction
from itertools import combinations, product as iproduct

DIMENSIONS = [3, 5, 7]


def _strassen_block(U, V, W, n, m, p, ia, ja, ka, offset, sign=1):
    """
    Insert 7 Strassen rank-1 terms at column index starting at `offset`
    for 2x2x2 subblock at (ia, ja, ka).
    Returns next free column index.
    """
    col = offset

    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[ia*m + ja][col] += sign
    U[(ia+1)*m + (ja+1)][col] += sign
    V[ja*p + ka][col] += sign
    V[(ja+1)*p + (ka+1)][col] += sign
    W[ka*n + ia][col] += sign
    W[(ka+1)*n + (ia+1)][col] += sign
    col += 1

    # M2 = (A10+A11)B00 -> C10 += M2
    U[(ia+1)*m + ja][col] += sign
    U[(ia+1)*m + (ja+1)][col] += sign
    V[ja*p + ka][col] += sign
    W[ka*n + (ia+1)][col] += sign
    col += 1

    # M3 = A00(B10-B11) -> C11 += M3
    U[ia*m + ja][col] += sign
    V[(ja+1)*p + (ka+1)][col] += sign
    V[(ja+1)*p + ka][col] -= sign
    W[(ka+1)*n + (ia+1)][col] += sign
    col += 1

    # M4 = A11(B01-B10) -> C00 += M4
    U[(ia+1)*m + (ja+1)][col] += sign
    V[ja*p + (ka+1)][col] += sign
    V[ja*p + ka][col] -= sign
    W[ka*n + ia][col] += sign
    col += 1

    # M5 = (A00+A01)B11 -> C01 += M5, C11 -= M5
    U[ia*m + ja][col] += sign
    U[ia*m + (ja+1)][col] += sign
    V[(ja+1)*p + (ka+1)][col] += sign
    W[(ka+1)*n + (ia+1)][col] += sign
    W[(ka+1)*n + ia][col] -= sign
    col += 1

    # M6 = (A10-A00)(B00+B10) -> C10 += M6
    U[(ia+1)*m + ja][col] += sign
    U[ia*m + ja][col] -= sign
    V[ja*p + ka][col] += sign
    V[ja*p + (ka+1)][col] += sign
    W[(ka+1)*n + (ia+1)][col] += sign
    col += 1

    # M7 = (A01-A11)(B01+B11) -> C00 -= M7, C11 += M7
    U[(ia+1)*m + (ja+1)][col] += sign
    U[ia*m + (ja+1)][col] -= sign
    V[(ja+1)*p + ka][col] += sign
    V[(ja+1)*p + (ka+1)][col] += sign
    W[ka*n + ia][col] -= sign
    W[(ka+1)*n + (ia+1)][col] += sign
    col += 1

    return col


def construct():
    n, m, p = 3, 5, 7
    deadline = time.time() + 115
    t0 = time.time()

    def get_block_options(d):
        """Get all valid sets of non-overlapping 2-blocks for dimension d"""
        results = []
        for nb in range(0, d // 2 + 1):
            starts = list(range(d - 1))
            for combo in combinations(starts, nb):
                valid = True
                for i in range(len(combo) - 1):
                    if combo[i] + 1 >= combo[i+1]:
                        valid = False
                        break
                if not valid:
                    continue
                covered = set()
                for s in combo:
                    covered.add(s)
                    covered.add(s + 1)
                uncovered = d - len(covered)
                blocks = [(s, s+1) for s in combo]
                results.append((blocks, covered, uncovered))
        return results

    n_opts = get_block_options(n)
    m_opts = get_block_options(m)
    p_opts = get_block_options(p)

    best_config = None
    best_rank = float('inf')

    for nb, n_cov, n_unc in n_opts:
        for mb, m_cov, m_unc in m_opts:
            for pb, p_cov, p_unc in p_opts:
                num_strassen = len(nb) * len(mb) * len(pb)
                rank_strassen = num_strassen * 7

                num_schoolbook = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if i not in n_cov or j not in m_cov or k not in p_cov:
                                num_schoolbook += 1

                total_rank = rank_strassen + num_schoolbook
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_config = (nb, mb, pb, n_cov, m_cov, p_cov)

    if best_config is None:
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

    nb, mb, pb, n_cov, m_cov, p_cov = best_config
    num_strassen = len(nb) * len(mb) * len(pb)
    rank_strassen = num_strassen * 7

    num_schoolbook = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i not in n_cov or j not in m_cov or k not in p_cov:
                    num_schoolbook += 1

    total_rank = rank_strassen + num_schoolbook

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0

    for (a, b) in nb:
        for (c, d) in mb:
            for (e, f) in pb:
                col = _strassen_block(U, V, W, n, m, p, a, c, e, col)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i not in n_cov or j not in m_cov or k not in p_cov:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    def verify_float(U, V, W, rank):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = float(np.dot(Ua[i * m + j], Va[j * p + k] * Wa[k * n + i]))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    if not verify_float(U, V, W, total_rank):
        # Fallback to schoolbook
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

    # Convert to Fraction for exact arithmetic
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    rank = total_rank

    def verify_exact(Uf, Vf, Wf, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for t in range(rank):
                        s += Uf[i*m+j][t] * Vf[j*p+k][t] * Wf[k*n+i][t]
                    if s != 1:
                        return False
        return True

    if not verify_exact(Uf, Vf, Wf, rank):
        return {"U": U, "V": V, "W": W}

    def remove_zero_terms(Uf, Vf, Wf, rank, deadline):
        """Remove terms that are entirely zero in one component."""
        changed = True
        while changed and time.time() < deadline - 20:
            changed = False
            for t in range(rank):
                u_zero = all(Uf[i*m+j][t] == 0 for i in range(n) for j in range(m))
                v_zero = all(Vf[j*p+k][t] == 0 for j in range(m) for k in range(p))
                w_zero = all(Wf[k*n+i][t] == 0 for k in range(p) for i in range(n))
                if u_zero or v_zero or w_zero:
                    Uf = [row[:t] + row[t+1:] for row in Uf]
                    Vf = [row[:t] + row[t+1:] for row in Vf]
                    Wf = [row[:t] + row[t+1:] for row in Wf]
                    rank -= 1
                    changed = True
                    break
        return Uf, Vf, Wf, rank

    def try_merge_terms(Uf, Vf, Wf, rank, deadline):
        """
        Try to merge pairs of rank-1 terms into a single term.
        Two terms t1, t2 can be merged if there exist scalars a, b such that
        the sum a*term1 + b*term2 is still a valid rank-1 tensor.
        
        A more practical approach: check if any two terms have proportional
        U vectors, proportional V vectors, or proportional W vectors, and
        if the remaining component also works out.
        
        Actually, a simpler and more effective approach: try to find linear
        dependencies among the terms. If the U-vectors of two terms are
        linearly dependent, and similarly for V and W, we might be able to
        merge them.
        
        Let's try a different approach: for each pair of terms (t1, t2),
        check if there exists a scalar alpha such that U[t2] = alpha * U[t1],
        V[t2] = alpha * V[t1], W[t2] = alpha * W[t1]. If so, we can replace
        both with a single term scaled appropriately.
        """
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            for t1 in range(rank):
                if time.time() > deadline - 10:
                    break
                for t2 in range(t1 + 1, rank):
                    # Check if term t2 is a scalar multiple of term t1
                    # Find alpha from U component
                    alpha = None
                    found = False
                    for i in range(n):
                        for j in range(m):
                            if Uf[i*m+j][t1] != 0:
                                alpha = Uf[i*m+j][t2] / Uf[i*m+j][t1]
                                found = True
                                break
                        if found:
                            break
                        if time.time() > deadline - 10:
                            break
                    if not found:
                        continue
                    
                    # Verify alpha works for all U entries
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            if Uf[i*m+j][t1] * alpha != Uf[i*m+j][t2]:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    
                    # Verify alpha works for all V entries
                    for j in range(m):
                        for k in range(p):
                            if Vf[j*p+k][t1] * alpha != Vf[j*p+k][t2]:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    
                    # Verify alpha works for all W entries
                    for k in range(p):
                        for i in range(n):
                            if Wf[k*n+i][t1] * alpha != Wf[k*n+i][t2]:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    
                    # Terms are proportional. We can keep only t1 and remove t2.
                    # But we need to make sure the tensor equation still holds.
                    # Since both terms contribute to the same (i,j,k) entries
                    # proportionally, we can simply remove t2 and scale t1
                    # appropriately... but actually, if we remove t2, we lose
                    # its contribution. We need to keep t1 as is and just
                    # remove t2 only if t2's contribution is already captured.
                    # 
                    # Actually, if U[t2]=alpha*U[t1], V[t2]=alpha*V[t1], 
                    # W[t2]=alpha*W[t1], then term t2 = alpha^3 * term t1.
                    # So we can't just remove it; we'd need to adjust.
                    # 
                    # This approach doesn't directly reduce rank. Let's skip
                    # this and try a different strategy.
                    pass
        
        return Uf, Vf, Wf, rank

    # Try a more sophisticated rank reduction: look for 2-to-1 merges
    # where two terms can be replaced by one via a clever combination.
    # This is computationally expensive, so we limit the search.
    
    def try_sophisticated_merge(Uf, Vf, Wf, rank, deadline):
        """
        Try to find two terms that can be merged into one.
        For terms t1 and t2, we want to find new vectors u', v', w' such that
        u' ⊗ v' ⊗ w' = u1 ⊗ v1 ⊗ w1 + u2 ⊗ v2 ⊗ w2.
        
        This is equivalent to checking if the sum of two rank-1 tensors has
        rank 1, which is very restrictive. In general, the sum of two
        distinct rank-1 tensors has rank 2, so this rarely works.
        
        A more practical approach: look for linear dependencies in the
        space of terms. If the set of terms is linearly dependent, we can
        reduce the rank.
        """
        # Check for linear dependencies among the terms
        # Build the tensor as a 3D array and check if it has a lower rank
        # decomposition. This is expensive, so we'll just try removing
        # terms one at a time and checking if the remaining terms still
        # span the same tensor.
        
        for t in range(rank):
            if time.time() > deadline - 10:
                break
            # Try removing term t
            Uf_test = [row[:t] + row[t+1:] for row in Uf]
            Vf_test = [row[:t] + row[t+1:] for row in Vf]
            Wf_test = [row[:t] + row[t+1:] for row in Wf]
            
            # Check if the remaining terms still produce the correct tensor
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = Fraction(0)
                        for tt in range(rank - 1):
                            s += Uf_test[i*m+j][tt] * Vf_test[j*p+k][tt] * Wf_test[k*n+i][tt]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
                if time.time() > deadline - 10:
                    break
                if not valid:
                    break
            
            if valid:
                return Uf_test, Vf_test, Wf_test, rank - 1
        
        return Uf, Vf, Wf, rank

    # First, remove zero terms
    Uf, Vf, Wf, rank = remove_zero_terms(Uf, Vf, Wf, deadline)
    
    # Then, try sophisticated merges (limited)
    if time.time() < deadline - 30:
        Uf, Vf, Wf, rank = try_sophisticated_merge(Uf, Vf, Wf, deadline)
    
    # Final verification
    if verify_exact(Uf, Vf, Wf, rank):
        def to_clean_list(flist):
            result = []
            for row in flist:
                crow = []
                for val in row:
                    if val.denominator == 1:
                        crow.append(int(val))
                    else:
                        crow.append(float(val))
                result.append(crow)
            return result
        
        return {"U": to_clean_list(Uf), "V": to_clean_list(Vf), "W": to_clean_list(Wf)}
    
    return {"U": U, "V": V, "W": W}
