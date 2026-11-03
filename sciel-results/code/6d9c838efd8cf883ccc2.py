import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Strassen M_{2,2,2}, rank 7."""
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]

    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1

    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1

    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[2][3] = 1

    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1

    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return {"U": U, "V": V, "W": W}


def _decomp_4x4x4():
    """M_{4,4,4} via 2x2 Strassen blocks. Rank 49."""
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]

    inner = _strassen_2x2()

    outer_A = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [0, 0, 1, -1], [0, 1, 0, -1]
    ]
    outer_B = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, -1],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    outer_C = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [-1, 0, 1, 0],
        [1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]

    for t in range(7):
        col_offset = t * 7
        for r in range(7):
            col = col_offset + r
            for li in range(2):
                for lj in range(2):
                    b = li * 2 + lj
                    coeff_U = outer_A[t][b] * inner["U"][li * 2 + lj][r]
                    for ei in range(2):
                        for ej in range(2):
                            g_i = 2 * (b // 2) + ei
                            g_j = 2 * (b % 2) + ej
                            U[g_i * 4 + g_j][col] = coeff_U

                    coeff_V = outer_B[t][b] * inner["V"][li * 2 + lj][r]
                    for ei in range(2):
                        for ej in range(2):
                            g_j_B = 2 * (b // 2) + ei
                            g_k_B = 2 * (b % 2) + ej
                            V[g_j_B * 4 + g_k_B][col] = coeff_V

                    coeff_W = outer_C[t][b] * inner["W"][li * 2 + lj][r]
                    for ei in range(2):
                        for ej in range(2):
                            g_k_C = 2 * (b // 2) + ei
                            g_i_C = 2 * (b % 2) + ej
                            W[g_k_C * 4 + g_i_C][col] = coeff_W

    return {"U": U, "V": V, "W": W}


def _schoolbook(n, m, p):
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


def _combine_decomps(d1, d2, n, m, p1, p2):
    r1 = len(d1["U"][0])
    r2 = len(d2["U"][0])
    r = r1 + r2

    U = [row + [0] * r2 for row in d1["U"]]

    V = [[0] * r for _ in range(m * (p1 + p2))]
    for j in range(m):
        for k in range(p1):
            V[j * (p1 + p2) + k] = d1["V"][j * p1 + k][:r1] + [0] * r2
        for k in range(p2):
            V[j * (p1 + p2) + p1 + k] = [0] * r1 + d2["V"][j * p2 + k]

    W = [[0] * r for _ in range((p1 + p2) * n)]
    for k in range(p1):
        for i in range(n):
            W[k * n + i] = d1["W"][k * n + i][:r1] + [0] * r2
    for k in range(p2):
        for i in range(n):
            W[(p1 + k) * n + i] = [0] * r1 + d2["W"][k * n + i]

    return {"U": U, "V": V, "W": W}


def _verify_fast(U, V, W, n, m, p):
    """Exact tensor check: for all (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)

    U_r = U_arr.reshape(n, m, rank)
    V_r = V_arr.reshape(m, p, rank)
    W_r = W_arr.reshape(p, n, rank)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_r[i, j, :], V_r[j, k, :] * W_r[k, i, :])
                if abs(s - 1.0) > 1e-7:
                    return False
    return True


def _verify_exact(U, V, W, n, m, p):
    """Verify with exact rational arithmetic using fractions."""
    from fractions import Fraction
    rank = len(U[0])
    U_r = np.array(U, dtype=object).reshape(n, m, rank)
    V_r = np.array(V, dtype=object).reshape(m, p, rank)
    W_r = np.array(W, dtype=object).reshape(p, n, rank)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    u = U_r[i, j, r]
                    v = V_r[j, k, r]
                    w = W_r[k, i, r]
                    if u == 0 or v == 0 or w == 0:
                        continue
                    s += Fraction(u) * Fraction(v) * Fraction(w)
                if s != Fraction(1):
                    return False
    return True


def _reduce_rank(U, V, W, n, m, p, deadline):
    """
    Attempt to reduce rank by finding linear dependencies among rank-1 terms.
    
    Strategy: The decomposition is valid if for all (i,j,k):
      sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    We look for a set of indices R such that removing them still preserves
    the identity. We use the following approach:
    
    1. Check if any single term is redundant (removing it still gives correct result).
    2. If not, try pairs.
    
    For efficiency, we use the fact that if term r is redundant, then for all (i,j,k):
      U[i,j,r]*V[j,k,r]*W[k,i,r] = 0  (i.e., the term contributes nothing)
    OR the contributions cancel in a structured way.
    
    A more powerful approach: find the minimum subset of terms that spans the tensor.
    We use a greedy elimination: try removing each term and check if the remaining
    still decomposes correctly.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)

    # Precompute contributions of each term to each (i,j,k)
    # contribution[r][i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r]
    # This is expensive for large rank, but rank=114 and n*m*p=144, so 114*144 = 16416 values
    
    # Actually, let's think differently. The tensor T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # We want T[i,j,k] = 1 for all (i,j,k).
    
    # A term r is "zero" if U[i,j,r]*V[j,k,r]*W[k,i,r] = 0 for all (i,j,k).
    # That happens if for every (i,j,k), at least one of U[i,j,r], V[j,k,r], W[k,i,r] is 0.
    
    # More generally, we can check: is the tensor still correct if we zero out term r?
    # T' = T - U[:, :, r] * V[:, :, r] * W[:, :, r] (outer product structure)
    # We need T'[i,j,k] = 1 for all (i,j,k), i.e., T[i,j,k] - contrib_r[i,j,k] = 1
    # Since T[i,j,k] = 1, this means contrib_r[i,j,k] = 0 for all (i,j,k).
    
    # So: term r is removable iff U[i,j,r]*V[j,k,r]*W[k,i,r] = 0 for all (i,j,k).
    
    # Let's check this for each term.
    removable = []
    for r in range(rank):
        if time.time() > deadline:
            break
        u = U_arr[:, :, r]  # shape (n*m,)
        v = V_arr[:, :, r]  # shape (m*p,)
        w = W_arr[:, :, r]  # shape (p*n,)
        
        u_r = u.reshape(n, m)
        v_r = v.reshape(m, p)
        w_r = w.reshape(p, n)
        
        # Check if u_r[i,j] * v_r[j,k] * w_r[k,i] = 0 for all i,j,k
        # This is true iff: for all (i,j,k), at least one factor is zero.
        # Equivalently: the support of u_r, v_r, w_r have no "triangular" overlap.
        
        # Efficient check: for each i,j where u_r[i,j] != 0,
        # we need v_r[j,k]*w_r[k,i] = 0 for all k.
        # i.e., for fixed (i,j) with u_r[i,j]!=0:
        #   for all k: v_r[j,k] == 0 or w_r[k,i] == 0
        
        is_zero = True
        for i in range(n):
            if not is_zero:
                break
            for j in range(m):
                if not is_zero:
                    break
                if u_r[i, j] == 0:
                    continue
                for k in range(p):
                    if v_r[j, k] != 0 and w_r[k, i] != 0:
                        is_zero = False
                        break
        if is_zero:
            removable.append(r)

    # Remove all zero-contribution terms
    if removable:
        keep = [r for r in range(rank) if r not in set(removable)]
        U_new = U_arr[:, :, keep]
        V_new = V_arr[:, :, keep]
        W_new = W_arr[:, :, keep]
        
        # Convert back to list format
        U_list = U_new.reshape(-1, len(keep)).tolist()
        V_list = V_new.reshape(-1, len(keep)).tolist()
        W_list = W_new.reshape(-1, len(keep)).tolist()
        
        # Verify
        if _verify_fast(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}

    return {"U": U, "V": V, "W": W}


def _reduce_rank_greedy(U, V, W, n, m, p, deadline):
    """
    Greedy rank reduction: iteratively try to remove terms.
    A term r can be removed if the remaining terms still give T[i,j,k]=1.
    This is equivalent to: the "correction" from removing r must be zero everywhere.
    
    For a single term: contrib_r[i,j,k] = U[i,j,r]*V[j,k,r]*W[k,i,r] must be 0 for all (i,j,k).
    
    For multiple terms, we can try: can we replace a set of terms with fewer?
    This is harder. Let's stick with single-term elimination for now,
    but do it iteratively (removing one may make another removable).
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    n_rows_U = n * m
    n_rows_V = m * p
    n_rows_W = p * n
    
    current_U = U_arr.copy()
    current_V = V_arr.copy()
    current_W = W_arr.copy()
    current_rank = rank
    
    # Iteratively remove zero-contribution terms
    changed = True
    while changed and time.time() < deadline:
        changed = False
        removable = []
        for r in range(current_rank):
            if time.time() > deadline:
                break
            u = current_U[:, r].reshape(n, m)
            v = current_V[:, r].reshape(m, p)
            w = current_W[:, r].reshape(p, n)
            
            is_zero = True
            for i in range(n):
                if not is_zero:
                    break
                for j in range(m):
                    if not is_zero:
                        break
                    if u[i, j] == 0:
                        continue
                    for k in range(p):
                        if v[j, k] != 0 and w[k, i] != 0:
                            is_zero = False
                            break
            if is_zero:
                removable.append(r)
        
        if removable:
            keep = [r for r in range(current_rank) if r not in set(removable)]
            current_U = current_U[:, keep]
            current_V = current_V[:, keep]
            current_W = current_W[:, keep]
            current_rank = len(keep)
            changed = True
    
    if current_rank < rank:
        U_list = current_U.reshape(-1, current_rank).tolist()
        V_list = current_V.reshape(-1, current_rank).tolist()
        W_list = current_W.reshape(-1, current_rank).tolist()
        
        if _verify_fast(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return {"U": U, "V": V, "W": W}


def _try_rank_reduction_via_subspace(U, V, W, n, m, p, deadline):
    """
    More sophisticated: use the fact that the decomposition corresponds to a
    bilinear map. We can try to find a lower-rank representation by
    exploiting linear structure.
    
    Key insight: If we have a decomposition of rank r, we can try to find
    a decomposition of rank r-1 by checking if one of the rank-1 tensors
    is in the span of the others (in the tensor product space).
    
    The rank-1 tensor for term r is: T_r = u_r ⊗ v_r ⊗ w_r (as a 3-tensor)
    where u_r[i,j] = U[i*m+j, r], v_r[j,k] = V[j*p+k, r], w_r[k,i] = W[k*n+i, r]
    
    We need: sum_r T_r = T_matmul (the matrix multiplication tensor)
    
    If T_r = sum_{s != r} c_s T_s for some coefficients, then we can remove r.
    But this is a tensor equation, hard to solve directly.
    
    Alternative: use the "border rank" or "approximate rank" ideas, but we need exact.
    
    Let's try a different approach: use the structure of the problem.
    For M_{4,4,9}, we can decompose as M_{4,4,8} ⊕ M_{4,4,1}.
    M_{4,4,8} = M_{4,4,4} ⊕ M_{4,4,4}, each with rank 49 (Strassen-based).
    M_{4,4,1} has rank 16 (schoolbook).
    Total: 49 + 49 + 16 = 114.
    
    Can we do better? The theoretical lower bound for M_{4,4,9} is at least 4*4*9/2 = 72 (from rank bounds).
    Known best for M_{4,4,4} is 49 (Strassen). For M_{4,4,8}, it might be less than 98.
    
    Actually, let's try a different decomposition of M_{4,4,9}:
    Split as M_{4,4,3} ⊕ M_{4,4,3} ⊕ M_{4,4,3}.
    M_{4,4,3} can be decomposed using Strassen-like methods.
    
    Or: M_{4,4,9} = M_{4,4,4} ⊕ M_{4,4,4} ⊕ M_{4,4,1} (current approach, rank 114)
    vs M_{4,4,9} = M_{4,4,3} ⊕ M_{4,4,3} ⊕ M_{4,4,3}
    
    For M_{4,4,3}: we can use a 2x2 Strassen block for the 4x4 part, but the 3 columns
    make it tricky. Let's try: treat it as M_{4,4,2} ⊕ M_{4,4,1}.
    M_{4,4,2}: rank ? Using Strassen for 4x4 with 2 columns...
    
    Actually, let's try yet another approach: use the Winograd method or
    other known fast matrix multiplication for non-square cases.
    
    For now, let's focus on what we can do: the current 114-rank decomposition.
    Let's see if we can find a better one by trying different block splits.
    """
    # Try different splits of the 9 columns
    # Option 1: 4+4+1 (current, rank 49+49+16=114)
    # Option 2: 3+3+3
    # Option 3: 2+2+2+2+1
    # Option 4: 4+5
    # Option 5: 3+6
    # Option 6: 2+7
    # Option 7: 1+8
    
    # For each split, we need decompositions of M_{4,4,k} for various k.
    # M_{4,4,k} for k=1: rank 16 (schoolbook)
    # M_{4,4,k} for k=2: ?
    # M_{4,4,k} for k=3: ?
    # M_{4,4,k} for k=4: rank 49 (Strassen)
    # M_{4,4,k} for k=5: ?
    # etc.
    
    # Let's try to construct M_{4,4,2} and M_{4,4,3} using Strassen-like methods.
    
    # M_{4,4,2}: multiply 4x4 by 4x2.
    # We can view this as two M_{4,4,1} problems, rank 16+16=32.
    # Or use Strassen: split 4x4 into 2x2 blocks, but the B matrix is 4x2.
    # B = [[B00, B01], [B10, B11]] where each is 2x1.
    # Strassen for M_{2,2,1}: rank 4 (schoolbook) or can we do better?
    # M_{2,2,1}: multiply 2x2 by 2x1. Rank is at most 4.
    # Can we do rank 3? Let's check.
    # The tensor M_{2,2,1} has rank 4 (it's essentially two independent dot products of length 2,
    # but with the constraint that they share structure... actually no).
    # C = A*B where A is 2x2, B is 2x1.
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # This is a 2x2x1 tensor. Its rank is 4 (each output entry needs 2 multiplications,
    # and they're independent). Actually, let me think again.
    # The rank-1 tensors are u⊗v⊗w where u is 4-dim (A entries), v is 2-dim (B entries), w is 2-dim (C entries).
    # We need sum_r u_r⊗v_r⊗w_r = T where T[i,j,k] = 1 if k==0 and i==j, or k==1 and i==j+1... 
    # Wait, let me be more careful.
    # A = [[a00, a01], [a10, a11]], B = [[b0], [b1]], C = [[c0], [c1]]
    # c0 = a00*b0 + a01*b1
    # c1 = a10*b0 + a11*b1
    # The tensor T has T[a00,b0,c0]=1, T[a01,b1,c0]=1, T[a10,b0,c1]=1, T[a11,b1,c1]=1.
    # This is a 4x2x2 tensor. Its rank is 4 (each of the 4 non-zero entries is in a different
    # "fiber" and they're all needed).
    # So M_{2,2,1} has rank 4.
    # Then M_{4,4,2} using 2x2 Strassen blocks: 7 * 4 = 28? No, that's not right.
    # Strassen for M_{2,2,2} uses 7 multiplications. For M_{2,2,1}, we can't use Strassen directly.
    # We'd use schoolbook: 4 multiplications per 2x2 block, so for 4 blocks: 16 total.
    # Or: use the structure more cleverly.
    
    # Let me try a different approach for M_{4,4,2}:
    # View it as M_{4,4,1} ⊕ M_{4,4,1}, rank 16+16=32.
    # Or: use a "half-Strassen" where we exploit the 2-column structure.
    
    # Actually, for M_{4,4,2}, we can do:
    # Split A into 2x2 blocks, B into 2x1 blocks (4 of them).
    # Use Strassen-like formulas but adapted for the 2x1 case.
    # This is getting complex. Let me just try the straightforward splits.
    
    # For the purpose of this program, let's try a few different constructions
    # and pick the one with lowest rank that passes verification.
    
    results = []
    
    # Construction 1: 4+4+1 (current)
    if time.time() < deadline:
        d1 = _decomp_4x4x4()
        d2 = _decomp_4x4x4()
        d3 = _schoolbook(4, 4, 1)
        d12 = _combine_decomps(d1, d2, 4, 4, 4, 4)
        d_total = _combine_decomps(d12, d3, 4, 4, 8, 1)
        r = len(d_total["U"][0])
        if _verify_fast(d_total["U"], d_total["V"], d_total["W"], 4, 4, 9):
            results.append((r, d_total))
    
    # Construction 2: 3+3+3
    # Need M_{4,4,3}. Let's build it as M_{4,4,2} ⊕ M_{4,4,1}.
    # M_{4,4,2} via schoolbook: rank 32.
    # M_{4,4,1}: rank 16.
    # Total for M_{4,4,3}: 32+16=48.
    # Three of these: 48*3=144. Worse than 114.
    
    # Construction 3: Try M_{4,4,3} with a smarter method.
    # Use 2x2 Strassen for the 4x4 part, but with 3 columns.
    # This is non-standard. Let's skip for now.
    
    # Construction 4: 2+2+2+2+1
    # M_{4,4,2} via schoolbook: rank 32.
    # Four of these: 32*4=128, plus M_{4,4,1}=16. Total: 144. Worse.
    
    # Construction 5: 4+5
    # M_{4,4,5} = M_{4,4,4} ⊕ M_{4,4,1}, rank 49+16=65.
    # M_{4,4,4} ⊕ M_{4,4,5} = 49+65=114. Same as before.
    
    # So the best we can do with these methods is 114.
    # Let's try to reduce 114 by finding linear dependencies.
    
    if results:
        results.sort(key=lambda x: x[0])
        best_rank, best_decomp = results[0]
        # Try to reduce
        reduced = _reduce_rank_greedy(best_decomp["U"], best_decomp["V"], best_decomp["W"], 4, 4, 9, deadline)
        if len(reduced["U"][0]) < best_rank:
            return reduced
        return best_decomp
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Try the rank-reduction approach
    result = _try_rank_reduction_via_subspace(None, None, None, n, m, p, deadline)
    
    if result is not None:
        # Final verification
        if _verify_fast(result["U"], result["V"], result["W"], n, m, p):
            return result
    
    # Fallback to schoolbook
    if time.time() < deadline:
        sb = _schoolbook(n, m, p)
        if _verify_fast(sb["U"], sb["V"], sb["W"], n, m, p):
            return sb
    
    # Absolute fallback
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
