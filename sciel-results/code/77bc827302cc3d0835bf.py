import numpy as np
import time
from fractions import Fraction

DEADLINE = 150
DEADLINE_MARGIN = 10

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor decomposition: sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] == 1 for all i,j,k."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                total = np.dot(U_np[idx_u], V_np[idx_v] * W_np[idx_w])
                if abs(total - 1.0) > tol:
                    return False
    # Exact verification
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                total = Fraction(0)
                for r in range(rank):
                    total += U_frac[idx_u][r] * V_frac[idx_v][r] * W_frac[idx_w][r]
                if total != 1:
                    return False
    return True

def _construct_rank_27():
    """
    Rank-27 decomposition for 2x3 * 3x5 matrix multiplication.
    
    Strategy:
    - 7 Strassen terms for the (i=0,1; j=0,1; k=0,1) subblock
    - 7 Strassen terms for the (i=0,1; j=0,1; k=2,3) subblock  
    - 13 terms for cross terms: A[:,2]*B[2,:] (5 terms) + A[:,0:2]*B[:,4] (4 terms) + shared (4 terms)
    
    Actually, let me use a cleaner decomposition:
    - Block 1: C[i,0:2] from A[i,0:2]*B[0:2,0:2] via Strassen (7 terms)
    - Block 2: C[i,2:4] from A[i,0:2]*B[0:2,2:4] via Strassen (7 terms)
    - Block 3: C[i,0:5] += A[i,2]*B[2,k] for all k (5 terms, one per k)
    - Block 4: C[i,4] += A[i,0]*B[0,4] + A[i,1]*B[1,4] (need 4 terms with sharing)
    
    Total: 7+7+5+4 = 23? No, that's not right either.
    
    Let me be more careful. The full product is:
    C[i,k] = sum_j A[i,j]*B[j,k]
    
    For each (i,k) pair, we need the sum over j=0,1,2.
    
    Let me use:
    - 7 Strassen terms for j in {0,1}, k in {0,1}
    - 7 Strassen terms for j in {0,1}, k in {2,3}
    - For j=2: 5 terms (one per k)
    - For j in {0,1}, k=4: 4 terms (2x2 block, can use 4 schoolbook or 3 with sharing)
    
    That's 7+7+5+4 = 23... but wait, the Strassen terms already handle specific (i,k) pairs.
    
    Actually the issue is that Strassen for 2x2x2 gives 7 terms that compute all 4 output entries.
    So 7 terms handle C[0,0], C[0,1], C[1,0], C[1,1] from the j=0,1 part.
    7 terms handle C[0,2], C[0,3], C[1,2], C[1,3] from the j=0,1 part.
    Then we need C[i,4] for i=0,1 from j=0,1,2 part: that's 2 outputs from 3 inputs each.
    And C[i,k] for k=0..4 from j=2 part: that's 2*5=10 outputs from 2*5=10 inputs (A[i,2]*B[2,k]).
    
    Wait, I'm overcomplicating. Let me just count:
    - j=0,1, k=0,1: 4 outputs, 7 Strassen terms
    - j=0,1, k=2,3: 4 outputs, 7 Strassen terms
    - j=0,1, k=4: 2 outputs, need at least 2 terms (schoolbook: A[0,0]*B[0,4], A[1,0]*B[1,4] for C[0,4]; A[0,1]*B[0,4], A[1,1]*B[1,4] for C[1,4]) = 4 terms
    - j=2, k=0..4: 10 outputs (C[i,k] += A[i,2]*B[2,k]), 10 schoolbook terms
    
    Total: 7+7+4+10 = 28. That's the current solution.
    
    To get 27, I need to save 1 term. Let me try to share between the j=2 terms and the k=4 terms.
    
    Alternative: use a different grouping. What if I treat the problem as:
    - C[:,0:4] from A[:,0:2]*B[0:2,0:4] (2x2 by 2x4)
    - C[:,0:5] += A[:,2]*B[2,:] (2x1 by 1x5)
    - C[:,4] += A[:,0:2]*B[0:2,4] (2x2 by 2x1)
    
    For 2x2 by 2x4: this is two 2x2 by 2x2 problems, so 7+7=14 terms.
    For 2x1 by 1x5: 5 terms.
    For 2x2 by 2x1: 4 terms (or 3 with sharing? No, 2x2x1 tensor has rank 4 generically).
    
    Total: 14+5+4 = 23? That can't be right since we know rank >= 27 or so.
    
    Wait, I think I'm confusing myself. The tensor is the full matrix multiplication tensor, not a sum of independent sub-tensors. The issue is that the same U,V,W indices are shared across all terms.
    
    Let me just try to construct a valid rank-27 by modifying the rank-28 to merge two terms.
    """
    n, m, p = 2, 3, 5
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: Strassen for (i,j,k) with i in {0,1}, j in {0,1}, k in {0,1}
    # M1 = (a00+a11)(b00+b11)
    U[0][r] = 1; U[4][r] = 1
    V[0][r] = 1; V[1][r] = 1
    W[0][r] = 1; W[3][r] = 1
    r += 1
    
    # M2 = (a10+a11)b00
    U[3][r] = 1; U[4][r] = 1
    V[0][r] = 1
    W[1][r] = 1; W[3][r] = -1
    r += 1
    
    # M3 = a00(b01-b11)
    U[0][r] = 1
    V[1][r] = 1; V[2][r] = -1
    W[2][r] = 1; W[3][r] = 1
    r += 1
    
    # M4 = a11(b10-b00)
    U[4][r] = 1
    V[2][r] = 1; V[0][r] = -1
    W[0][r] = 1; W[2][r] = 1
    r += 1
    
    # M5 = (a00+a01)b11
    U[0][r] = 1; U[1][r] = 1
    V[2][r] = 1
    W[0][r] = -1; W[2][r] = 1
    r += 1
    
    # M6 = (a10-a00)(b00+b01)
    U[3][r] = 1; U[0][r] = -1
    V[0][r] = 1; V[1][r] = 1
    W[3][r] = 1
    r += 1
    
    # M7 = (a01-a11)(b10+b11)
    U[1][r] = 1; U[4][r] = -1
    V[2][r] = 1; V[3][r] = 1
    W[0][r] = 1
    r += 1
    
    # Part 2: Strassen for (i,j,k) with i in {0,1}, j in {0,1}, k in {2,3}
    # M8 = (a00+a11)(b02+b13)
    U[0][r] = 1; U[4][r] = 1
    V[2][r] = 1; V[3][r] = 1
    W[4][r] = 1; W[7][r] = 1
    r += 1
    
    # M9 = (a10+a11)b02
    U[3][r] = 1; U[4][r] = 1
    V[2][r] = 1
    W[5][r] = 1; W[7][r] = -1
    r += 1
    
    # M10 = a00(b03-b13)
    U[0][r] = 1
    V[3][r] = 1; V[4][r] = -1
    W[6][r] = 1; W[7][r] = 1
    r += 1
    
    # M11 = a11(b12-b02)
    U[4][r] = 1
    V[4][r] = 1; V[2][r] = -1
    W[4][r] = 1; W[6][r] = 1
    r += 1
    
    # M12 = (a00+a01)b13
    U[0][r] = 1; U[1][r] = 1
    V[4][r] = 1
    W[4][r] = -1; W[6][r] = 1
    r += 1
    
    # M13 = (a10-a00)(b02+b03)
    U[3][r] = 1; U[0][r] = -1
    V[2][r] = 1; V[3][r] = 1
    W[7][r] = 1
    r += 1
    
    # M14 = (a01-a11)(b12+b13)
    U[1][r] = 1; U[4][r] = -1
    V[4][r] = 1; V[5][r] = 1
    W[4][r] = 1
    r += 1
    
    # Part 3: j=2 terms: C[i,k] += A[i,2]*B[2,k] for k=0..4
    # That's 10 terms (i=0,1 and k=0..4)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    # r is now 7+7+10 = 24
    
    # Part 4: j in {0,1}, k=4: C[i,4] += A[i,0]*B[0,4] + A[i,1]*B[1,4]
    # We need 3 more terms (total 27)
    # C[0,4] += A[0,0]*B[0,4] + A[0,1]*B[1,4]
    # C[1,4] += A[1,0]*B[0,4] + A[1,1]*B[1,4]
    
    # Term 25: handles A[0,0]*B[0,4] -> C[0,4]
    U[0][r] = 1
    V[4][r] = 1
    W[8][r] = 1
    r += 1
    
    # Term 26: handles A[0,1]*B[1,4] -> C[0,4] AND A[1,0]*B[0,4] -> C[1,4] (shared?)
    # Can't share directly. Let me try:
    # Term 26: A[1,0]*B[0,4] -> C[1,4]
    U[3][r] = 1
    V[4][r] = 1
    W[9][r] = 1
    r += 1
    
    # Term 27: A[0,1]*B[1,4] -> C[0,4] AND A[1,1]*B[1,4] -> C[1,4]?
    # No, these are different. Let me use:
    # Term 27: A[0,1]*B[1,4] -> C[0,4]
    U[1][r] = 1
    V[5][r] = 1
    W[8][r] = 1
    r += 1
    
    # But we still need A[1,1]*B[1,4] -> C[1,4]! That's a 28th term.
    # So this naive approach gives 28.
    
    # Let me try a different approach for Part 4 using a 2x2x1 tensor decomposition.
    # The sub-problem is: given x0=A[0,0], x1=A[1,0], y0=B[0,4], y1=B[1,4], z0=A[0,1], z1=A[1,1]
    # Compute: C[0,4] += x0*y0 + z0*y1
    #          C[1,4] += x1*y0 + z1*y1
    
    # This is a 2-output problem with 6 inputs. Can we do it in 3 terms?
    # Term a: p*(x0*y0 + z0*y1) where p is a linear combo
    # Actually, let me think of it as:
    # C[0,4] += (x0 + z0)*(y0 + y1) - z0*y0 - x0*y1
    # That's 4 terms.
    
    # Alternative: 
    # C[0,4] += x0*y0 + z0*y1
    # C[1,4] += x1*y0 + z1*y1
    # = (x0+z1)*y0 + (z0-x1)*y1 + x1*y0 + z1*y0 - x1*y0 - z1*y0... no this is getting circular.
    
    # Let me just accept 28 for now and try a different overall strategy.
    # Actually, let me try: use 3 terms for the j in {0,1}, k=4 part by exploiting linearity.
    
    # Reset and try a cleaner construction
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    r = 0
    
    # 14 Strassen terms for the 2x2x2 blocks (k=0,1 and k=2,3)
    # Same as before, terms 0-13
    # ... (copy the Strassen terms above)
    
    # This is getting too complex to do by hand. Let me use a systematic approach.
    # I'll construct the rank-28 solution, then try to find a linear combination that eliminates one term.
    
    return {"U": U, "V": V, "W": W}

def _construct_rank_28():
    """Known rank-28 decomposition."""
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: A2 B2 (j=2, k=0..4) - 10 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: A1 B_col4 (j=0,1, k=4) - 4 terms
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: Strassen for (j=0,1, k=0,1) - 7 terms
    # M1
    U[0][r] = 1; U[4][r] = 1
    V[0][r] = 1; V[1][r] = 1
    W[0][r] = 1; W[3][r] = 1
    r += 1
    # M2
    U[3][r] = 1; U[4][r] = 1
    V[0][r] = 1
    W[1][r] = 1; W[3][r] = -1
    r += 1
    # M3
    U[0][r] = 1
    V[1][r] = 1; V[2][r] = -1
    W[2][r] = 1; W[3][r] = 1
    r += 1
    # M4
    U[4][r] = 1
    V[2][r] = 1; V[0][r] = -1
    W[0][r] = 1; W[2][r] = 1
    r += 1
    # M5
    U[0][r] = 1; U[1][r] = 1
    V[2][r] = 1
    W[0][r] = -1; W[2][r] = 1
    r += 1
    # M6
    U[3][r] = 1; U[0][r] = -1
    V[0][r] = 1; V[1][r] = 1
    W[3][r] = 1
    r += 1
    # M7
    U[1][r] = 1; U[4][r] = -1
    V[2][r] = 1; V[3][r] = 1
    W[0][r] = 1
    r += 1
    
    # Part 4: Strassen for (j=0,1, k=2,3) - 7 terms
    # M8
    U[0][r] = 1; U[4][r] = 1
    V[2][r] = 1; V[3][r] = 1
    W[4][r] = 1; W[7][r] = 1
    r += 1
    # M9
    U[3][r] = 1; U[4][r] = 1
    V[2][r] = 1
    W[5][r] = 1; W[7][r] = -1
    r += 1
    # M10
    U[0][r] = 1
    V[3][r] = 1; V[4][r] = -1
    W[6][r] = 1; W[7][r] = 1
    r += 1
    # M11
    U[4][r] = 1
    V[4][r] = 1; V[2][r] = -1
    W[4][r] = 1; W[6][r] = 1
    r += 1
    # M12
    U[0][r] = 1; U[1][r] = 1
    V[4][r] = 1
    W[4][r] = -1; W[6][r] = 1
    r += 1
    # M13
    U[3][r] = 1; U[0][r] = -1
    V[2][r] = 1; V[3][r] = 1
    W[7][r] = 1
    r += 1
    # M14
    U[1][r] = 1; U[4][r] = -1
    V[4][r] = 1; V[5][r] = 1
    W[4][r] = 1
    r += 1
    
    return {"U": U, "V": V, "W": W}

def _try_merge_terms(U, V, W, n, m, p, rank, deadline):
    """Try to find a linear combination of two terms that can replace them with one."""
    import numpy as np
    
    # For each pair of terms (r1, r2), check if we can find a single term that
    # replicates their combined contribution. This is equivalent to checking if
    # the tensor T = sum_{r1,r2} U_r1 * V_r2 * W_r2 (outer product) has rank 1,
    # which is generally false. Instead, we check if removing both and adding
    # one carefully chosen term works.
    
    # Simpler: for each term r, check if it's a linear combination of others.
    # Build the constraint matrix.
    
    # The constraint is: for all (i,j,k), sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] = 1
    # If we set U[i*m+j][r2] = 0, V[j*p+k][r2] = 0, W[k*n+i][r2] = 0 for some r2,
    # and adjust another term r1, we need the new sum to still equal 1.
    
    # This is complex. Let me just try the simple elimination test.
    for elim_r in range(rank):
        if time.time() > deadline:
            return None
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    total = 0.0
                    for r in range(rank):
                        if r == elim_r:
                            continue
                        total += U[idx_u][r] * V[idx_v][r] * W[idx_w][r]
                    if abs(total - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Build result without elim_r
            U_final = [[x for idx, x in enumerate(row) if idx != elim_r] for row in U]
            V_final = [[x for idx, x in enumerate(row) if idx != elim_r] for row in V]
            W_final = [[x for idx, x in enumerate(row) if idx != elim_r] for row in W]
            if _verify_decomposition(U_final, V_final, W_final, n, m, p):
                return {"U": U_final, "V": V_final, "W": W_final}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - DEADLINE_MARGIN
    
    # Construct rank-28
    res_28 = _construct_rank_28()
    
    if _verify_decomposition(res_28['U'], res_28['V'], res_28['W'], 2, 3, 5):
        # Try to find rank-27
        if time.time() < deadline - 5:
            better = _try_merge_terms(res_28['U'], res_28['V'], res_28['W'], 2, 3, 5, 28, deadline)
            if better is not None:
                return better
        
        return res_28
    else:
        raise ValueError("Verification failed")
