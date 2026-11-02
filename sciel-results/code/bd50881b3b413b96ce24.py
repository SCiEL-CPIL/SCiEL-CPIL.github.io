"""Rewritten program targeting rank-10 exact decomposition for 2x2 x 2x3 matmul."""
import time
import numpy as np

DIMENSIONS = [2, 2, 3]

def _verify(U, V, W, n=2, m=2, p=3, tol=1e-10):
    """Verify that sum_r U[:,r] outer V[:,r] outer W[:,r] equals the matmul tensor."""
    rank = U[0].__len__() if isinstance(U[0], list) else U.shape[1]
    # Build tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    # Tensor T has shape (n*m, m*p, p*n)
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    # Expected: T[a,b,c] = 1 iff a=b=c=0 (i.e., the matmul tensor)
    # For 2x2 * 2x3: A is 2x2 (rows i=0,1; cols j=0,1), B is 2x3 (rows j=0,1; cols k=0,1,2)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor index: a = i*m + j (i in 0..n-1, j in 0..m-1), b = j*p + k (j in 0..m-1, k in 0..p-1), c = k*n + i (k in 0..p-1, i in 0..n-1)
    # T[a,b,c] = 1 iff (i,j),(j,k),(k,i) form a valid multiplication triple
    # i.e., a = i*m+j, b = j*p+k, c = k*n+i, with the same i,j,k
    expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                expected[a, b, c] = 1.0
    diff = np.max(np.abs(T - expected))
    return diff < tol, diff

def _rank11_solution():
    """The known rank-11 solution: Strassen for cols 0,1 + schoolbook for col 2."""
    n, m, p = 2, 2, 3
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen part for columns 0 and 1 of B
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    W[2][2] = 1; W[2][4] = 1
    W[1][1] = 1; W[1][3] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    # Schoolbook part for column 2 of B
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1
    
    U[1][8] = 1
    V[5][8] = 1
    W[4][8] = 1
    
    U[2][9] = 1
    V[2][9] = 1
    W[5][9] = 1
    
    U[3][10] = 1
    V[5][10] = 1
    W[5][10] = 1
    
    return U, V, W

def _try_rank10_construction():
    """
    Attempt rank-10 by using a Strassen-like decomposition for column 2.
    
    The column-2 sub-tensor: T[a,b,c] for b in {2,5}, c in {4,5}, a in {0,1,2,3}
    Nonzero entries:
      T[0,2,4] = 1  (a=0=i0j0, b=2=j0k2, c=4=k2i0)
      T[1,2,5] = 1  (a=1=i0j1, b=2=j0k2, c=5=k2i1)  -- wait, let me recheck
      T[2,5,4] = 1  (a=2=i1j0, b=5=j1k2, c=4=k2i0)
      T[3,5,5] = 1  (a=3=i1j1, b=5=j1k2, c=5=k2i1)
    
    Actually let me re-index carefully.
    a = i*m + j, i in {0,1}, j in {0,1} -> a in {0,1,2,3}
    b = j*p + k, j in {0,1}, k in {0,1,2} -> b in {0,1,2,3,4,5}
    c = k*n + i, k in {0,1,2}, i in {0,1} -> c in {0,1,2,3,4,5}
    
    For column 2 (k=2): b in {2,5}, c in {4,5}
    T[0,2,4]: i=0,j=0,k=2 -> a=0*2+0=0, b=0*3+2=2, c=2*2+0=4 ✓
    T[1,2,5]: i=0,j=1,k=2 -> a=0*2+1=1, b=1*3+2=5, c=2*2+0=4 -> c=4, not 5!
    
    Wait, c = k*n + i. For i=0, k=2: c = 2*2+0 = 4. For i=1, k=2: c = 2*2+1 = 5.
    
    So:
    T[0,2,4] = 1: i=0,j=0,k=2
    T[1,5,4] = 1: i=0,j=1,k=2  (b = j*p+k = 1*3+2 = 5)
    T[2,2,5] = 1: i=1,j=0,k=2  (a = i*m+j = 1*2+0 = 2, c = 5)
    T[3,5,5] = 1: i=1,j=1,k=2  (a = 1*2+1 = 3, b = 1*3+2 = 5, c = 5)
    
    So the 4x2x2 sub-tensor has entries:
    (a=0, b=2, c=4) = 1
    (a=1, b=5, c=4) = 1
    (a=2, b=2, c=5) = 1
    (a=3, b=5, c=5) = 1
    
    This is the tensor T where T[a, b', c'] with b' = b-2 in {0,1}, c' = c-4 in {0,1}:
    T[0,0,0] = 1, T[1,1,0] = 1, T[2,0,1] = 1, T[3,1,1] = 1
    
    This is isomorphic to the 2x2x2 matrix multiplication tensor!
    It has rank 7 (Strassen) or rank 4 (schoolbook) for 2x2x2, but this is a 
    different tensor. Actually this is the "direct sum" type tensor.
    
    Let me check: can this be done in rank 3?
    We need: sum_r u_r[a] * v_r[b'] * w_r[c'] = T[a,b',c']
    
    The four nonzero entries are at (0,0,0), (1,1,0), (2,0,1), (3,1,1).
    
    For rank 3, we'd need 3 terms. Let me try:
    Term 1: covers (0,0,0) and (2,0,1) -> needs v to be nonzero at b'=0
    Term 2: covers (1,1,0) and (3,1,1) -> needs v to be nonzero at b'=1
    
    Term 1: u1 = [1, 0, 1, 0], v1 = [1, 0], w1 = [1, 1]
    This gives: (0,0,0)=1, (0,0,1)=1, (2,0,0)=1, (2,0,1)=1
    But we don't want (0,0,1) or (2,0,0) to be 1.
    
    This is getting complex. Let me just try to construct and verify.
    """
    n, m, p = 2, 2, 3
    rank = 10
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Copy Strassen terms (r=0..6)
    U11, V11, W11 = _rank11_solution()
    for r in range(7):
        for i in range(n * m):
            U[i][r] = U11[i][r]
        for j in range(m * p):
            V[j][r] = V11[j][r]
        for k in range(p * n):
            W[k][r] = W11[k][r]
    
    # Column 2 sub-tensor needs: (0,2,4)=1, (1,5,4)=1, (2,2,5)=1, (3,5,5)=1
    # Try rank 3 for this sub-tensor
    # Term 7: U[0,7]=1, V[2,7]=1, W[4,7]=1  -> covers (0,2,4)
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1
    
    # Term 8: U[1,8]=1, V[5,8]=1, W[4,8]=1  -> covers (1,5,4)
    U[1][8] = 1
    V[5][8] = 1
    W[4][8] = 1
    
    # Term 9: U[2,9]=1, U[3,9]=1, V[2,9]=1, V[5,9]=1, W[5,9]=1
    # This gives: (2,2,5), (3,2,5), (2,5,5), (3,5,5)
    # We want (2,2,5)=1 and (3,5,5)=1, but also get (3,2,5)=1 and (2,5,5)=1 which we don't want
    U[2][9] = 1
    U[3][9] = 1
    V[2][9] = 1
    V[5][9] = 1
    W[5][9] = 1
    
    valid, diff = _verify(U, V, W)
    if valid:
        return U, V, W, 10
    return None

def _try_rank10_alternative():
    """Alternative rank-10: use 3 terms for column 2 with different structure."""
    n, m, p = 2, 2, 3
    rank = 10
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Copy Strassen terms (r=0..6)
    U11, V11, W11 = _rank11_solution()
    for r in range(7):
        for i in range(n * m):
            U[i][r] = U11[i][r]
        for j in range(m * p):
            V[j][r] = V11[j][r]
        for k in range(p * n):
            W[k][r] = W11[k][r]
    
    # Try different 3-term decomposition for column 2
    # Term 7: a*b02 contribution to C02
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1  # C02
    
    # Term 8: c*b02 contribution to C12
    U[2][8] = 1
    V[2][8] = 1
    W[5][8] = 1  # C12
    
    # Term 9: (b+d)*b12 with split
    # C02 needs +b*b12, C12 needs +d*b12
    # Use: (b+d)*b12, then C02 gets (b+d)*b12 and we subtract d*b12
    # C12 gets (b+d)*b12 and we subtract b*b12
    # But that's 5 terms total. Not helpful.
    
    # Instead: term 9 = (a+c) * b02? No, that mixes.
    
    # Let me try: term 9 covers both b*b12 and d*b12 using a trick
    # (b+d) * b12: U has b and d, V has b12, W has both C02 and C12
    U[1][9] = 1  # b
    U[3][9] = 1  # d
    V[5][9] = 1  # b12
    W[4][9] = 1  # C02
    W[5][9] = 1  # C12
    
    valid, diff = _verify(U, V, W)
    if valid:
        return U, V, W, 10
    return None

def _try_rank10_third():
    """
    Third approach: merge one Strassen term with column-2 handling.
    The key idea: some Strassen terms involve (a+d) or (a+b) combinations.
    Perhaps we can fold b02 or b12 into these.
    
    Strassen M1 = (a+d)(b00+b11). What if we use (a+d)(b00+b11+b02+b12)?
    That would add 2 extra products but might save more elsewhere.
    Net: 1 term replaced, but we'd need to subtract off the extra contributions.
    Not obviously helpful.
    
    Alternative: Can we share a factor between Strassen and column-2?
    M3 = a*(b01-b11). What about a*(b01-b11+b02)? Then C02 gets a*b02 automatically.
    But C02 = a*b02 + b*b12, so we'd still need b*b12.
    And we'd need to subtract a*b02 from other outputs if it leaks.
    
    Let me try: modify M3 to a*(b01-b11+b02) and add a term for b*b12.
    M3' = a*(b01-b11+b02): U[0]=1, V[1]=1, V[4]=-1, V[2]=1
    This contributes to C01 (via M3) and to C02 (via the new a*b02 part).
    C01 = M3 + M5 = a*(b01-b11+b02) + (a+b)*b11
    = a*b01 - a*b11 + a*b02 + a*b11 + b*b11
    = a*b01 + a*b02 + b*b11
    But we want C01 = a*b01 + b*b11. The a*b02 term is wrong!
    
    So we'd need to subtract a*b02 from C01. That's an extra term. Not helpful.
    
    Let me try yet another approach: use a non-integer decomposition.
    """
    n, m, p = 2, 2, 3
    rank = 10
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Copy Strassen terms (r=0..6)
    U11, V11, W11 = _rank11_solution()
    for r in range(7):
        for i in range(n * m):
            U[i][r] = float(U11[i][r])
        for j in range(m * p):
            V[j][r] = float(V11[j][r])
        for k in range(p * n):
            W[k][r] = float(W11[k][r])
    
    # For column 2, try a rank-3 decomposition with non-integer values
    # Sub-tensor: (0,2,4)=1, (1,5,4)=1, (2,2,5)=1, (3,5,5)=1
    # Let b' = b-2 in {0,1}, c' = c-4 in {0,1}
    # T[a,b',c']: (0,0,0)=1, (1,1,0)=1, (2,0,1)=1, (3,1,1)=1
    
    # This is actually the matrix multiplication tensor for 2x2 matrices!
    # Let me verify: M2x2 tensor has T[i,j,k] = 1 if i=j=k... no.
    # M2x2: C[i,k] = sum_j A[i,j]*B[j,k]
    # T[i,j,k] = 1 for all i,j,k in {0,1} (it's the full 2x2x2 tensor of 1s? No!)
    # Actually T[i,j,k] = 1 always for M2x2? No.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # So T[0,0,0] = 1 (i=0,j=0,k=0), T[1,0,1] = 1 (i=0,j=1,k=1 -> a=1,b=1,c=1)
    # Wait, for M2x2: a = i*2+j, b = j*2+k, c = k*2+i
    # T[0,0,0]=1: i=0,j=0,k=0
    # T[1,0,2]=1: i=0,j=1,k=0 -> a=1,b=2,c=2 -> but b should be <4 for 2x2... 
    # For 2x2: m=2, p=2. b = j*2+k, j in {0,1}, k in {0,1} -> b in {0,1,2,3}
    # c = k*2+i, k in {0,1}, i in {0,1} -> c in {0,1,2,3}
    # T[0,0,0]=1: i=0,j=0,k=0
    # T[1,2,0]=1: i=0,j=1,k=0 -> a=0*2+1=1, b=1*2+0=2, c=0*2+0=0
    # T[2,0,2]=1: i=1,j=0,k=0 -> a=1*2+0=2, b=0*2+0=0, c=0*2+1=2
    # T[3,2,2]=1: i=1,j=1,k=0 -> a=3, b=2, c=2
    # T[0,1,1]=1: i=0,j=0,k=1 -> a=0, b=1, c=1
    # T[1,3,1]=1: i=0,j=1,k=1 -> a=1, b=3, c=1
    # T[2,1,3]=1: i=1,j=0,k=1 -> a=2, b=1, c=3
    # T[3,3,3]=1: i=1,j=1,k=1 -> a=3, b=3, c=3
    
    # So M2x2 tensor has 8 nonzero entries, all = 1.
    # My column-2 sub-tensor has only 4 nonzero entries. It's a sub-tensor, not M2x2.
    
    # The 4-entry tensor: (0,0,0), (1,1,0), (2,0,1), (3,1,1) in (a,b',c') space
    # This looks like it could be rank 3. Let me try to construct.
    # Term 7: u=[1,0,0,0], v=[1,0], w=[1,0] -> (0,0,0)=1
    # Term 8: u=[0,1,0,0], v=[0,1], w=[1,0] -> (1,1,0)=1
    # Term 9: u=[0,0,1,1], v=[1,1], w=[0,1] -> (2,0,1)=1, (2,1,1)=1, (3,0,1)=1, (3,1,1)=1
    # Extra: (2,1,1), (3,0,1) are unwanted.
    
    # Try: Term 9: u=[0,0,1,0], v=[1,0], w=[0,1] -> (2,0,1)=1
    # Term 10: u=[0,0,0,1], v=[0,1], w=[0,1] -> (3,1,1)=1
    # That's rank 5 for column 2. Total rank = 7+5 = 12. Worse.
    
    # The issue is that (1,1,0) and (3,1,1) share b'=1 but different a and c'.
    # And (0,0,0) and (2,0,1) share b'=0 but different a and c'.
    
    # Can we do it in rank 3? 
    # We need: sum_{r=1}^3 u_r[a]*v_r[b']*w_r[c'] = T[a,b',c']
    # With 4 constraints and 3*4*2*2 = 48 unknowns, it should be possible if no contradictions.
    # Let me set up the system.
    # For each (a,b',c') with T=1: sum_r u_r[a]*v_r[b']*w_r[c'] = 1
    # For each (a,b',c') with T=0: sum_r u_r[a]*v_r[b']*w_r[c'] = 0
    # There are 4*2*2 = 16 total (a,b',c') combos, 4 are 1, 12 are 0.
    # 36 unknowns (3*4 + 3*2 + 3*2 = 12+6+6), 16 equations. Underdetermined, should have solution.
    
    # Let me try a specific construction.
    # Term 7: v=[1,0], covers b'=0 entries: (0,0,0)=1, (2,0,1)=1
    #   u7=[1, x, 1, y], w7=[1, 1] -> (0,0,0)=1, (0,0,1)=1, (2,0,0)=1, (2,0,1)=1
    #   Unwanted: (0,0,1), (2,0,0). Need to cancel with other terms.
    
    # This is getting complex. Let me just return None and fall back.
    return None

def construct():
    _t0 = time.time()
    _deadline = 130.0  # safety margin before 150s kill
    
    # Try rank-10 constructions
    if time.time() - _t0 < _deadline:
        result = _try_rank10_construction()
        if result is not None:
            U, V, W, rank = result
            return {"U": U, "V": V, "W": W}
    
    if time.time() - _t0 < _deadline:
        result = _try_rank10_alternative()
        if result is not None:
            U, V, W, rank = result
            return {"U": U, "V": V, "W": W}
    
    if time.time() - _t0 < _deadline:
        result = _try_rank10_third()
        if result is not None:
            U, V, W, rank = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to rank 11
    U, V, W = _rank11_solution()
    return {"U": U, "V": V, "W": W}
