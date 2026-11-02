import numpy as np
import time
import itertools

DIMENSIONS = [2, 2, 15]
_DEADLINE = time.time() + 140  # safety margin

def _find_2x2x3_rank10():
    """Search for a rank-10 decomposition of 2x2x3 matrix multiplication.
    Returns (U, V, W) arrays of shape (4,10), (6,10), (6,10) or None."""
    n, m, p = 2, 2, 3
    
    tensor = np.zeros((4, 6, 6), dtype=float)
    for a in range(4):
        for b in range(6):
            for c in range(6):
                i, j = a // m, a % m
                j2, k = b // p, b % p
                k2, i2 = c // n, c % n
                if j == j2 and k == k2 and i == i2:
                    tensor[a, b, c] = 1.0
    
    # Build rank-11 reference solution (Strassen + schoolbook)
    U11 = np.zeros((4, 11))
    V11 = np.zeros((6, 11))
    W11 = np.zeros((6, 11))
    
    # Strassen for cols 0,1 (using same convention as main code)
    # U: 0=A00, 1=A01, 2=A10, 3=A11
    # V: 0=B00, 1=B01, 2=B02, 3=B10, 4=B11, 5=B12
    # W: 0=C00, 1=C10, 2=C01, 3=C11, 4=C02, 5=C12
    
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    U11[0, 0] = 1; U11[3, 0] = 1
    V11[0, 0] = 1; V11[4, 0] = 1
    W11[0, 0] = 1; W11[3, 0] = 1
    
    # M2 = (A10+A11)B00 -> C10
    U11[2, 1] = 1; U11[3, 1] = 1
    V11[0, 1] = 1
    W11[1, 1] = 1
    
    # M3 = A00(B01-B11) -> C01
    U11[0, 2] = 1
    V11[1, 2] = 1; V11[4, 2] = -1
    W11[2, 2] = 1
    
    # M4 = A11(B10-B00) -> C00, C10
    U11[3, 3] = 1
    V11[3, 3] = 1; V11[0, 3] = -1
    W11[0, 3] = 1; W11[1, 3] = 1
    
    # M5 = (A00+A01)B11 -> C00, C01
    U11[0, 4] = 1; U11[1, 4] = 1
    V11[4, 4] = 1
    W11[0, 4] = -1; W11[2, 4] = 1
    
    # M6 = (A10-A00)(B00+B01) -> C11
    U11[2, 5] = 1; U11[0, 5] = -1
    V11[0, 5] = 1; V11[1, 5] = 1
    W11[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11) -> C00, C01
    U11[1, 6] = 1; U11[3, 6] = -1
    V11[3, 6] = 1; V11[4, 6] = 1
    W11[0, 6] = 1; W11[2, 6] = 1
    
    # Schoolbook for col 2
    U11[0, 7] = 1; V11[2, 7] = 1; W11[4, 7] = 1
    U11[1, 8] = 1; V11[5, 8] = 1; W11[4, 8] = 1
    U11[2, 9] = 1; V11[2, 9] = 1; W11[5, 9] = 1
    U11[3, 10] = 1; V11[5, 10] = 1; W11[5, 10] = 1
    
    # Verify
    T11 = np.einsum('al,bl,cl->abc', U11, V11, W11)
    if np.max(np.abs(T11 - tensor)) > 1e-10:
        return None
    
    # Try to find rank-10 by searching near the rank-11 solution
    # Use a parameterized family: merge two of the schoolbook terms with Strassen
    # Try all possible "merge" patterns
    
    best_err = float('inf')
    best_result = None
    
    # Strategy: try to express one of the 4 schoolbook terms as a linear combination
    # of the 7 Strassen terms plus modifications
    # This is equivalent to finding a rank-10 solution in the span
    
    # Alternative: use a random search with the constraint that the solution is close to rank-11
    np.random.seed(123)
    
    for trial in range(300):
        if time.time() > _DEADLINE - 25:
            break
        
        # Start from rank-11 and try to zero out one column
        # Then re-optimize the remaining 10 columns
        drop_idx = np.random.randint(11)
        U = U11[:, list(range(11))[:drop_idx] + list(range(11))[drop_idx+1:]].copy()
        V = V11[:, list(range(11))[:drop_idx] + list(range(11))[drop_idx+1:]].copy()
        W = W11[:, list(range(11))[:drop_idx] + list(range(11))[drop_idx+1:]].copy()
        
        # Add small perturbation
        U += np.random.randn(4, 10) * 0.1
        V += np.random.randn(6, 10) * 0.1
        W += np.random.randn(6, 10) * 0.1
        
        # Simple gradient descent on the residual
        for it in range(200):
            T_approx = np.einsum('al,bl,cl->abc', U, V, W)
            resid = T_approx - tensor
            
            # Gradient w.r.t U
            dU = np.einsum('abc,bl,cl->al', resid, V, W)
            dV = np.einsum('abc,al,cl->bl', resid, U, W)
            dW = np.einsum('abc,al,bl->cl', resid, U, V)
            
            # Adaptive learning rate
            lr = 0.01 / (1 + it * 0.01)
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            err = np.sum(resid**2)
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-10:
                break
        
        if best_err < 1e-10:
            break
    
    if best_err < 1e-8 and best_result is not None:
        U_best, V_best, W_best = best_result
        # Try to snap to half-integers
        U_clean = np.round(U_best * 2) / 2
        V_clean = np.round(V_best * 2) / 2
        W_clean = np.round(W_best * 2) / 2
        
        T_clean = np.einsum('al,bl,cl->abc', U_clean, V_clean, W_clean)
        if np.max(np.abs(T_clean - tensor)) < 1e-8:
            return U_clean, V_clean, W_clean
        else:
            # Try integers
            U_clean = np.round(U_best)
            V_clean = np.round(V_best)
            W_clean = np.round(W_best)
            T_clean = np.einsum('al,bl,cl->abc', U_clean, V_clean, W_clean)
            if np.max(np.abs(T_clean - tensor)) < 1e-8:
                return U_clean, V_clean, W_clean
            else:
                return U_best, V_best, W_best
    
    return None

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # Try to find a rank-10 2x2x3 decomposition
    # If found: 5 Strassen blocks (35) + 1 rank-10 block (10) = 45
    # If not: 7 Strassen blocks (49) + 1 schoolbook (4) = 53
    
    triple_decomp = None
    if time.time() < _DEADLINE - 60:
        triple_decomp = _find_2x2x3_rank10()
    
    if triple_decomp is not None:
        U3, V3, W3 = triple_decomp
        # 5 Strassen blocks for cols (0,1),(2,3),(4,5),(6,7),(8,9)
        # 1 rank-10 block for cols (10,11,12,13,14)
        rank = 5 * 7 + 10  # 45
    else:
        # Fallback: 7 Strassen + 1 schoolbook = 53
        # Using lists as in the seed program to ensure compatibility
        rank = 7 * 7 + 10  # 59 (7 Strassen blocks of rank 7 + 1 schoolbook block of rank 10)
    U = [[0.0] * rank for _ in range(n * m)]  # 4 x rank
    V = [[0.0] * rank for _ in range(m * p)]  # 30 x rank
    W = [[0.0] * rank for _ in range(p * n)]  # 30 x rank
    
    # A indices in U:
    # 0: A00, 1: A01, 2: A10, 3: A11
    
    # B indices in V:
    # j*p + k. j=0 -> 0..14, j=1 -> 15..29
    
    # C indices in W:
    # k*n + i. k=0 -> 0,1; k=1 -> 2,3; ... k=14 -> 28,29
    
    def add_strassen_block(base_r, k0, k1):
        # k0, k1 are the column indices in B (and C)
        r1 = base_r
        r2 = base_r + 1
        r3 = base_r + 2
        r4 = base_r + 3
        r5 = base_r + 4
        r6 = base_r + 5
        r7 = base_r + 6
        
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        U[0][r1] += 1.0
        U[3][r1] += 1.0
        V[0*p + k0][r1] += 1.0
        V[1*p + k1][r1] += 1.0
        
        # M2 = (A10 + A11) * B0k0
        U[2][r2] += 1.0
        U[3][r2] += 1.0
        V[0*p + k0][r2] += 1.0
        
        # M3 = A00 * (B0k1 - B1k1)
        U[0][r3] += 1.0
        V[0*p + k1][r3] += 1.0
        V[1*p + k1][r3] += -1.0
        
        # M4 = A11 * (B1k0 - B0k0)
        U[3][r4] += 1.0
        V[1*p + k0][r4] += 1.0
        V[0*p + k0][r4] += -1.0
        
        # M5 = (A00 + A01) * B1k1
        U[0][r5] += 1.0
        U[1][r5] += 1.0
        V[1*p + k1][r5] += 1.0
        
        # M6 = (A10 - A00) * (B0k0 + B0k1)
        U[2][r6] += 1.0
        U[0][r6] += -1.0
        V[0*p + k0][r6] += 1.0
        V[0*p + k1][r6] += 1.0
        
        # M7 = (A01 - A11) * (B1k0 + B1k1)
        U[1][r7] += 1.0
        U[3][r7] += -1.0
        V[1*p + k0][r7] += 1.0
        V[1*p + k1][r7] += 1.0
        
        # Reconstruction W
        # C0k0 = M1 + M4 - M5 + M7
        W[2*k0 + 0][r1] += 1.0
        W[2*k0 + 0][r4] += 1.0
        W[2*k0 + 0][r5] += -1.0
        W[2*k0 + 0][r7] += 1.0
        
        # C1k0 = M2 + M4
        W[2*k0 + 1][r2] += 1.0
        W[2*k0 + 1][r4] += 1.0
        
        # C0k1 = M3 + M5
        W[2*k1 + 0][r3] += 1.0
        W[2*k1 + 0][r5] += 1.0
        
        # C1k1 = M1 - M2 + M3 + M6
        W[2*k1 + 1][r1] += 1.0
        W[2*k1 + 1][r2] += -1.0
        W[2*k1 + 1][r3] += 1.0
        W[2*k1 + 1][r6] += 1.0

    if triple_decomp is not None:
        # 5 Strassen blocks for pairs (0,1),(2,3),(4,5),(6,7),(8,9)
        for i in range(5):
            k0 = 2 * i
            k1 = 2 * i + 1
            base_r = 7 * i
            add_strassen_block(base_r, k0, k1)
        
        # Rank-10 block for columns 10-14
        base_r = 35
        U3, V3, W3 = triple_decomp
        for l in range(10):
            r = base_r + l
            # U3 has shape (4, 10)
            for a in range(4):
                U[a][r] += U3[a, l]
            # V3 has shape (6, 10), indices: j*p+k where p=3 for the sub-problem
            # But in the full problem, p=15. The sub-problem cols are 10,11,12,13,14
            # V3 index j*p_sub+k_sub where p_sub=3, k_sub=0,1,2 -> maps to k=10+k_sub
            # Actually V3 has 6 rows: j in {0,1}, k in {0,1,2}
            # Full V index: j*15 + (10+k_sub)
            for b in range(6):
                j_sub = b // 3
                k_sub = b % 3
                V[j_sub * p + 10 + k_sub][r] += V3[b, l]
            # W3 has shape (6, 10), indices: k_sub*n+i where n=2, i in {0,1}
            # Full W index: (10+k_sub)*n + i
            for c in range(6):
                k_sub = c // 2
                i_sub = c % 2
                W[(10 + k_sub) * n + i_sub][r] += W3[c, l]
    else:
        # Fallback: 7 Strassen + 1 schoolbook
        for i in range(7):
            k0 = 2 * i
            k1 = 2 * i + 1
            base_r = 7 * i
            add_strassen_block(base_r, k0, k1)
        
        # Add trivial block for column 14
        base_r = 49
        k_last = 14
        
        r1 = base_r
        r2 = base_r + 1
        r3 = base_r + 2
        r4 = base_r + 3
        
        # Q1 = A00 * B0k_last
        U[0][r1] += 1.0
        V[0*p + k_last][r1] += 1.0
        W[2*k_last + 0][r1] += 1.0
        
        # Q2 = A01 * B1k_last
        U[1][r2] += 1.0
        V[1*p + k_last][r2] += 1.0
        W[2*k_last + 0][r2] += 1.0
        
        # Q3 = A10 * B0k_last
        U[2][r3] += 1.0
        V[0*p + k_last][r3] += 1.0
        W[2*k_last + 1][r3] += 1.0
        
        # Q4 = A11 * B1k_last
        U[3][r4] += 1.0
        V[1*p + k_last][r4] += 1.0
        W[2*k_last + 1][r4] += 1.0

    return {"U": U, "V": V, "W": W}
