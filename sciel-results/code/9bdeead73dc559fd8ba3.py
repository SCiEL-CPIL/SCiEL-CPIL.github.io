"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def _verify(U, V, W, n, m, p):
    """Verify the tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 if valid matmul term"""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # The tensor T[i,j,k] should equal sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # For matmul, T[i,j,k] = 1 for all i,j,k (it's the full tensor)
    # Actually T[i,j,k] = 1 for the matmul tensor: C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is T(i,j,k) = 1 for all i in [n], j in [m], k in [p]
    # Wait no - the matmul tensor has T(i,j,k) = 1 for all valid i,j,k
    # The decomposition requires: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    total = np.zeros((n, m, p))
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total[i,j,k] += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
    expected = np.ones((n, m, p))
    if not np.allclose(total, expected, atol=1e-6):
        diff = np.max(np.abs(total - expected))
        return False, diff
    return True, 0.0

def _try_rank(n, m, p, target_rank, timeout=10):
    """Try to find a decomposition of given rank (placeholder for future optimization)"""
    pass

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill
    
    def _build_rank54():
        rank = 54
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        j_pairs = [(0, 1), (2, 3)]
        k_pairs = [(0, 1), (2, 3), (4, 5)]
        
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # P1 = (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
                U[0*m + j0][col] += 1
                U[1*m + j1][col] += 1
                V[j0*p + k0][col] += 1
                V[j1*p + k1][col] += 1
                W[k0*n + 0][col] += 1
                W[k1*n + 1][col] += 1
                col += 1
                
                # P2 = (A_1,j0 + A_1,j1) B_j0,k0
                U[1*m + j0][col] += 1
                U[1*m + j1][col] += 1
                V[j0*p + k0][col] += 1
                W[k0*n + 1][col] += 1
                W[k1*n + 1][col] -= 1
                col += 1
                
                # P3 = A_0,j0 (B_j0,k1 - B_j1,k1)
                U[0*m + j0][col] += 1
                V[j0*p + k1][col] += 1
                V[j1*p + k1][col] -= 1
                W[k1*n + 0][col] += 1
                W[k1*n + 1][col] += 1
                col += 1
                
                # P4 = A_1,j1 (B_j1,k0 - B_j0,k0)
                U[1*m + j1][col] += 1
                V[j1*p + k0][col] += 1
                V[j0*p + k0][col] -= 1
                W[k0*n + 0][col] += 1
                W[k0*n + 1][col] += 1
                col += 1
                
                # P5 = (A_0,j0 + A_0,j1) B_j1,k1
                U[0*m + j0][col] += 1
                U[0*m + j1][col] += 1
                V[j1*p + k1][col] += 1
                W[k0*n + 0][col] -= 1
                W[k1*n + 0][col] += 1
                col += 1
                
                # P6 = (A_1,j0 - A_0,j0)(B_j0,k0 + B_j0,k1)
                U[1*m + j0][col] += 1
                U[0*m + j0][col] -= 1
                V[j0*p + k0][col] += 1
                V[j0*p + k1][col] += 1
                W[k1*n + 1][col] += 1
                col += 1
                
                # P7 = (A_0,j1 - A_1,j1)(B_j1,k0 + B_j1,k1)
                U[0*m + j1][col] += 1
                U[1*m + j1][col] -= 1
                V[j1*p + k0][col] += 1
                V[j1*p + k1][col] += 1
                W[k0*n + 0][col] += 1
                col += 1
        
        # Remaining j=4
        for k in range(p):
            U[0*m + 4][col] += 1
            V[4*p + k][col] += 1
            W[k*n + 0][col] += 1
            col += 1
            U[1*m + 4][col] += 1
            V[4*p + k][col] += 1
            W[k*n + 1][col] += 1
            col += 1
        
        return {"U": U, "V": V, "W": W}
    
    def _build_rank52():
        """Try rank 52: use Strassen for j-pairs {0,1} and {2,3} with k-pairs,
        but for j=4 use a smarter scheme combining with adjacent j's."""
        # Actually, let's try: pair j as {0,1}, {2,3}, {4} 
        # For {0,1} x all 6 k's: use 3 k-pairs * 7 = 21
        # For {2,3} x all 6 k's: use 3 k-pairs * 7 = 21
        # For {4} x all 6 k's: 12 terms
        # Total = 54. Same.
        
        # Alternative: what if we use a 3x3 scheme for j={0,1,2}?
        # 2x3x6 tensor. Can we do better than 2x2x6 + 2x1x6 = 21+12 = 33?
        # Try: for j={0,1,2}, k={0,1,2,3,4,5}
        # Use Strassen on j-pairs {0,1} and handle j=2 separately within each k-pair
        # {0,1} x {0,1}: 7, {0,1} x {2,3}: 7, {0,1} x {4,5}: 7 = 21
        # {2} x {0,1}: 4, {2} x {2,3}: 4, {2} x {4,5}: 4 = 12
        # Total for j={0,1,2}: 33
        # j={3,4}: 2x2x6 -> 21
        # Total: 54 again.
        
        # Let me try a completely different approach: use the fact that
        # we can share terms across j indices.
        # For a 2xm x mxp matmul, think of it as: for each i in {0,1}, k in {0..5},
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So we need to represent 12 dot products of length 5.
        
        # The trivial way: 12*5 = 60 rank-1 terms
        # Strassen helps by combining.
        
        # Let me try: group k into 3 pairs, and for each k-pair, use a 2x5x2 scheme
        # 2x5x2: multiply 2x5 by 5x2. Rank?
        # Trivial: 2*5*2 = 20. Can we do better?
        # For each of 2 outputs (k0, k1), C[i,k] = sum_j A[i,j]*B[j,k]
        # That's 4 dot products of length 5.
        # Using Strassen-like: pair up j's.
        # j={0,1}: 2x2x2 -> 7 terms
        # j={2,3}: 2x2x2 -> 7 terms  
        # j={4}: 2x1x2 -> 4 terms
        # Total per k-pair: 18
        # 3 k-pairs: 54. Same!
        
        # What about 2x5x2 with a different scheme?
        # Actually 2x1x2 has rank 4 (each of 2 outputs needs 2 terms for 1 input dim... 
        # wait: C0 = A0*B0 + A1*B1, C1 = A0*B2 + A1*B3 where B is 1x2.
        # So C0 = A0*B0 + A1*B1, C1 = A0*B2 + A1*B3. That's 4 independent rank-1 terms.
        
        # Hmm, I keep getting 54. Let me try to see if 53 or 52 is possible.
        # 
        # Actually, let me try a randomized approach: start with rank 52 and try to
        # solve the linear system for the tensor entries.
        
        return None  # placeholder
    
    # Build baseline
    baseline = _build_rank54()
    valid, diff = _verify(baseline["U"], baseline["V"], baseline["W"], n, m, p)
    
    best = baseline
    best_rank = 54
    
    if time.time() > deadline - 30:
        return best
    
    # Try rank 53 and 52 using linear algebra
    # The condition is: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # This is a system of n*m*p = 60 equations in rank^3 unknowns (but structured).
    # For rank r, we have r*(n*m + m*p + p*n) = r*(10+30+12) = 52r unknowns.
    # For r=53: 52*53 = 2756 unknowns, 60 equations. Very underdetermined.
    # But the equations are nonlinear (products of 3 variables per term).
    
    # Alternative: fix U and V, solve for W linearly, or vice versa.
    # For fixed U and V, each equation is linear in W entries.
    # Equation (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # This is linear in W[k*n+i,r] for each r, but the coefficient depends on r.
    # Actually for fixed (i,j,k), we get: sum_r (U[i*m+j,r]*V[j*p+k,r]) * W[k*n+i,r] = 1
    # This is one linear equation in the W variables W[0,0],W[0,1],...,W[5,1] (12 vars)
    # with coefficients c_r = U[i*m+j,r]*V[j*p+k,r] for r=0..rank-1.
    # But we only use W[k*n+i,r] for the specific (k,i) pair!
    # So equation (i,j,k) only involves W[k*n+i, r] for all r.
    # That means for each (k,i) pair, we have m equations (one for each j) all involving
    # the same W[k*n+i, :].
    
    # So: fix U and V. For each (k,i) in range(p*n):
    #   For each j in range(m):
    #     sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # This is a system of m linear equations in rank unknowns W[k*n+i,0..rank-1].
    # If rank >= m, this is solvable (generically).
    
    # Strategy: 
    # 1. Pick random U and V with rank columns
    # 2. For each (k,i), solve the linear system for W[k*n+i,:]
    # 3. Verify the full decomposition
    
    import random
    random.seed(42)
    
    def _try_random_decomposition(target_rank, n, m, p, max_attempts=50):
        """Try to find a decomposition of given rank by fixing U,V randomly and solving for W."""
        for attempt in range(max_attempts):
            # Initialize U and V with random values
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            
            # Use sparse random initialization: each row has a few nonzeros
            for _ in range(target_rank):
                # Pick a few entries to be nonzero
                for _ in range(random.randint(1, 3)):
                    row = random.randint(0, n*m - 1)
                    U[row, _] = random.choice([-1, 1])
                for _ in range(random.randint(1, 3)):
                    row = random.randint(0, m*p - 1)
                    V[row, _] = random.choice([-1, 1])
            
            # For each (k, i), solve for W[k*n + i, :]
            W = np.zeros((p*n, target_rank))
            feasible = True
            for k in range(p):
                for i in range(n):
                    # Build system: for each j, sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
                    # Let c_r = U[i*m+j,r]*V[j*p+k,r] for the specific j
                    # System: M @ W[k*n+i,:] = 1 where M[j,r] = U[i*m+j,r]*V[j*p+k,r]
                    M = np.zeros((m, target_rank))
                    for j in range(m):
                        for r in range(target_rank):
                            M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    
                    # Solve M @ w = ones(m)
                    try:
                        # Use least squares
                        w, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        # Check if it's an exact solution
                        if np.allclose(M @ w, np.ones(m), atol=1e-8):
                            W[k*n+i, :] = w
                        else:
                            # Try exact solution if possible
                            if rank_m == m and rank_m <= target_rank:
                                # Use pinv
                                w = np.linalg.pinv(M) @ np.ones(m)
                                if np.allclose(M @ w, np.ones(m), atol=1e-8):
                                    W[k*n+i, :] = w
                                else:
                                    feasible = False
                                    break
                            else:
                                feasible = False
                                break
                    except:
                        feasible = False
                        break
                if not feasible:
                    break
            
            if not feasible:
                continue
            
            # Verify
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            valid, diff = _verify(U_list, V_list, W_list, n, m, p)
            if valid:
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try rank 52
    if time.time() < deadline - 15:
        result = _try_random_decomposition(52, n, m, p, max_attempts=20)
        if result:
            best = result
            best_rank = 52
    
    # Try rank 51
    if time.time() < deadline - 10 and best_rank > 51:
        result = _try_random_decomposition(51, n, m, p, max_attempts=20)
        if result:
            best = result
            best_rank = 51
    
    # Try rank 50
    if time.time() < deadline - 5 and best_rank > 50:
        result = _try_random_decomposition(50, n, m, p, max_attempts=20)
        if result:
            best = result
            best_rank = 50
    
    return best
# EVOLVE-BLOCK-END
