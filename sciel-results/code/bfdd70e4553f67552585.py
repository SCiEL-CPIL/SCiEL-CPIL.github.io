import time
import numpy as np

DIMENSIONS = [4, 4, 5]
DEADLINE = 130.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return U, V, W (each 4x7) for Strassen 2x2x2 matrix multiplication."""
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    # Indices: A[i,j] -> U[i*2+j], B[j,k] -> V[j*2+k], C[i,k] -> W[k*2+i]
    # C[0,0] -> W[0], C[0,1] -> W[2], C[1,0] -> W[1], C[1,1] -> W[3]
    
    # P1 = (A00 + A11) * (B00 + B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (A10 + A11) * B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = A00 * (B10 - B00)
    U[0][2] = 1
    V[2][2] = 1; V[0][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = A11 * (B01 - B11)
    U[3][3] = 1
    V[1][3] = 1; V[3][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (A00 + A01) * B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[2][4] = 1; W[0][4] = -1
    
    # P6 = (A01 - A11) * (B00 + B01)
    U[1][5] = 1; U[3][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (A00 - A10) * (B10 + B11)
    U[0][6] = 1; U[2][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _strassen_4x4x4():
    """Construct rank 49 decomposition for 4x4x4."""
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    U2, V2, W2 = _strassen_2x2x2()
    
    steps = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(1,0): 1, (0,0): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(0,1): 1, (1,1): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,0): 1, (1,0): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    for r, (a_coeffs, b_coeffs, c_coeffs) in enumerate(steps):
        for s in range(7):
            col = r * 7 + s
            
            for bi in range(2):
                for bj in range(2):
                    coeff_a = a_coeffs.get((bi, bj), 0)
                    if coeff_a == 0: continue
                    for li in range(2):
                        for lj in range(2):
                            local_u = U2[li*2+lj][s]
                            if local_u == 0: continue
                            global_i = bi*2 + li
                            global_j = bj*2 + lj
                            U[global_i*4 + global_j][col] += coeff_a * local_u
            
            for bj in range(2):
                for bk in range(2):
                    coeff_b = b_coeffs.get((bj, bk), 0)
                    if coeff_b == 0: continue
                    for lj in range(2):
                        for lk in range(2):
                            local_v = V2[lj*2+lk][s]
                            if local_v == 0: continue
                            global_j = bj*2 + lj
                            global_k = bk*2 + lk
                            V[global_j*4 + global_k][col] += coeff_b * local_v
            
            for ci in range(2):
                for ck in range(2):
                    coeff_c = c_coeffs.get((ci, ck), 0)
                    if coeff_c == 0: continue
                    for li in range(2):
                        for lk in range(2):
                            local_w = W2[lk*2+li][s]
                            if local_w == 0: continue
                            global_k = ck*2 + lk
                            global_i = ci*2 + li
                            W[global_k*4 + global_i][col] += coeff_c * local_w
                            
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(decomp, n, m, p, num_checks=50):
    """Verify the decomposition."""
    import random
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    rank = len(U[0]) if U else 0
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(val - 1) > 1e-9:
                    return False
    
    random.seed(42)
    for _ in range(num_checks):
        a = random.randint(0, n*m - 1)
        b = random.randint(0, m*p - 1)
        c = random.randint(0, p*n - 1)
        i = a // m; j = a % m
        j2 = b // p; k = b % p
        k2 = c // n; i2 = c % n
        if i == i2 and j == j2 and k == k2:
            continue
        val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
        if abs(val) > 1e-9:
            return False
    
    return True

def _try_find_rank_49_extension(U4, V4, W4, timeout=30.0):
    """
    Try to find V4_ext (4x49) and W4_ext (4x49) such that
    sum_r U4[i,j,r] * V4_ext[j,r] * W4_ext[i,r] = 1 for all i,j.
    Uses structured V_ext patterns and solves linear systems for W_ext,
    then rounds to half-integers and verifies.
    """
    start_time = time.time()
    n = 4
    rank = 49
    
    U_np = np.array(U4, dtype=float)  # 16x49
    
    def _try_v_ext(V_ext):
        """Given V_ext (4x49), solve for W_ext and check if rounding works."""
        W_ext = np.zeros((4, 49))
        for i in range(4):
            # A_i[j, r] = U[i*4+j, r] * V_ext[j, r]
            A_i = np.zeros((4, 49))
            for j in range(4):
                A_i[j, :] = U_np[i*4+j, :] * V_ext[j, :]
            
            # Solve A_i @ W_ext[i,:]^T = ones(4)
            try:
                sol, res, rk, sv = np.linalg.lstsq(A_i, np.ones(4), rcond=None)
                if not np.allclose(A_i @ sol, np.ones(4), atol=1e-10):
                    return None
                W_ext[i, :] = sol
            except:
                return None
        
        # Round to half-integers
        W_ext_rounded = np.round(W_ext * 2) / 2
        
        # Verify exactness with rounded values
        for i in range(4):
            for j in range(4):
                val = float(np.dot(U_np[i*4+j, :], V_ext[j, :] * W_ext_rounded[i, :]))
                if abs(val - 1) > 1e-9:
                    return None
        
        return W_ext_rounded
    
    # Strategy 1: Try V_ext with all entries = 1
    V_ext_ones = np.ones((4, 49))
    result = _try_v_ext(V_ext_ones)
    if result is not None:
        return V_ext_ones, result
    
    # Strategy 2: Try V_ext with structured patterns based on Strassen-like coefficients
    # Use the 7 Strassen steps as basis for V_ext patterns
    strassen_V = [
        [1, 1, 0, 0, 0, 1, 1],   # step 0: B00+B11
        [1, 0, 0, 0, 0, 1, 0],   # step 1: B00
        [0, 0, 1, 0, 0, 0, 1],   # step 2: B10-B00 -> wait, let me use actual V2
        [0, 0, 0, 1, 0, 0, 0],   # step 3
        [0, 0, 0, 0, 1, 0, 0],   # step 4
        [0, 0, 0, 0, 0, 1, 1],   # step 5
        [0, 0, 0, 0, 0, 0, 1],   # step 6
    ]
    # Actually use the real Strassen V2
    U2, V2, W2 = _strassen_2x2x2()
    # V2 is 4x7: V2[j*2+k, s]
    # For 4x4x4, each Strassen step uses a 2x2 block of B
    # For the 5th column (k=4), we need to assign B[j,4] values
    
    # Let's try: V_ext[j, r] where r indexes the 49 columns
    # Column r corresponds to (step, substep) where step in 0..6, substep in 0..6
    # In the 4x4x4 decomposition, column r = step*7 + substep
    # The B factor in step s, substep t involves B blocks
    # For k=4, we need to define how B[j,4] enters
    
    # Simplest: try V_ext[j, r] = c_j where c is a small vector
    # Try various small patterns
    import itertools
    small_vals = [-1, -0.5, 0, 0.5, 1]
    
    # Try all-ones, all-zeros (won't work), and random patterns
    rng = np.random.RandomState(42)
    
    # First try some deterministic patterns
    patterns = []
    # Pattern: V_ext[j, r] = 1 for all
    patterns.append(np.ones((4, 49)))
    # Pattern: V_ext[j, r] = (-1)^j
    for j in range(4):
        row = np.ones(49) * ((-1) ** j)
        patterns.append(row)
    # Combine into list of full V_ext candidates
    v_candidates = []
    # All ones
    v_candidates.append(np.ones((4, 49)))
    # Alternating
    v_alt = np.array([1.0, -1.0, 1.0, -1.0])[:, np.newaxis] * np.ones((1, 49))
    v_candidates.append(v_alt)
    # Random patterns
    for _ in range(200):
        if time.time() - start_time > timeout - 5:
            break
        v_cand = rng.choice([-1, -0.5, 0, 0.5, 1], size=(4, 49))
        v_candidates.append(v_cand)
    
    for v_cand in v_candidates:
        if time.time() - start_time > timeout - 2:
            break
        result = _try_v_ext(v_cand)
        if result is not None:
            return v_cand, result
    
    # Strategy 3: Use sparse V_ext - only a few nonzero entries per row
    for _ in range(500):
        if time.time() - start_time > timeout - 2:
            break
        v_cand = np.zeros((4, 49))
        for j in range(4):
            # Pick 5-10 random columns to be nonzero
            nnz = rng.randint(5, 11)
            cols = rng.choice(49, size=nnz, replace=False)
            v_cand[j, cols] = rng.choice([-1, 1], size=nnz)
        result = _try_v_ext(v_cand)
        if result is not None:
            return v_cand, result
    
    return None

def construct():
    """Return a valid tensor decomposition for 4x4 by 4x5 matrix multiplication."""
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    try:
        result_444 = _strassen_4x4x4()
        
        U4 = result_444["U"]
        V4 = result_444["V"]
        W4 = result_444["W"]
        rank_444 = len(U4[0])
        
        # Try to find rank 49 extension with more time
        remaining_time = DEADLINE - (time.time() - start_time)
        if remaining_time > 15:
            ext_timeout = min(remaining_time - 15, 80)
            ext = _try_find_rank_49_extension(U4, V4, W4, timeout=ext_timeout)
            if ext is not None:
                V_ext, W_ext = ext
                new_rank = rank_444
                U = [row[:] for row in U4]
                V = [[0] * new_rank for _ in range(m * p)]
                W = [[0] * new_rank for _ in range(p * n)]
                
                # Copy V4
                for j in range(4):
                    for k in range(4):
                        for r in range(rank_444):
                            V[j*5 + k][r] = V4[j*4 + k][r]
                
                # Copy W4
                for k in range(4):
                    for i in range(4):
                        for r in range(rank_444):
                            W[k*4 + i][r] = W4[k*4 + i][r]
                            
                # Add extension
                for j in range(4):
                    for r in range(rank_444):
                        V[j*5 + 4][r] = V_ext[j][r]
                        
                for i in range(4):
                    for r in range(rank_444):
                        W[4*4 + i][r] = W_ext[i][r]
                        
                result = {"U": U, "V": V, "W": W}
                
                if _verify_decomposition(result, n, m, p):
                    return result
        
        # Fallback to Rank 50
        new_rank = rank_444 + 1
        U = [[0] * new_rank for _ in range(n * m)]
        V = [[0] * new_rank for _ in range(m * p)]
        W = [[0] * new_rank for _ in range(p * n)]
        
        for i in range(16):
            for r in range(rank_444):
                U[i][r] = U4[i][r]
                
        for j in range(4):
            for k in range(4):
                for r in range(rank_444):
                    V[j*5 + k][r] = V4[j*4 + k][r]
                    
        for k in range(4):
            for i in range(4):
                for r in range(rank_444):
                    W[k*4 + i][r] = W4[k*4 + i][r]
                    
        new_col = rank_444
        for i in range(4):
            for j in range(4):
                U[i*4 + j][new_col] = 1
            V[j*5 + 4][new_col] = 1
            W[4*4 + i][new_col] = 1
        
        result = {"U": U, "V": V, "W": W}
        
        if _verify_decomposition(result, n, m, p):
            return result
            
    except Exception as e:
        pass
    
    # Fallback: schoolbook decomposition (rank 80)
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

# EVOLVE-BLOCK-END
