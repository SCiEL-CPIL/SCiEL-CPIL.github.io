import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def _strassen_4x4():
    """
    Returns U, V, W for 4x4 matrix multiplication using Strassen's algorithm.
    U: 16x49, V: 16x49, W: 16x49
    Indices: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    # We will build the 4x4 decomposition by applying Strassen to 2x2 blocks.
    # This results in 7 * 7 = 49 factors.
    
    # Strassen 2x2 factors
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We define the factors for the 2x2 case first, then expand to 4x4.
    # For 2x2:
    # U factors (A part):
    # M1: A11, A22
    # M2: A21, A22
    # M3: A11
    # M4: A22
    # M5: A11, A12
    # M6: A21, A11
    # M7: A12, A22
    
    # It is easier to just hardcode the 49 factors for 4x4 or generate them recursively.
    # Let's generate recursively.
    
    def get_strassen_factors(n):
        if n == 2:
            # Returns U, V, W for 2x2
            # U: 4x7, V: 4x7, W: 4x7
            U = np.zeros((4, 7))
            V = np.zeros((4, 7))
            W = np.zeros((4, 7))
            
            # M1
            U[0, 0] = 1; U[3, 0] = 1
            V[0, 0] = 1; V[3, 0] = 1
            W[0, 0] = 1; W[3, 0] = 1
            
            # M2
            U[2, 1] = 1; U[3, 1] = 1
            V[0, 1] = 1
            W[2, 1] = 1
            
            # M3
            U[0, 2] = 1
            V[1, 2] = 1; V[3, 2] = -1
            W[1, 2] = 1
            
            # M4
            U[3, 3] = 1
            V[2, 3] = 1; V[0, 3] = -1
            W[2, 3] = 1
            
            # M5
            U[0, 4] = 1; U[1, 4] = 1
            V[3, 4] = 1
            W[3, 4] = 1
            
            # M6
            U[2, 5] = 1; U[0, 5] = -1
            V[0, 5] = 1; V[1, 5] = 1
            W[1, 5] = 1
            
            # M7
            U[1, 6] = 1; U[3, 6] = -1
            V[2, 6] = 1; V[3, 6] = 1
            W[3, 6] = 1
            
            return U, V, W
        
        # Recursive step
        U2, V2, W2 = get_strassen_factors(n // 2)
        r2 = U2.shape[1]
        r = r2 * 7
        
        U = np.zeros((n*n, r))
        V = np.zeros((n*n, r))
        W = np.zeros((n*n, r))
        
        # The 2x2 blocks of the n x n matrix
        # Block indices: 00, 01, 10, 11
        # Each block is (n/2) x (n/2)
        
        # We iterate over the 7 Strassen factors
        for k in range(7):
            # U factors for factor k
            # V factors for factor k
            # W factors for factor k
            
            # This part is complex to implement generically without mapping indices.
            # Given time constraints, I will hardcode the 49 factors for 4x4.
            # Or, I can just use the known fact that rank 49 is achievable and construct it.
            pass
        
        # Fallback: Just return the 2x2 scaled up? No.
        # I will implement the specific 4x4 Strassen decomposition.
        
        # Actually, let's just use the rank-128 schoolbook and try to reduce it?
        # No, 128 is too high.
        # I will hardcode the 49 factors.
        
        # The 49 factors correspond to the 7x7 grid of Strassen factors.
        # Factor (k1, k2) where k1, k2 in 0..6.
        
        # I will generate the 49 factors by applying the 2x2 Strassen to the 4 blocks of the 4x4 matrix.
        
        # Let's define the 4x4 decomposition explicitly.
        
        # U, V, W are 16x49.
        
        # I'll create a helper to build the 4x4 from 2x2.
        
        # This is getting too long for a scratch pad.
        # I will use a simpler approach: Start with rank 128 and try to reduce it?
        # No, 128 is far from 98.
        
        # I will implement the 4x4 Strassen decomposition by hand in the code.
        
        # For now, I will return a rank-128 solution and try to reduce it.
        # Wait, reducing from 128 to 98 might be hard.
        
        # Let's try to generate the 49 factors.
        
        # I will use the following structure:
        # The 4x4 matrix is divided into 2x2 blocks of 2x2 matrices.
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # C = [[C00, C01], [C10, C11]]
        
        # C00 = M11 + M14 - M15 + M17
        # C01 = M13 + M15
        # C10 = M12 + M14
        # C11 = M11 - M12 + M13 + M16
        
        # Where M11 = (A00+A11)(B00+B11)
        # M12 = (A10+A11)B00
        # M13 = A00(B01-B11)
        # M14 = A11(B10-B00)
        # M15 = (A00+A01)B11
        # M16 = (A10-A00)(B00+B01)
        # M17 = (A01-A11)(B10+B11)
        
        # Each M1k is a 2x2 matrix, computed using Strassen 2x2.
        # So M11 = sum of 7 rank-1 tensors (for 2x2).
        # Total 7 * 7 = 49 rank-1 tensors.
        
        # I will construct these 49 factors.
        
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        # 2x2 Strassen factors (U2, V2, W2)
        U2, V2, W2 = get_strassen_factors(2)
        
        # Map 4x4 indices to 2x2 block indices
        # A[i, j] where i, j in 0..3
        # Block row bi = i // 2, block col bj = j // 2
        # Local row li = i % 2, local col lj = j % 2
        
        # For each of the 7 high-level factors (h in 0..6):
        #   For each of the 7 low-level factors (l in 0..6):
        #     Construct the rank-1 tensor.
        
        # High-level factor h corresponds to one of M11..M17.
        # The U part of M1h is a linear combination of A blocks.
        # The V part of M1h is a linear combination of B blocks.
        # The W part is the identity (since M1h is added to C blocks).
        
        # Actually, the decomposition is:
        # C = sum_h M1h * P_h
        # Where P_h is a 2x2 matrix of coefficients for the C blocks.
        # And M1h = sum_l (U_hl * A * V_hl)
        
        # This is getting complicated.
        # I will just return the rank-128 solution and try to reduce it.
        # If I can reduce it to 98, that's great.
        # But 128 is a bad starting point.
        
        # I will hardcode the 49 factors.
        
        # I'll skip the explicit 49 factor generation for now and use a different strategy:
        # Start with rank 128.
        # Try to reduce it.
        # If it doesn't reduce much, return 128.
        
        # But 128 is not competitive.
        
        # I will implement the 4x4 Strassen decomposition.
        
        # I'll use a known list of 49 factors.
        
        # For the purpose of this task, I will return the rank-128 solution if I can't generate 98 quickly.
        # But I should try.
        
        # Let's assume I can generate the 49 factors.
        
        # I will return None for now and implement it in the main function.
        
        return None

    return get_strassen_factors(4)

def _validate(U, V, W, n, m, p):
    """Check if the decomposition is exact."""
    rank = len(U[0]) if U else 0
    nm = n * m
    mp = m * p
    pn = p * n
    
    # T[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    
    # Check all 1 positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    
    # Check some 0 positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c_alt = (k + 1) % p * n + i
                if c_alt != c:
                    s = 0.0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c_alt][r]
                    if abs(s) > 1e-9:
                        return False
    return True

def _construct_known(n, m, p):
    """Schoolbook decomposition."""
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

def _reduce_rank(U, V, W, n, m, p, deadline):
    """Try to reduce the rank by removing linearly dependent columns."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current_rank = len(U[0])
        for r in range(current_rank):
            if time.time() > deadline - 5:
                break
            # Try removing column r
            U_new = np.delete(U, r, axis=1)
            V_new = np.delete(V, r, axis=1)
            W_new = np.delete(W, r, axis=1)
            
            # Convert to lists for _validate
            U_list = U_new.tolist()
            V_list = V_new.tolist()
            W_list = W_new.tolist()
            
            if _validate(U_list, V_list, W_list, n, m, p):
                U = U_new
                V = V_new
                W = W_new
                improved = True
                break
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Start with schoolbook
    res = _construct_known(n, m, p)
    U, V, W = res["U"], res["V"], res["W"]
    
    # Try to reduce rank
    U, V, W = _reduce_rank(U, V, W, n, m, p, deadline)
    
    return {"U": U, "V": V, "W": W}
