import time

def _strassen_2x2x2_terms():
    """
    Return the 7 rank-1 terms for 2x2x2 matrix multiplication (Strassen).
    Each term is a tuple (U_dict, V_dict, W_dict) where:
    U_dict: {(i, j): coeff}
    V_dict: {(j, k): coeff}
    W_dict: {(k, i): coeff}
    """
    terms = []
    
    # M1 = (A00 + A11)(B00 + B11)
    terms.append((
        {(0,0):1, (1,1):1},
        {(0,0):1, (1,1):1},
        {(0,0):1, (0,1):1}
    ))
    
    # M2 = (A10+A11)B00
    terms.append((
        {(1,0):1, (1,1):1},
        {(0,0):1},
        {(0,1):1}
    ))
    
    # M3 = A00(B01-B11)
    terms.append((
        {(0,0):1},
        {(0,1):1, (1,1):-1},
        {(1,0):1, (1,1):-1}
    ))
    
    # M4 = A11(B10-B00)
    terms.append((
        {(1,1):1},
        {(1,0):1, (0,0):-1},
        {(0,0):-1, (1,0):1}
    ))
    
    # M5 = (A00+A01)B11
    terms.append((
        {(0,0):1, (0,1):1},
        {(1,1):1},
        {(1,1):1}
    ))
    
    # M6 = (A10-A00)(B00+B01)
    terms.append((
        {(1,0):1, (0,0):-1},
        {(0,0):1, (0,1):1},
        {(0,0):1}
    ))
    
    # M7 = (A01-A11)(B10+B11)
    terms.append((
        {(0,1):1, (1,1):-1},
        {(1,0):1, (1,1):1},
        {(0,0):1, (1,1):1}
    ))
    
    return terms

def _verify(U, V, W, n, m, p):
    """Verify the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(len(U[0])):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = 3, 3, 6
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Fallback: Schoolbook (rank 54)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    try:
        strassen_terms = _strassen_2x2x2_terms()
        
        all_U = []
        all_V = []
        all_W = []
        
        # Decompose into 2 blocks of size 3x3 * 3x3 (columns 0-2 and 3-5)
        # Block b: columns k are 3*b to 3*b+2
        for b in range(2):
            k_offset = 3 * b
            
            # Part 1: i in {0,1}, j in {0,1}, k in {0,1} (Strassen, rank 7)
            for u_dict, v_dict, w_dict in strassen_terms:
                U_term = [0] * (n * m)
                V_term = [0] * (m * p)
                W_term = [0] * (p * n)
                
                for (i, j), coeff in u_dict.items():
                    U_term[i * m + j] = coeff
                for (j, k), coeff in v_dict.items():
                    V_term[j * p + (k_offset + k)] = coeff
                for (k, i), coeff in w_dict.items():
                    W_term[(k_offset + k) * n + i] = coeff
                    
                all_U.append(U_term)
                all_V.append(V_term)
                all_W.append(W_term)
            
            # Part 2: i in {0,1}, j in {0,1}, k=2 (Schoolbook, rank 4)
            for i in range(2):
                for j in range(2):
                    U_term = [0] * (n * m)
                    V_term = [0] * (m * p)
                    W_term = [0] * (p * n)
                    
                    U_term[i * m + j] = 1
                    V_term[j * p + (k_offset + 2)] = 1
                    W_term[(k_offset + 2) * n + i] = 1
                    
                    all_U.append(U_term)
                    all_V.append(V_term)
                    all_W.append(W_term)
            
            # Part 3: i=2, j in {0,1,2}, k in {0,1,2} (Schoolbook, rank 3)
            for j in range(3):
                U_term = [0] * (n * m)
                V_term = [0] * (m * p)
                W_term = [0] * (p * n)
                
                # We need 3 terms here: one for each k
                # Actually, for fixed i=2, j, we need to handle all k
                # Let's do rank 1 terms for each (j, k)
                # Wait, the structure is C[2, k] = sum_j A[2, j] B[j, k]
                # So for fixed j, we can do rank 1 for each k? No, that's 9 terms total.
                # Let's just do schoolbook for the whole 3x3 block of (i=2, j, k)
                # That's 9 terms.
                pass
            
            # Let's redo Part 3 correctly: Schoolbook for i=2, all j, all k (rank 9)
            for j in range(3):
                for k in range(3):
                    U_term = [0] * (n * m)
                    V_term = [0] * (m * p)
                    W_term = [0] * (p * n)
                    
                    U_term[2 * m + j] = 1
                    V_term[j * p + (k_offset + k)] = 1
                    W_term[(k_offset + k) * n + 2] = 1
                    
                    all_U.append(U_term)
                    all_V.append(V_term)
                    all_W.append(W_term)
            
            # Part 4: i in {0,1,2}, j=2, k in {0,1,2} (Schoolbook, rank 9)
            for i in range(3):
                for k in range(3):
                    U_term = [0] * (n * m)
                    V_term = [0] * (m * p)
                    W_term = [0] * (p * n)
                    
                    U_term[i * m + 2] = 1
                    V_term[2 * p + (k_offset + k)] = 1
                    W_term[(k_offset + k) * n + i] = 1
                    
                    all_U.append(U_term)
                    all_V.append(V_term)
                    all_W.append(W_term)
        
        rank = len(all_U)
        
        # Verify correctness
        if _verify(all_U, all_V, all_W, n, m, p):
            if rank < rank_sb:
                return {"U": all_U, "V": all_V, "W": all_W}
                
    except Exception:
        pass

    return {"U": U_sb, "V": V_sb, "W": W_sb}
