import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def _laderman_4x4_decomp():
    """
    Return the 21 rank-1 terms for 4x4x4 matrix multiplication using Laderman's algorithm.
    Indices are 0-15 for 4x4 matrices (row-major).
    """
    # Laderman's 21 multiplications
    # A indices: 0..15, B indices: 0..15, C indices: 0..15
    patterns = []
    
    # Helper to add a term
    def add_term(u_dict, v_dict, w_dict):
        patterns.append((u_dict, v_dict, w_dict))

    # M1
    add_term({0:1, 15:1}, {0:1, 15:1}, {0:1, 15:1})
    # M2
    add_term({3:1, 15:1}, {12:1}, {3:-1, 15:1}) # (A14+A44)B41 -> C14-C44+C41? No.
    # Standard Laderman:
    # c11 = M1 - M2 + M3
    # c12 = M4 + M6 - M7
    # c13 = M4 + M7
    # c14 = M3 - M5
    # ...
    
    # I will use a verified list of Laderman patterns.
    # Source: "Fast Matrix Multiplication" by Strassen, or online references for Laderman.
    
    # Re-implementing Laderman patterns explicitly to be safe.
    # A = [a11 a12 a13 a14; a21 ... a44]
    # B = [b11 ... b44]
    
    # M1 = (a11+a44)(b11+b44)
    add_term({0:1, 15:1}, {0:1, 15:1}, {0:1, 15:1})
    
    # M2 = (a14+a44)b41
    add_term({3:1, 15:1}, {12:1}, {3:-1, 15:1}) # c14 gets -M2, c44 gets M2?
    # Correction: c14 = M3 - M5. c44 = M1 - M2 + M10?
    
    # Let's use a known correct set.
    # I will hardcode the 21 patterns based on a reliable source.
    
    # Reset patterns
    patterns = []
    
    # 1
    patterns.append(({0:1, 15:1}, {0:1, 15:1}, {0:1, 15:1}))
    # 2
    patterns.append(({3:1, 15:1}, {12:1}, {3:-1, 15:1}))
    # 3
    patterns.append(({0:1}, {3:1, 15:-1}, {0:1, 3:1}))
    # 4
    patterns.append(({1:1, 0:1}, {12:1}, {1:-1, 4:1})) # (a12+a11)b21? No.
    
    # This is getting risky to guess. I will use the Strassen-based rank 56 if Laderman is too complex to verify quickly.
    # However, rank 42 is a significant improvement.
    # I will try to construct Laderman carefully.
    
    return patterns

def construct():
    # Fallback to rank 56 if Laderman implementation is not ready or fails
    # For this iteration, I will stick to the robust rank 56 Strassen solution 
    # because implementing Laderman correctly without a reference table is error-prone 
    # and might lead to an invalid score (0).
    # The current program already has a working rank 56 solution.
    # I will return the current program's logic.
    
    return _build_rank_56_current()

# Copy of current program logic
def _strassen_2x2_decomp():
    patterns = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1},     {2:1, 3:-1}),
        ({0:1},       {1:1, 3:-1}, {1:1, 3:1}),
        ({3:1},       {2:1, 0:-1}, {0:1, 2:1}),
        ({0:1, 1:1}, {3:1},     {0:-1, 1:1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),
    ]
    return patterns

def _build_rank_56_current():
    n, m, p = DIMENSIONS
    total_rank = 56
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    strassen_patterns = _strassen_2x2_decomp()
    
    for a_block in range(2):
        for b_col_block in range(4):
            a_rows = [0, 1]
            a_cols = [2*a_block, 2*a_block+1]
            
            b_rows = [2*a_block, 2*a_block+1]
            b_cols = [2*b_col_block, 2*b_col_block+1]
            
            c_rows = [0, 1]
            c_cols = [2*b_col_block, 2*b_col_block+1]
            
            base_col = 7 * (4 * a_block + b_col_block)
            
            for s, (u_c, v_c, w_c) in enumerate(strassen_patterns):
                col = base_col + s
                
                for local_idx, val in u_c.items():
                    li = local_idx // 2
                    lj = local_idx % 2
                    global_i = a_rows[li]
                    global_j = a_cols[lj]
                    U[global_i * m + global_j][col] += val
                
                for local_idx, val in v_c.items():
                    lj = local_idx // 2
                    lk = local_idx % 2
                    global_j = b_rows[lj]
                    global_k = b_cols[lk]
                    V[global_j * p + global_k][col] += val
                
                for local_idx, val in w_c.items():
                    li = local_idx // 2
                    lk = local_idx % 2
                    global_i = c_rows[li]
                    global_k = c_cols[lk]
                    W[global_k * n + global_i][col] += val
                    
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition():
    n, m, p = DIMENSIONS
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

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    n, m, p = DIMENSIONS
    
    try:
        if time.time() < deadline:
            result = _build_rank_56_current()
            if _verify(result["U"], result["V"], result["W"], n, m, p, tol=1e-9):
                return result
    except Exception:
        pass
    
    return _schoolbook_decomposition()
