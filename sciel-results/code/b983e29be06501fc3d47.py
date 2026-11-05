"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]
DEADLINE = 120.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _remaining():
    return DEADLINE - _elapsed() - 5.0  # 5s safety margin

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using numpy."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    if Ua.shape != (n*m, rank) or Va.shape != (m*p, rank) or Wa.shape != (p*n, rank):
        return False
    
    # Check all entries are integers or half-integers
    for arr in [Ua, Va, Wa]:
        if not np.allclose(2*arr, np.round(2*arr), atol=1e-9):
            return False
    
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _add_strassen_block(U, V, W, i_off, j_off, k_off):
    """
    Adds Strassen decomposition for a 2x2 * 2x2 block at offsets.
    U, V, W are lists of lists. We append to them.
    """
    # Local indices 0,1 map to global i_off+0, i_off+1, etc.
    # Strassen terms:
    # M1: U(0,0)=1, U(1,1)=1; V(0,0)=1, V(1,1)=1; W(0,0)=1, W(1,1)=1
    # M2: U(1,0)=1, U(1,1)=1; V(0,0)=1; W(0,1)=1
    # M3: U(0,0)=1; V(0,1)=1, V(1,1)=-1; W(1,0)=1, W(1,1)=1
    # M4: U(1,1)=1; V(1,0)=1, V(0,0)=-1; W(0,0)=1, W(0,1)=1
    # M5: U(0,0)=1, U(0,1)=1; V(1,1)=1; W(0,0)=-1, W(1,0)=1
    # M6: U(1,0)=1, U(0,0)=-1; V(0,0)=1, V(0,1)=1; W(1,1)=1
    # M7: U(0,1)=1, U(1,1)=-1; V(1,0)=1, V(1,1)=1; W(0,0)=1
    
    terms = [
        # (U_dict, V_dict, W_dict)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
    ]
    
    for u_dict, v_dict, w_dict in terms:
        # Append new column
        u_row = [0] * (2 * 2) # 4 entries
        v_row = [0] * (2 * 2)
        w_row = [0] * (2 * 2)
        
        for (li, lj), val in u_dict.items():
            gi = i_off + li
            gj = j_off + lj
            # U index is gi*m + gj? No, U is local to the block in terms of shape, 
            # but we are appending to the global U list which has rows for all (i,j).
            # Wait, the global U has rows 0..14.
            # We need to append a value to the row corresponding to (gi, gj).
            # But U is a list of lists. We can't just append a value to a specific row 
            # if we are adding a new column.
            # We need to restructure: U is list of rows. Each row is list of values.
            # When we add a new rank term, we add a new element to EVERY row.
            
            pass # This logic is flawed for list-of-lists appending.

def _construct_strassen_block(n, m, p, i_off, j_off, k_off, U, V, W):
    """
    Recursively decompose block.
    U, V, W are lists of rows. We will build them as lists of lists.
    """
    if n == 2 and m == 2 and p == 2:
        # Strassen
        terms = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1}),
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
        ]
        for u_dict, v_dict, w_dict in terms:
            # Append to global U, V, W
            # We need to add 1 to the specific row in U, V, W for this term.
            # But U, V, W are growing. 
            # It is easier to accumulate the terms in temporary lists and then merge.
            # Or, since we know the total rank in advance, we can pre-allocate.
            # But rank is dynamic.
            # Let's use a list of lists and append to each row.
            
            # For U: rows are i*m+j. 
            # We are adding a new column.
            # We must append a value to EVERY row in U.
            
            # This is inefficient if done one by one.
            # Better: Collect all terms for this block, then add them to the global structures.
            pass

def construct():
    n, m, p = DIMENSIONS
    
    # We will build U, V, W as lists of lists.
    # To make it easy to add columns, we will store them as lists of rows, 
    # and when we add a term, we append to all rows.
    
    # However, the number of rows is fixed (15, 30, 18).
    # We can initialize U, V, W with empty lists for each row.
    U = [[] for _ in range(n * m)]
    V = [[] for _ in range(m * p)]
    W = [[] for _ in range(p * n)]
    
    def add_term(u_val, v_val, w_val, i_idx, j_idx, k_idx):
        # u_val, v_val, w_val are dicts mapping local (li, lj) etc to coeff.
        # But we need to add to specific global rows.
        pass

    # This approach is getting messy with the list appending.
    # Let's use numpy arrays and pre-allocate with a large rank, then trim.
    # Or, simply generate the lists of tuples (row_idx, col_val) and then build the matrix.
    
    # Let's use a different structure:
    # U_data: list of (row, col, val)
    # V_data: list of (row, col, val)
    # W_data: list of (row, col, val)
    
    U_data = []
    V_data = []
    W_data = []
    current_rank = 0
    
    def add_strassen(i_off, j_off, k_off):
        nonlocal current_rank
        terms = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1}),
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
        ]
        for u_d, v_d, w_d in terms:
            r = current_rank
            current_rank += 1
            for (li, lj), val in u_d.items():
                U_data.append(((i_off + li) * m + (j_off + lj), r, val))
            for (lj, lk), val in v_d.items():
                V_data.append(((j_off + lj) * p + (k_off + lk), r, val))
            for (lk, li), val in w_d.items():
                W_data.append(((k_off + lk) * n + (i_off + li), r, val))

    def add_schoolbook(i_off, j_off, k_off, n, m, p):
        nonlocal current_rank
        for li in range(n):
            for lj in range(m):
                for lk in range(p):
                    r = current_rank
                    current_rank += 1
                    U_data.append(((i_off + li) * m + (j_off + lj), r, 1))
                    V_data.append(((j_off + lj) * p + (k_off + lk), r, 1))
                    W_data.append(((k_off + lk) * n + (i_off + li), r, 1))

    def decompose(i_off, j_off, k_off, n, m, p):
        if n == 2 and m == 2 and p == 2:
            add_strassen(i_off, j_off, k_off)
        elif n == 1:
            add_schoolbook(i_off, j_off, k_off, n, m, p)
        else:
            # Specific splits for 3,5,6
            if n == 3 and m == 5 and p == 6:
                # Split n: 2, 1
                # Split p: 4, 2
                decompose(i_off, j_off, k_off, 2, 5, 4)
                decompose(i_off, j_off, k_off + 4, 2, 5, 2)
                decompose(i_off + 2, j_off, k_off, 1, 5, 4)
                decompose(i_off + 2, j_off, k_off + 4, 1, 5, 2)
            elif n == 2 and m == 5 and p == 4:
                # Split m: 2, 3
                decompose(i_off, j_off, k_off, 2, 2, 4)
                decompose(i_off, j_off + 2, k_off, 2, 3, 4)
            elif n == 2 and m == 5 and p == 2:
                # Split m: 2, 3
                decompose(i_off, j_off, k_off, 2, 2, 2)
                decompose(i_off, j_off + 2, k_off, 2, 3, 2)
            elif n == 2 and m == 2 and p == 4:
                # Split p: 2, 2
                decompose(i_off, j_off, k_off, 2, 2, 2)
                decompose(i_off, j_off, k_off + 2, 2, 2, 2)
            elif n == 2 and m == 3 and p == 4:
                # Split m: 1, 2
                decompose(i_off, j_off, k_off, 2, 1, 4)
                decompose(i_off, j_off + 1, k_off, 2, 2, 4)
            elif n == 2 and m == 3 and p == 2:
                # Split m: 1, 2
                decompose(i_off, j_off, k_off, 2, 1, 2)
                decompose(i_off, j_off + 1, k_off, 2, 2, 2)
            else:
                # Fallback schoolbook
                add_schoolbook(i_off, j_off, k_off, n, m, p)

    decompose(0, 0, 0, n, m, p)
    
    # Build U, V, W matrices
    U = [[0] * current_rank for _ in range(n * m)]
    V = [[0] * current_rank for _ in range(m * p)]
    W = [[0] * current_rank for _ in range(p * n)]
    
    for row, col, val in U_data:
        U[row][col] = val
    for row, col, val in V_data:
        V[row][col] = val
    for row, col, val in W_data:
        W[row][col] = val
        
    if _verify_fast(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        return _schoolbook(n, m, p)
