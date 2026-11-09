import numpy as np
import time
from fractions import Fraction
import sys

# Ensure we have enough recursion depth if we were doing recursive, but here we are iterative
sys.setrecursionlimit(10000)

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # We will construct a rank 46 decomposition.
    # This is based on applying Strassen's algorithm to the 2x2x2 blocks 
    # covering indices 0-3, and using schoolbook for the remaining index 4.
    
    rank = 46
    # Use float for construction as entries are integers
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a standard schoolbook term: U[i*m+j, col]=a, V[j*p+k, col]=b, W[k*n+i, col]=c
    def add_term(i, j, k, a, b, c):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        U[i*m+j, col] += a
        V[j*p+k, col] += b
        W[k*n+i, col] += c
        col += 1

    # Helper to add Strassen block for A rows {ja, jb}, B cols {ka, kb}, C cols {ka, kb}
    # Note: The standard Strassen multiplication of A_{2x2} * B_{2x2} = C_{2x2}
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    # The indices in the global tensor are:
    # U index: i*m + j. Here i is global row (0 or 1), j is global col.
    # In the block, j is either ja or jb.
    # V index: j*p + k. Here j is global row of B (which is col of A), k is global col of B.
    # In the block, j is ja/jb, k is ka/kb.
    # W index: k*n + i. Here k is global row of C (col of B), i is global col of C (row of A).
    # In the block, k is ka/kb, i is 0 or 1 (global row of A).
    
    def add_strassen_block(ja, jb, ka, kb):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        # A00 is U[0, ja], A11 is U[1, jb]
        # B00 is V[ja, ka], B11 is V[jb, kb]
        # C00 is W[ka, 0], C11 is W[kb, 1]
        
        # M1
        U[0*m+ja, col] = 1
        U[1*m+jb, col] = 1
        V[ja*p+ka, col] = 1
        V[jb*p+kb, col] = 1
        W[ka*n+0, col] = 1
        W[kb*n+1, col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        # A10 is U[1, ja], A11 is U[1, jb]
        # B00 is V[ja, ka]
        # C10 is W[ka, 1], C11 is W[kb, 1]
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        U[1*m+ja, col] = 1
        U[1*m+jb, col] = 1
        V[ja*p+ka, col] = 1
        W[ka*n+1, col] = 1
        W[kb*n+1, col] = -1
        col += 1
        
        # M3 = A00(B01-B11)
        # A00 is U[0, ja]
        # B01 is V[ja, kb], B11 is V[jb, kb]
        # C01 is W[kb, 0], C11 is W[kb, 1]
        # C01 = M3 + M5
        # C11 = ... + M3 ...
        
        U[0*m+ja, col] = 1
        V[ja*p+kb, col] = 1
        V[jb*p+kb, col] = -1
        W[kb*n+0, col] = 1
        W[kb*n+1, col] = 1
        col += 1
        
        # M4 = A11(B10-B00)
        # A11 is U[1, jb]
        # B10 is V[jb, ka], B00 is V[ja, ka]
        # C00 is W[ka, 0], C10 is W[ka, 1]
        # C00 = M1 + M4 - M5 + M7
        # C10 = M2 + M4
        
        U[1*m+jb, col] = 1
        V[jb*p+ka, col] = 1
        V[ja*p+ka, col] = -1
        W[ka*n+0, col] = 1
        W[ka*n+1, col] = 1
        col += 1
        
        # M5 = (A00+A01)B11
        # A00 is U[0, ja], A01 is U[0, jb]
        # B11 is V[jb, kb]
        # C00 is W[ka, 0], C01 is W[kb, 0]
        # C00 = ... - M5 ...
        # C01 = ... + M5 ...
        
        U[0*m+ja, col] = 1
        U[0*m+jb, col] = 1
        V[jb*p+kb, col] = 1
        W[ka*n+0, col] = -1
        W[kb*n+0, col] = 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        # A10 is U[1, ja], A00 is U[0, ja]
        # B00 is V[ja, ka], B01 is V[ja, kb]
        # C11 is W[kb, 1]
        # C11 = ... + M6 ...
        
        U[1*m+ja, col] = 1
        U[0*m+ja, col] = -1
        V[ja*p+ka, col] = 1
        V[ja*p+kb, col] = 1
        W[kb*n+1, col] = 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        # A01 is U[0, jb], A11 is U[1, jb]
        # B10 is V[jb, ka], B11 is V[jb, kb]
        # C00 is W[ka, 0]
        # C00 = ... + M7 ...
        
        U[0*m+jb, col] = 1
        U[1*m+jb, col] = -1
        V[jb*p+ka, col] = 1
        V[jb*p+kb, col] = 1
        W[ka*n+0, col] = 1
        col += 1

    # Apply Strassen to the 2x2 blocks of the 4x4 core
    add_strassen_block(0, 1, 0, 1)
    add_strassen_block(2, 3, 0, 1)
    add_strassen_block(0, 1, 2, 3)
    add_strassen_block(2, 3, 2, 3)
    
    # Handle the 5th index (index 4) with schoolbook terms
    # Terms involving A[:, 4] or B[:, 4] or B[4, :]
    
    # 1. A[i, 4] * B[4, k] -> C[i, k] for k in 0..3
    for i in [0, 1]:
        for k in [0, 1, 2, 3]:
            add_term(i, 4, k, 1, 1, 1)
            
    # 2. A[i, j] * B[j, 4] -> C[i, 4] for j in 0..3
    for i in [0, 1]:
        for j in [0, 1, 2, 3]:
            add_term(i, j, 4, 1, 1, 1)
            
    # 3. A[i, 4] * B[4, 4] -> C[i, 4]
    for i in [0, 1]:
        add_term(i, 4, 4, 1, 1, 1)
        
    # Total rank should be 28 + 8 + 8 + 2 = 46
    if col != 46:
        # Fallback if construction logic changes
        pass

    # Verification
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    T_target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Check for errors
    error = np.max(np.abs(T_recon - T_target))
    
    if error > 1e-9:
        # Fallback to schoolbook rank 50
        rank_school = n * m * p
        U_school = np.zeros((n*m, rank_school), dtype=float)
        V_school = np.zeros((m*p, rank_school), dtype=float)
        W_school = np.zeros((p*n, rank_school), dtype=float)
        
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j, c] = 1
                    V_school[j*p+k, c] = 1
                    W_school[k*n+i, c] = 1
                    c += 1
                    
        return {"U": U_school.tolist(), 
                "V": V_school.tolist(), 
                "W": W_school.tolist()}
    
    return {"U": U.tolist(), 
            "V": V.tolist(), 
            "W": W.tolist()}
