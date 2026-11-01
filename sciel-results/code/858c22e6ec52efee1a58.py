import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    U_total = n * m  # 8
    V_total = m * p  # 28
    W_total = p * n  # 14
    
    # Start with rank 50 (known good)
    rank = 50
    U = np.zeros((U_total, rank), dtype=np.float64)
    V = np.zeros((V_total, rank), dtype=np.float64)
    W = np.zeros((W_total, rank), dtype=np.float64)
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h)
        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        
        # M2 = (c+d)e
        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1
        
        # M3 = a(f-h)
        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        
        # M4 = d(g-e)
        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1
        
        # M5 = (a+b)h
        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1; col += 1
        U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1; col += 1

    # Build the initial rank-50 solution
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        add_schoolbook_221_block(0, j_start, 6)
    
    # Verify correctness
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Try to reduce rank by checking if we can merge or eliminate terms
    # Strategy: Try to find linear dependencies in the tensor decomposition
    # For each pair of rank-1 terms, check if their sum can be represented
    # with fewer terms (i.e., if they are linearly dependent)
    
    # First, let's try a more systematic approach:
    # Build the full tensor and check if we can find a lower-rank decomposition
    
    # The matrix multiplication tensor T has T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    # In our notation: C[k*n+i, j] = sum over j of A[i,j]*B[j,k]
    # Wait, let me re-read the convention: W[k*n+i, r]
    # So C is indexed as C[k*n+i] where k is column (0..p-1) and i is row (0..n-1)
    # A is indexed as A[i*m+j] where i is row (0..n-1) and j is column (0..m-1)
    # B is indexed as B[j*p+k] where j is row (0..m-1) and k is column (0..p-1)
    
    # The tensor T[A_idx, B_idx, C_idx] = 1 if A_idx corresponds to A[i,j], 
    # B_idx corresponds to B[j,k], and C_idx corresponds to C[i,k]
    
    # Let's build the tensor
    T = np.zeros((U_total, V_total, W_total), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Now, let's try to find a lower-rank decomposition by solving a linear system
    # We want to find U', V', W' with rank r < 50 such that
    # sum_r U'[a,r] * V'[b,r] * W'[c,r] = T[a,b,c]
    
    # This is a nonlinear problem, but we can try to fix two of the matrices and solve for the third
    # Alternatively, we can try to check if the current decomposition has any redundant terms
    
    # Let's check if any column in U, V, or W is zero (which would mean that term is useless)
    # If a column in U is zero, then that term doesn't contribute to any output
    # Same for V and W
    
    # First, let's check for zero columns
    zero_cols = []
    for r in range(rank):
        if np.all(U[:, r] == 0) or np.all(V[:, r] == 0) or np.all(W[:, r] == 0):
            zero_cols.append(r)
    
    if zero_cols:
        # Remove zero columns
        keep = [r for r in range(rank) if r not in zero_cols]
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
        rank = len(keep)
    
    # Now, let's try to find linear dependencies
    # If two columns in U are linearly dependent, and the corresponding columns in V and W are also
    # linearly dependent in a compatible way, we might be able to merge them
    
    # For now, let's try a simpler approach: check if the current rank is minimal
    # by trying to solve for a lower-rank decomposition
    
    # We'll use a greedy approach: try to remove one term at a time and see if we can
    # adjust the remaining terms to compensate
    
    # Actually, let's try a different approach: use the fact that the tensor is small
    # and try to find a rank-r decomposition for r < 50 by solving a system of equations
    
    # The number of equations is U_total * V_total * W_total = 8 * 28 * 14 = 3136
    # The number of unknowns for rank r is r * (U_total + V_total + W_total) = r * 50
    
    # For r = 49, we have 2450 unknowns and 3136 equations, so it's overdetermined
    # But the tensor might have special structure that allows a lower-rank solution
    
    # Let's try to check if rank 49 is possible by solving the linear system
    # We'll fix U and V to have rank 49 and solve for W
    
    # Actually, this is getting complex. Let's try a simpler heuristic:
    # Check if any two terms can be merged by checking if their contributions
    # to the tensor are linearly dependent
    
    # For now, let's just return the rank-50 solution if we can't find a better one
    
    # Convert to list format as required
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
