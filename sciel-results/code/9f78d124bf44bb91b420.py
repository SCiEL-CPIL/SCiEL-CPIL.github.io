"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random
from fractions import Fraction

_start_time = time.time()
_TIME_LIMIT = 145  # seconds, 5s safety margin

def _verify_decomposition(n, m, p, U, V, W, num_tests=5):
    """
    Verify if U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r]*A[i,j]) * (sum_j V[j*p+k, r]*B[j,k])
    """
    R = len(U[0])
    for _ in range(num_tests):
        A = np.random.randint(-3, 4, size=(n, m))
        B = np.random.randint(-3, 4, size=(m, p))
        C_direct = A @ B
        
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        U_reshaped = U_arr.reshape(n, m, R)
        V_reshaped = V_arr.reshape(m, p, R)
        W_reshaped = W_arr.reshape(p, n, R)
        
        LA = np.einsum('ij,ijr->ir', A, U_reshaped)
        LB = np.einsum('jk,jkr->kr', B, V_reshaped)
        C_decomp = np.einsum('kir,ir,kr->ik', W_reshaped, LA, LB)
        
        if not np.allclose(C_direct, C_decomp, atol=1e-9):
            return False
    return True

def _strassen_2x2x2():
    """
    Return Strassen's rank-7 decomposition for 2x2 * 2x2 multiplication.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 = A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4 = A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    return U, V, W

def _find_rank7_2x3x2():
    """
    Try to find a rank-7 decomposition for 2x3 * 3x2 multiplication.
    We'll set up a linear system and solve it.
    
    The tensor T[i,j,k] = 1 if (i,j,k) is a valid index, else 0.
    We want: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    for i in {0,1}, j in {0,1,2}, k in {0,1}
    
    This is a nonlinear system. Instead, we'll use a bilinear approach:
    Fix U and V, then solve for W linearly. Or use a different strategy.
    
    Alternative: use the fact that this is a small tensor and try random search
    with verification.
    """
    # Let's try a structured approach inspired by Strassen
    # Group the 3 middle indices into pairs
    
    # Try: use 7 products
    # M1 = A[0,0]*(B[0,0]+B[1,0]+B[0,1]+B[1,1])  -- no, this doesn't work cleanly
    
    # Let me try a different grouping:
    # Think of A as [[a,b,c],[d,e,f]] and B as [[g,h],[i,j],[k,l]]
    # C[0,0] = ag + bi + ck
    # C[0,1] = ah + bj + cl
    # C[1,0] = dg + ei + fk
    # C[1,1] = dh + ej + fl
    
    # Try Strassen-like:
    # M1 = (a+e)*(g+i+k)  -- no, this mixes things up
    
    # Let me just do a random search with a tight budget
    n_local, m_local, p_local = 2, 3, 2
    R = 7
    
    for attempt in range(2000):
        if time.time() - _start_time > _TIME_LIMIT - 30:
            break
            
        # Generate random sparse U, V, W with entries in {-1, 0, 1}
        U = [[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(n_local * m_local)]
        V = [[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(m_local * p_local)]
        W = [[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(p_local * n_local)]
        
        # Verify
        A = np.random.randint(-3, 4, size=(n_local, m_local))
        B = np.random.randint(-3, 4, size=(m_local, p_local))
        C_direct = A @ B
        
        U_arr = np.array(U).reshape(n_local, m_local, R)
        V_arr = np.array(V).reshape(m_local, p_local, R)
        W_arr = np.array(W).reshape(p_local, n_local, R)
        
        LA = np.einsum('ij,ijr->ir', A, U_arr)
        LB = np.einsum('jk,jkr->kr', B, V_arr)
        C_decomp = np.einsum('kir,ir,kr->ik', W_arr, LA, LB)
        
        if np.allclose(C_direct, C_decomp, atol=1e-9):
            # Verify with more tests
            if _verify_decomposition(n_local, m_local, p_local, U, V, W, num_tests=5):
                return U, V, W
    
    return None

def construct():
    global _start_time
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    # Schoolbook solution (always valid fallback)
    schoolbook_rank = n * m * p  # 36
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Structured decomposition: split 6 columns into 3 pairs
    # For each pair of columns, use rank-1 for A[:,0]*B[0,:] + rank-7 Strassen for A[:,1:3]*B[1:3,:]
    # Total rank: 3 * (1 + 7) = 24
    
    U_struct = [[0] * 24 for _ in range(n * m)]  # (6, 24)
    V_struct = [[0] * 24 for _ in range(m * p)]  # (18, 24)
    W_struct = [[0] * 24 for _ in range(p * n)]  # (12, 24)
    
    col = 0
    for block in range(3):
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # Part 1: A[:,0] * B[0, k0:k1+1] (rank 1)
        for i in range(n):
            U_struct[i * m + 0][col] = 1
        for k in range(k0, k1 + 1):
            V_struct[0 * p + k][col] = 1
        for i in range(n):
            for k in range(k0, k1 + 1):
                W_struct[k * n + i][col] = 1
        col += 1
        
        # Part 2: A[:,1:3] * B[1:3, k0:k1+1] using Strassen (rank 7)
        U_str, V_str, W_str = _strassen_2x2x2()
        
        for r in range(7):
            for i_prime in range(2):
                for j_prime in range(2):
                    global_i = i_prime
                    global_j = j_prime + 1
                    u_val = U_str[i_prime * 2 + j_prime][r]
                    if u_val != 0:
                        U_struct[global_i * m + global_j][col + r] = u_val
            
            for j_prime in range(2):
                for k_prime in range(2):
                    global_j = j_prime + 1
                    global_k = k0 + k_prime
                    v_val = V_str[j_prime * 2 + k_prime][r]
                    if v_val != 0:
                        V_struct[global_j * p + global_k][col + r] = v_val
            
            for k_prime in range(2):
                for i_prime in range(2):
                    global_k = k0 + k_prime
                    global_i = i_prime
                    w_val = W_str[k_prime * 2 + i_prime][r]
                    if w_val != 0:
                        W_struct[global_k * n + global_i][col + r] = w_val
        col += 7
    
    if _verify_decomposition(n, m, p, U_struct, V_struct, W_struct, num_tests=10):
        best_solution = {"U": U_struct, "V": V_struct, "W": W_struct}
        best_rank = 24
    
    # Try to find a rank-7 decomposition for 2x3*3x2 to get total rank 21
    if time.time() - _start_time < _TIME_LIMIT - 20:
        result_2x3x2 = _find_rank7_2x3x2()
        if result_2x3x2 is not None:
            U_2x3x2, V_2x3x2, W_2x3x2 = result_2x3x2
            R_block = 7
            total_rank = 3 * R_block
            
            U_new = [[0] * total_rank for _ in range(n * m)]
            V_new = [[0] * total_rank for _ in range(m * p)]
            W_new = [[0] * total_rank for _ in range(p * n)]
            
            col = 0
            for block in range(3):
                k0 = 2 * block
                k1 = 2 * block + 1
                
                for r in range(R_block):
                    # U: local (i, j) -> global (i, j)
                    for i in range(2):
                        for j in range(3):
                            u_val = U_2x3x2[i * 3 + j][r]
                            if u_val != 0:
                                U_new[i * m + j][col + r] = u_val
                    
                    # V: local (j, k') -> global (j, k0+k')
                    for j in range(3):
                        for k_prime in range(2):
                            global_k = k0 + k_prime
                            v_val = V_2x3x2[j * 2 + k_prime][r]
                            if v_val != 0:
                                V_new[j * p + global_k][col + r] = v_val
                    
                    # W: local (k', i) -> global (k0+k', i)
                    for k_prime in range(2):
                        for i in range(2):
                            global_k = k0 + k_prime
                            w_val = W_2x3x2[k_prime * 2 + i][r]
                            if w_val != 0:
                                W_new[global_k * n + i][col + r] = w_val
                col += R_block
            
            if _verify_decomposition(n, m, p, U_new, V_new, W_new, num_tests=10):
                best_solution = {"U": U_new, "V": V_new, "W": W_new}
                best_rank = total_rank
    
    return best_solution
# EVOLVE-BLOCK-END
