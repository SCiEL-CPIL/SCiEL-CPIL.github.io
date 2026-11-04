import numpy as np
import time
import sys
from fractions import Fraction
import random

DIMENSIONS = [4, 4, 7]
_START_TIME = time.time()
TIME_LIMIT = 145

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _verify_decomposition(U, V, W):
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Check shapes
    if U.shape != (n*m, U.shape[1]): return False
    if V.shape != (m*p, V.shape[1]): return False
    if W.shape != (p*n, W.shape[1]): return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]: return False
    
    R = U.shape[1]
    
    # Check integer/half-integer
    for mat in [U, V, W]:
        if not np.allclose(mat * 2, np.round(mat * 2)):
            return False
            
    # Check tensor equation
    # T_{u, v, w} = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    # Sum_r U[u,r] V[v,r] W[w,r] == T[u,v,w]
    
    # Construct T as a sparse tensor or just check the equation
    # Equation: Sum_v V[v,r] * (Sum_u U[u,r] * T[u,v,w]) ? No.
    # Sum_r U[u,r] V[v,r] W[w,r] = T[u,v,w]
    
    # This is a 3-way sum.
    # We can compute the product tensor P[u,v,w] = Sum_r U[u,r] V[v,r] W[w,r]
    # and check if P == T.
    
    # Efficiently:
    # For each r, the term is U[:,r] * V[:,r] * W[:,r] (outer product)
    # P = sum_r np.einsum('ir,jr,kr->ijk', U, V, W)
    # But U, V, W are 2D.
    # P[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
    
    # Let's use einsum
    P = np.einsum('ur,vr,wr->uvw', U, V, W)
    
    # Construct T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
                
    if not np.allclose(P, T):
        return False
        
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Time guard: ensure we complete within deadline
    if _time_remaining() < 5:
        # Emergency: return minimal valid decomposition (schoolbook rank 112)
        U = [[0.0] * 112 for _ in range(n * m)]
        V = [[0.0] * 112 for _ in range(m * p)]
        W = [[0.0] * 112 for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx][r] = 1.0
                    V[v_idx][r] = 1.0
                    W[w_idx][r] = 1.0
                    r += 1
        return {"U": U, "V": V, "W": W}

    # Fall back to known rank 93 solution
    total_rank = 49 + 44
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    current_rank = 0
    
    # --- Part 1: 4x4x4 using Strassen (Rank 49) ---
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}),       # M2
        ({0: 1}, {1: 1, 3: -1}),      # M3
        ({3: 1}, {2: 1, 0: -1}),      # M4
        ({0: 1, 1: 1}, {3: 1}),       # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}),# M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}) # M7
    ]
    
    sub_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # p0
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),            # p1
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),           # p2
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),           # p3
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),            # p4
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}), # p5
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})  # p6
    ]
    
    for m_idx in range(7):
        x_def, y_def = strassen_terms[m_idx]
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub = sub_terms[t_idx]
            
            u_coeffs = [0.0] * (n*m)
            v_coeffs = [0.0] * (m*p)
            
            for (li, lj), c_x in x_sub.items():
                for b, coeff_block in x_def.items():
                    bi = b // 2
                    bj = b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    u_idx = i * m + j
                    u_coeffs[u_idx] += c_x * coeff_block
            
            for (li, lj), c_y in y_sub.items():
                for b, coeff_block in y_def.items():
                    bi = b // 2
                    bj = b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    v_idx = i * p + j
                    v_coeffs[v_idx] += c_y * coeff_block
            
            for i in range(n*m):
                U[i][r] = u_coeffs[i]
            for j in range(m*p):
                V[j][r] = v_coeffs[j]
            
            w_coeffs = [0.0] * (p*n)
            
            cm_coeffs = {
                (0,0): 0.0, (0,1): 0.0, (1,0): 0.0, (1,1): 0.0
            }
            
            if t_idx == 0:
                cm_coeffs[(0,0)] += 1.0
                cm_coeffs[(1,1)] += 1.0
            elif t_idx == 1:
                cm_coeffs[(1,0)] += 1.0
                cm_coeffs[(1,1)] += -1.0
            elif t_idx == 2:
                cm_coeffs[(0,1)] += 1.0
                cm_coeffs[(1,1)] += 1.0
            elif t_idx == 3:
                cm_coeffs[(0,0)] += 1.0
                cm_coeffs[(1,0)] += 1.0
            elif t_idx == 4:
                cm_coeffs[(0,0)] += -1.0
                cm_coeffs[(0,1)] += 1.0
            elif t_idx == 5:
                cm_coeffs[(1,1)] += 1.0
            elif t_idx == 6:
                cm_coeffs[(0,0)] += 1.0
            
            c_block_coeffs = {}
            if m_idx == 0:
                c_block_coeffs = {(0,0): 1.0, (1,1): 1.0}
            elif m_idx == 1:
                c_block_coeffs = {(1,0): 1.0, (1,1): -1.0}
            elif m_idx == 2:
                c_block_coeffs = {(0,1): 1.0, (1,1): 1.0}
            elif m_idx == 3:
                c_block_coeffs = {(0,0): 1.0, (1,0): 1.0}
            elif m_idx == 4:
                c_block_coeffs = {(0,0): -1.0, (0,1): 1.0}
            elif m_idx == 5:
                c_block_coeffs = {(1,1): 1.0}
            elif m_idx == 6:
                c_block_coeffs = {(0,0): 1.0}
            
            for (li, lj), cm_c in cm_coeffs.items():
                if cm_c == 0.0: continue
                for (br, bc), cb_c in c_block_coeffs.items():
                    if cb_c == 0.0: continue
                    i_final = 2 * br + li
                    k_final = 2 * bc + lj
                    w_idx = k_final * n + i_final
                    w_coeffs[w_idx] += cm_c * cb_c
            
            for i in range(p*n):
                W[i][r] = w_coeffs[i]

    # --- Part 2: 4x4x3 using 4 * (2x2x3) (Rank 44) ---
    # Try to find rank-8 decomposition for 2x2x3 to reduce from 44 to 40
    
    def _search_2x2x3_rank8():
        """Try to find a rank-8 decomposition of 2x2x3 matmul tensor.
        A: 2x2, B: 2x3, C: 2x3
        T[i,j,k] = 1 if C[i,k] depends on A[i,j]*B[j,k]
        """
        # Tensor entries: T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,1,2}
        # That's 12 nonzero entries.
        # We want sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k]
        # U: 2 x 8, V: 6 x 8, W: 6 x 8
        # Note: V and W have 6 rows (2*3=6)
        # V[j,k] indexed as j*3+k, W[i,k] indexed as i*3+k
        
        # Try structured approach: use Strassen for 2x2x2 (rank 7) + 1 extra
        # But maybe we can share the extra term.
        # Let's try a direct search with rational entries.
        
        import itertools
        
        # The 2x2x3 tensor: T[i,j,k] = 1 for all (i,j,k) in {0,1}x{0,1}x{0,1,2}
        # This is actually the product of the 2x2x2 tensor with a 1x1x3 "all ones" tensor?
        # No, it's just 12 ones.
        
        # Lower bound: max(4, 6, 6) = 6. But known to be at least 7.
        # Let's try rank 8 with a structured ansatz.
        
        # Ansatz: Use 7 Strassen terms for the k=0,1 part, and 1 shared term for k=2.
        # Strassen 2x2x2 terms (for columns 0,1 of B and 0,1 of C):
        # We need to handle the 3rd column separately.
        
        # Alternative: 4 terms for (A*I2)*B_col2 where I2 is identity-like
        # Actually, let me just try: 7 Strassen + 1 = 8, where the 1 term handles all of column 2.
        # But column 2 of C depends on both rows of A, so we need at least 2 terms for column 2.
        # So 7 + 2 = 9 is the natural decomposition.
        
        # Can we do 8? We'd need to share one term between the Strassen part and column 2.
        # This is non-trivial. Let's try a numerical search.
        
        # For now, return None to indicate no improvement found, and fall back to rank 9.
        # A proper search would take too long for this budget.
        return None
    
    # For safety, use the known rank-9 decomposition (7 Strassen + 2 schoolbook for col 2)
    # which gives 44 total for the 4x4x3 part.
    
    def add_2x2x3(a_row_block, a_col_block, b_row_block, c_row_block):
        nonlocal current_rank
        
        strassen_2x2x2_terms = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # M1
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}),            # M2
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}),           # M3
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}),           # M4
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}),            # M5
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}), # M6
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})  # M7
        ]
        
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub = strassen_2x2x2_terms[t_idx]
            
            u_coeffs = [0.0] * (n*m)
            v_coeffs = [0.0] * (m*p)
            
            for (li, lj), c_x in x_sub.items():
                i = 2 * a_row_block + li
                j = 2 * a_col_block + lj
                u_idx = i * m + j
                u_coeffs[u_idx] += c_x
            
            for (li, lj), c_y in y_sub.items():
                i = 2 * b_row_block + li
                j = 4 + lj
                v_idx = i * p + j
                v_coeffs[v_idx] += c_y
            
            for i in range(n*m):
                U[i][r] = u_coeffs[i]
            for j in range(m*p):
                V[j][r] = v_coeffs[j]
            
            w_coeffs = [0.0] * (p*n)
            
            cm_coeffs = {
                (0,0): 0.0, (0,1): 0.0, (1,0): 0.0, (1,1): 0.0
            }
            
            if t_idx == 0:
                cm_coeffs[(0,0)] += 1.0
                cm_coeffs[(1,1)] += 1.0
            elif t_idx == 1:
                cm_coeffs[(1,0)] += 1.0
                cm_coeffs[(1,1)] += -1.0
            elif t_idx == 2:
                cm_coeffs[(0,1)] += 1.0
                cm_coeffs[(1,1)] += 1.0
            elif t_idx == 3:
                cm_coeffs[(0,0)] += 1.0
                cm_coeffs[(1,0)] += 1.0
            elif t_idx == 4:
                cm_coeffs[(0,0)] += -1.0
                cm_coeffs[(0,1)] += 1.0
            elif t_idx == 5:
                cm_coeffs[(1,1)] += 1.0
            elif t_idx == 6:
                cm_coeffs[(0,0)] += 1.0
            
            for (li, lj), cm_c in cm_coeffs.items():
                if cm_c == 0.0: continue
                i_final = 2 * c_row_block + li
                k_final = 4 + lj
                w_idx = k_final * n + i_final
                w_coeffs[w_idx] += cm_c
            
            for i in range(p*n):
                W[i][r] = w_coeffs[i]
        
        schoolbook_terms = [
            ({(0,0): 1}, {(0,0): 1}, (0,0)), # A00 B02 -> C00
            ({(0,1): 1}, {(1,0): 1}, (0,0)), # A01 B12 -> C00
            ({(1,0): 1}, {(0,0): 1}, (1,0)), # A10 B02 -> C10
            ({(1,1): 1}, {(1,0): 1}, (1,0))  # A11 B12 -> C10
        ]
        
        for t_idx in range(4):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_pos = schoolbook_terms[t_idx]
            
            u_coeffs = [0.0] * (n*m)
            v_coeffs = [0.0] * (m*p)
            
            for (li, lj), c_x in x_sub.items():
                i = 2 * a_row_block + li
                j = 2 * a_col_block + lj
                u_idx = i * m + j
                u_coeffs[u_idx] += c_x
            
            for (li, lj), c_y in y_sub.items():
                i = 2 * b_row_block + li
                j = 6
                v_idx = i * p + j
                v_coeffs[v_idx] += c_y
            
            for i in range(n*m):
                U[i][r] = u_coeffs[i]
            for j in range(m*p):
                V[j][r] = v_coeffs[j]
            
            w_coeffs = [0.0] * (p*n)
            i_final = 2 * c_row_block + c_pos[0]
            k_final = 6
            w_idx = k_final * n + i_final
            w_coeffs[w_idx] = 1.0
            
            for i in range(p*n):
                W[i][r] = w_coeffs[i]

    add_2x2x3(0, 0, 0, 0)
    add_2x2x3(0, 1, 1, 0)
    add_2x2x3(1, 0, 0, 1)
    add_2x2x3(1, 1, 1, 1)

    U = [row[:current_rank] for row in U]
    V = [row[:current_rank] for row in V]
    W = [row[:current_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
