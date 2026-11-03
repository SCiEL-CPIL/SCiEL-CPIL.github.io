import time
import numpy as np
from fractions import Fraction

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    n, m, p = 2, 4, 7
    
    # Build rank-50 decomposition using Strassen for 2x2x2 blocks and schoolbook for 2x2x1
    rank = 50
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(A_indices, B_indices, C_indices):
        nonlocal col
        def process_mk(A_coeffs, B_coeffs, C_coeffs):
            nonlocal col
            for local_a, coeff in A_coeffs.items():
                global_a = A_indices[local_a]
                U[global_a[0] * m + global_a[1]][col] += coeff
            for local_b, coeff in B_coeffs.items():
                global_b = B_indices[local_b]
                V[global_b[0] * p + global_b[1]][col] += coeff
            for local_c, coeff in C_coeffs.items():
                global_c = C_indices[local_c]
                W[global_c[1] * n + global_c[0]][col] += coeff
            col += 1

        # Strassen 2x2x2: 7 multiplications
        process_mk({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1})
        process_mk({2:1, 3:1}, {0:1}, {2:1, 3:-1})
        process_mk({0:1}, {1:1, 3:-1}, {1:1, 3:1})
        process_mk({3:1}, {2:1, 0:-1}, {0:1, 2:1})
        process_mk({0:1, 1:1}, {3:1}, {0:-1, 1:1})
        process_mk({2:1, 0:-1}, {0:1, 1:1}, {3:1})
        process_mk({1:1, 3:-1}, {2:1, 3:1}, {0:1})

    def add_schoolbook_221(A_indices, B_indices, C_indices):
        nonlocal col
        def process_mult(A_idx, B_idx, C_idx, coeff=1):
            nonlocal col
            global_a = A_indices[A_idx]
            global_b = B_indices[B_idx]
            global_c = C_indices[C_idx]
            
            U[global_a[0] * m + global_a[1]][col] += coeff
            V[global_b[0] * p + global_b[1]][col] += coeff
            W[global_c[1] * n + global_c[0]][col] += coeff
            col += 1

        process_mult(0, 0, 0, 1)
        process_mult(1, 1, 0, 1)
        process_mult(2, 0, 1, 1)
        process_mult(3, 1, 1, 1)

    A_L = [(0,0), (0,1), (1,0), (1,1)]
    A_R = [(0,2), (0,3), (1,2), (1,3)]
    
    B_1_L = [(0,0), (0,1), (1,0), (1,1)]
    B_1_R = [(2,0), (2,1), (3,0), (3,1)]
    C_1 = [(0,0), (0,1), (1,0), (1,1)]
    
    add_strassen_block(A_L, B_1_L, C_1)
    add_strassen_block(A_R, B_1_R, C_1)
    
    B_2_L = [(0,2), (0,3), (1,2), (1,3)]
    B_2_R = [(2,2), (2,3), (3,2), (3,3)]
    C_2 = [(0,2), (0,3), (1,2), (1,3)]
    
    add_strassen_block(A_L, B_2_L, C_2)
    add_strassen_block(A_R, B_2_R, C_2)
    
    B_3_L_L = [(0,4), (0,5), (1,4), (1,5)]
    B_3_L_R = [(2,4), (2,5), (3,4), (3,5)]
    C_3_L = [(0,4), (0,5), (1,4), (1,5)]
    
    add_strassen_block(A_L, B_3_L_L, C_3_L)
    add_strassen_block(A_R, B_3_L_R, C_3_L)
    
    B_3_R_L = [(0,6), (1,6)]
    B_3_R_R = [(2,6), (3,6)]
    C_3_R = [(0,6), (1,6)]
    
    add_schoolbook_221(A_L, B_3_R_L, C_3_R)
    add_schoolbook_221(A_R, B_3_R_R, C_3_R)
    
    assert col == 50, f"Expected 50 terms, got {col}"
    
    # === RANK REDUCTION: Look for scalar-multiple (u,v) pairs to merge ===
    U_arr = np.array([[float(x) for x in row] for row in U], dtype=float)
    V_arr = np.array([[float(x) for x in row] for row in V], dtype=float)
    AV = np.vstack([U_arr, V_arr])  # shape (36, 50)
    
    eliminated = set()
    W_arr = np.array([[float(x) for x in row] for row in W], dtype=float)
    
    for i in range(50):
        if time.time() > deadline - 5.0:
            break
        if i in eliminated:
            continue
        for j in range(i + 1, 50):
            if j in eliminated:
                continue
            ui = AV[:, i]
            uj = AV[:, j]
            nonzero_j = np.where(np.abs(uj) > 1e-12)[0]
            if len(nonzero_j) == 0:
                continue
            idx = nonzero_j[0]
            c = ui[idx] / uj[idx]
            if np.allclose(ui, c * uj, atol=1e-12):
                # Merge: u_i = c*u_j, v_i = c*v_j => u_i⊗v_i = c²*(u_j⊗v_j)
                W_arr[:, j] += c**2 * W_arr[:, i]
                eliminated.add(i)
                break
        if i in eliminated:
            continue
    
    if eliminated:
        keep = [k for k in range(50) if k not in eliminated]
        U_final = [[Fraction(W_arr[i, k]) if False else Fraction(str(U[i][k])) for k in keep] for i in range(n*m)]
        # Reconstruct with fractions properly
        U_final = []
        for i in range(n*m):
            U_final.append([U[i][k] for k in keep])
        V_final = []
        for i in range(m*p):
            V_final.append([V[i][k] for k in keep])
        W_final = []
        for i in range(p*n):
            W_final.append([Fraction(str(W_arr[i, k])) for k in keep])
        
        return {"U": U_final, "V": V_final, "W": W_final}
    
    return {"U": U, "V": V, "W": W}
