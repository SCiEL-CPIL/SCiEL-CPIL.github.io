import time
import numpy as np
from fractions import Fraction
from itertools import combinations

def construct():
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12

    start_time = time.time()
    deadline = start_time + 110

    def build_rank44():
        rank = 44
        U = [[Fraction(0)] * rank for _ in range(dim_u)]
        V = [[Fraction(0)] * rank for _ in range(dim_v)]
        W = [[Fraction(0)] * rank for _ in range(dim_w)]

        col = 0
        def add_strassen(X_u, Y_v, Z_w, col):
            U[X_u[0]][col] += 1; U[X_u[3]][col] += 1
            V[Y_v[0]][col] += 1; V[Y_v[3]][col] += 1
            W[Z_w[0]][col] += 1; W[Z_w[3]][col] += 1
            col += 1
            U[X_u[2]][col] += 1; U[X_u[3]][col] += 1
            V[Y_v[0]][col] += 1
            W[Z_w[2]][col] += 1; W[Z_w[3]][col] -= 1
            col += 1
            U[X_u[0]][col] += 1
            V[Y_v[1]][col] += 1; V[Y_v[3]][col] -= 1
            W[Z_w[1]][col] += 1; W[Z_w[3]][col] += 1
            col += 1
            U[X_u[3]][col] += 1
            V[Y_v[2]][col] += 1; V[Y_v[0]][col] -= 1
            W[Z_w[0]][col] += 1; W[Z_w[2]][col] += 1
            col += 1
            U[X_u[0]][col] += 1; U[X_u[1]][col] += 1
            V[Y_v[3]][col] += 1
            W[Z_w[0]][col] -= 1; W[Z_w[1]][col] += 1
            col += 1
            U[X_u[2]][col] += 1; U[X_u[0]][col] -= 1
            V[Y_v[0]][col] += 1; V[Y_v[1]][col] += 1
            W[Z_w[3]][col] += 1
            col += 1
            U[X_u[1]][col] += 1; U[X_u[3]][col] -= 1
            V[Y_v[2]][col] += 1; V[Y_v[3]][col] += 1
            W[Z_w[0]][col] += 1
            col += 1
            return col

        # Four 2x2x2 Strassen blocks for the top 2 rows of A (3x4) and columns of B (4x4)
        # A is 3x4, B is 4x4, C is 4x3
        # Split A into rows 0-1 (2x4) and row 2 (1x4)
        # Split B into columns 0-1 (4x2) and columns 2-3 (4x2)
        # C is 4x3, split into columns 0-1 (4x2) and column 2 (4x1)
        
        # Block (0,0): A[0:2, 0:4] * B[0:4, 0:2] -> C[0:4, 0:2]
        col = add_strassen([0, 1, 4, 5], [0, 1, 4, 5], [0, 3, 1, 4], col)
        # Block (0,1): A[0:2, 0:4] * B[0:4, 2:4] -> C[0:4, 2:3] (only col 2 of C)
        col = add_strassen([0, 1, 4, 5], [2, 3, 6, 7], [2, 5, 3, 7], col)
        # Block (1,0): A[2:3, 0:4] * B[0:4, 0:2] -> C[0:4, 0:2]
        col = add_strassen([6, 7, 10, 11], [0, 1, 4, 5], [0, 3, 1, 4], col)
        # Block (1,1): A[2:3, 0:4] * B[0:4, 2:4] -> C[0:4, 2:3]
        col = add_strassen([6, 7, 10, 11], [2, 3, 6, 7], [2, 5, 3, 7], col)

        # Wait, the original program had a different decomposition. Let me re-examine.
        # The original program had:
        # col = add_strassen([0,1,4,5], [0,1,4,5], [0,3,1,4], col)  # A[0:2,:] * B[:,0:2] -> C[:,0:2]
        # col = add_strassen([2,3,6,7], [8,9,12,13], [0,3,1,4], col)  # A[0:2,2:4] * B[2:4,0:2] -> C[:,0:2]
        # col = add_strassen([0,1,4,5], [2,3,6,7], [6,9,7,10], col)  # A[0:2,0:2] * B[:,2:4] -> C[:,2:4]
        # col = add_strassen([2,3,6,7], [10,11,14,15], [6,9,7,10], col)  # A[0:2,2:4] * B[2:4,2:4] -> C[:,2:4]
        # Then 8 schoolbook terms for row 2 of A.
        # That gives 4*7 + 8 = 36 terms. But the rank is 44, so 36 + 8 = 44. OK.
        
        # Let me use the original decomposition exactly.
        U = [[Fraction(0)] * rank for _ in range(dim_u)]
        V = [[Fraction(0)] * rank for _ in range(dim_v)]
        W = [[Fraction(0)] * rank for _ in range(dim_w)]
        col = 0
        
        col = add_strassen([0,1,4,5], [0,1,4,5], [0,3,1,4], col)
        col = add_strassen([2,3,6,7], [8,9,12,13], [0,3,1,4], col)
        col = add_strassen([0,1,4,5], [2,3,6,7], [6,9,7,10], col)
        col = add_strassen([2,3,6,7], [10,11,14,15], [6,9,7,10], col)
        
        for j in range(4):
            U[8+j][col] += 1; V[j*4+0][col] += 1; W[2][col] += 1
            col += 1
            U[8+j][col] += 1; V[j*4+1][col] += 1; W[5][col] += 1
            col += 1
        for j in range(4):
            U[8+j][col] += 1; V[j*4+2][col] += 1; W[8][col] += 1
            col += 1
            U[8+j][col] += 1; V[j*4+3][col] += 1; W[11][col] += 1
            col += 1

        return U, V, W, col

    def verify_frac(U, V, W, n, m, p):
        U_arr = np.array([[float(x) for x in row] for row in U], dtype=np.float64)
        V_arr = np.array([[float(x) for x in row] for row in V], dtype=np.float64)
        W_arr = np.array([[float(x) for x in row] for row in W], dtype=np.float64)
        
        T_expected = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_expected[i*m+j, j*p+k, k*n+i] = 1.0

        T_actual = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
        return np.allclose(T_actual, T_expected, atol=1e-10, rtol=1e-10)

    def verify_fast(U, V, W, T_expected):
        """Faster verification using precomputed expected tensor."""
        if len(U) == 0 or len(V) == 0 or len(W) == 0:
            return False
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        T_actual = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
        return np.allclose(T_actual, T_expected, atol=1e-10, rtol=1e-10)

    U, V, W, actual_rank = build_rank44()
    
    # Precompute expected tensor for fast verification
    T_expected = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1.0

    if not verify_frac(U, V, W, n, m, p):
        rank = 48
        U = [[Fraction(0)]*rank for _ in range(dim_u)]
        V = [[Fraction(0)]*rank for _ in range(dim_v)]
        W = [[Fraction(0)]*rank for _ in range(dim_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = Fraction(1)
                    V[j*p+k][col] = Fraction(1)
                    W[k*n+i][col] = Fraction(1)
                    col += 1
        return {"U": [[float(x) for x in row] for row in U], 
                "V": [[float(x) for x in row] for row in V], 
                "W": [[float(x) for x in row] for row in W]}

    current_U = U
    current_V = V
    current_W = W
    current_rank = actual_rank

    # Try to remove factors - single removal first
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        if current_rank <= 1:
            break

        found = False
        for r_idx in range(current_rank):
            if time.time() > deadline:
                break
            
            keep_indices = [i for i in range(current_rank) if i != r_idx]
            
            U_test = [[row[i] for i in keep_indices] for row in current_U]
            V_test = [[row[i] for i in keep_indices] for row in current_V]
            W_test = [[row[i] for i in keep_indices] for row in current_W]
            
            if verify_fast(U_test, V_test, W_test, T_expected):
                current_U = U_test
                current_V = V_test
                current_W = W_test
                current_rank -= 1
                found = True
                break
        
        if not found:
            break

    # Try to remove pairs of factors
    for attempt in range(30):
        if time.time() > deadline:
            break
        
        if current_rank <= 2:
            break

        found = False
        for r1, r2 in combinations(range(current_rank), 2):
            if time.time() > deadline:
                break
            
            keep_indices = [i for i in range(current_rank) if i != r1 and i != r2]
            
            U_test = [[row[i] for i in keep_indices] for row in current_U]
            V_test = [[row[i] for i in keep_indices] for row in current_V]
            W_test = [[row[i] for i in keep_indices] for row in current_W]
            
            if verify_fast(U_test, V_test, W_test, T_expected):
                current_U = U_test
                current_V = V_test
                current_W = W_test
                current_rank -= 2
                found = True
                break
        
        if not found:
            break

    # Convert to float for output
    U_out = [[float(x) for x in row] for row in current_U]
    V_out = [[float(x) for x in row] for row in current_V]
    W_out = [[float(x) for x in row] for row in current_W]

    # Final verification
    if not verify_fast(U_out, V_out, W_out, T_expected):
        # Fallback to schoolbook
        rank = 48
        U = [[0.0]*rank for _ in range(dim_u)]
        V = [[0.0]*rank for _ in range(dim_v)]
        W = [[0.0]*rank for _ in range(dim_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U_out, "V": V_out, "W": W_out}
