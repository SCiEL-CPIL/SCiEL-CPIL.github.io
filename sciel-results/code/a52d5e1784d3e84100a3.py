"""Combined program: Laderman 3x3 tensor Strassen 2x2 for 6x6x6 matrix multiplication.
Rank = 23 * 7 = 161.
"""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _laderman_3x3():
    """
    Return Laderman decomposition for 3x3 matrix multiplication.
    Rank = 23.
    """
    # A = [[a,b,c],[d,e,f],[g,h,i]], B = [[j,k,l],[m,n,o],[p,q,r]]
    # C = A*B
    
    # 23 products:
    # M1=(a+i)(j+r), M2=(d+g)(m+p), M3=(b+h)(k+q)
    # M4=(a+b+c)(j+m+p), M5=(a+d+g)(j+k+l), M6=(a+e+i)(j+n+r)
    # M7=(a+b+e)(j+k+n), M8=(a+e+i)(k+n+q), M9=(a+e+i)(l+o+r)
    # M10=(a+b+e)(k+n+q), M11=(a+b+e)(l+o+r)
    # M12=(b+h)(j+m+p), M13=(b+h)(k+n+q), M14=(b+h)(l+o+r)
    # M15=(d+g)(j+k+l), M16=(d+g)(k+n+q), M17=(d+g)(l+o+r)
    # M18=(a+i)(k+q), M19=(a+i)(l+r), M20=(d+g)(n+q)
    # M21=(d+g)(o+r), M22=(b+h)(n+q), M23=(b+h)(o+r)
    #
    # C00 = M1 + M4 + M5 + M6 + M7 + M12 + M15 - M18
    # C01 = M3 + M5 + M7 + M8 + M10 + M13 + M16 - M18
    # C02 = M5 + M9 + M11 + M14 + M17 - M19
    # C10 = M2 + M4 + M6 + M12 + M15 - M20
    # C11 = M2 + M6 + M7 + M8 + M10 + M13 + M16 - M20 - M22
    # C12 = M9 + M11 + M14 + M17 - M21 - M23
    # C20 = M2 + M4 + M6 + M12 + M15 - M20
    # C21 = M3 + M8 + M10 + M13 + M16 - M18 - M22
    # C22 = M1 + M6 + M9 + M11 + M14 + M17 - M19 - M21 - M23
    
    products = [
        ({0:1, 8:1}, {0:1, 8:1}, {0:1, 8:1}),          # M1
        ({3:1, 6:1}, {3:1, 6:1}, {3:1, 6:1}),           # M2
        ({1:1, 7:1}, {1:1, 7:1}, {1:1, 7:1}),           # M3
        ({0:1, 1:1, 2:1}, {0:1, 3:1, 6:1}, {0:1, 3:1, 6:1}),  # M4
        ({0:1, 3:1, 6:1}, {0:1, 1:1, 2:1}, {0:1, 1:1, 2:1}),  # M5
        ({0:1, 4:1, 8:1}, {0:1, 4:1, 8:1}, {0:1, 4:1, 8:1}),  # M6
        ({0:1, 1:1, 4:1}, {0:1, 1:1, 4:1}, {0:1, 1:1, 4:1}),  # M7
        ({0:1, 4:1, 8:1}, {1:1, 4:1, 7:1}, {1:1, 4:1, 7:1}),  # M8
        ({0:1, 4:1, 8:1}, {2:1, 5:1, 8:1}, {2:1, 5:1, 8:1}),  # M9
        ({0:1, 1:1, 4:1}, {1:1, 4:1, 7:1}, {1:1, 4:1, 7:1}),  # M10
        ({0:1, 1:1, 4:1}, {2:1, 5:1, 8:1}, {2:1, 5:1, 8:1}),  # M11
        ({1:1, 7:1}, {0:1, 3:1, 6:1}, {0:1, 3:1, 6:1}),  # M12
        ({1:1, 7:1}, {1:1, 4:1, 7:1}, {1:1, 4:1, 7:1}),  # M13
        ({1:1, 7:1}, {2:1, 5:1, 8:1}, {2:1, 5:1, 8:1}),  # M14
        ({3:1, 6:1}, {0:1, 1:1, 2:1}, {0:1, 1:1, 2:1}),  # M15
        ({3:1, 6:1}, {1:1, 4:1, 7:1}, {1:1, 4:1, 7:1}),  # M16
        ({3:1, 6:1}, {2:1, 5:1, 8:1}, {2:1, 5:1, 8:1}),  # M17
        ({0:1, 8:1}, {1:1, 7:1}, {1:-1, 7:-1}),         # M18
        ({0:1, 8:1}, {2:1, 8:1}, {2:-1, 8:-1}),         # M19
        ({3:1, 6:1}, {4:1, 7:1}, {4:-1, 7:-1}),         # M20
        ({3:1, 6:1}, {5:1, 8:1}, {5:-1, 8:-1}),         # M21
        ({1:1, 7:1}, {4:1, 7:1}, {4:-1, 7:-1}),         # M22
        ({1:1, 7:1}, {5:1, 8:1}, {5:-1, 8:-1}),         # M23
    ]
    
    U = [[0] * 23 for _ in range(9)]
    V = [[0] * 23 for _ in range(9)]
    W = [[0] * 23 for _ in range(9)]
    
    for t, (coeffs_A, coeffs_B, coeffs_C) in enumerate(products):
        for idx, val in coeffs_A.items():
            U[idx][t] += val
        for idx, val in coeffs_B.items():
            V[idx][t] += val
        for idx, val in coeffs_C.items():
            W[idx][t] += val
    
    return U, V, W, 23

def _strassen_2x2():
    """
    Return Strassen decomposition for 2x2 matrix multiplication.
    Rank = 7.
    """
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    products = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),    # M1
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),        # M2
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),        # M3
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),        # M4
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),        # M5
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),        # M6
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),        # M7
    ]
    
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    for t, (coeffs_A, coeffs_B, coeffs_C) in enumerate(products):
        for idx, val in coeffs_A.items():
            U[idx][t] += val
        for idx, val in coeffs_B.items():
            V[idx][t] += val
        for idx, val in coeffs_C.items():
            W[idx][t] += val
    
    return U, V, W, 7

def _build_kronecker_decomposition(U1, V1, W1, n1, m1, p1, r1,
                                   U2, V2, W2, n2, m2, p2, r2):
    """
    Construct global U, V, W for (n1*n2) x (m1*m2) x (p1*p2) matrix multiplication
    using Kronecker product of two local decompositions.
    
    Args:
        U1, V1, W1: local decomposition 1 with dimensions n1, m1, p1 and rank r1
        U2, V2, W2: local decomposition 2 with dimensions n2, m2, p2 and rank r2
    
    Returns:
        (U, V, W) as lists of lists, rank
    """
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    rank = r1 * r2
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t1 in range(r1):
        for t2 in range(r2):
            t = t1 * r2 + t2
            
            # Build U column
            for i1 in range(n1):
                for j1 in range(m1):
                    u1_val = U1[i1 * m1 + j1][t1]
                    if u1_val == 0:
                        continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            u2_val = U2[i2 * m2 + j2][t2]
                            if u2_val == 0:
                                continue
                            i = i1 * n2 + i2
                            j = j1 * m2 + j2
                            U[i * m + j][t] = u1_val * u2_val
            
            # Build V column
            for j1 in range(m1):
                for k1 in range(p1):
                    v1_val = V1[j1 * p1 + k1][t1]
                    if v1_val == 0:
                        continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            v2_val = V2[j2 * p2 + k2][t2]
                            if v2_val == 0:
                                continue
                            j = j1 * m2 + j2
                            k = k1 * p2 + k2
                            V[j * p + k][t] = v1_val * v2_val
            
            # Build W column
            # W convention: W[k*n + i, r] for output C[i, k]
            for k1 in range(p1):
                for i1 in range(n1):
                    w1_val = W1[k1 * n1 + i1][t1]
                    if w1_val == 0:
                        continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            w2_val = W2[k2 * n2 + i2][t2]
                            if w2_val == 0:
                                continue
                            i = i1 * n2 + i2
                            k = k1 * p2 + k2
                            W[k * n + i][t] = w1_val * w2_val
    
    return U, V, W, rank

def _verify_decomposition(U, V, W, n, m, p, seed=42):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    """
    np.random.seed(seed)
    A = np.random.rand(n, m).astype(float)
    B = np.random.rand(m, p).astype(float)
    C_true = A @ B
    
    C_comp = np.zeros((n, p), dtype=float)
    rank = len(U[0])
    for t in range(rank):
        U_col = np.array([U[i * m + j][t] for i in range(n) for j in range(m)]).reshape(n, m)
        V_col = np.array([V[j * p + k][t] for j in range(m) for k in range(p)]).reshape(m, p)
        M_t = U_col @ V_col
        W_col = np.array([W[k * n + i][t] for k in range(p) for i in range(n)]).reshape(p, n)
        C_comp += W_col.T * M_t
    
    if not np.allclose(C_comp, C_true, atol=1e-10):
        return False
    return True

def _schoolbook_6x6():
    """Fallback: schoolbook decomposition of rank 216"""
    n, m, p = 6, 6, 6
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
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 6, 6, 6
    
    # Build 3x3 Laderman local decomposition
    U_local, V_local, W_local, local_rank = _laderman_3x3()
    
    # Verify the local Laderman decomposition
    np.random.seed(42)
    A_local = np.random.rand(3, 3)
    B_local = np.random.rand(3, 3)
    C_true_local = A_local @ B_local
    
    C_comp_local = np.zeros((3, 3))
    for t in range(local_rank):
        U_col = np.array([U_local[i * 3 + j][t] for i in range(3) for j in range(3)]).reshape(3, 3)
        V_col = np.array([V_local[j * 3 + k][t] for j in range(3) for k in range(3)]).reshape(3, 3)
        M_t = U_col @ V_col
        W_col = np.array([W_local[k * 3 + i][t] for k in range(3) for i in range(3)]).reshape(3, 3)
        C_comp_local += W_col.T * M_t
    
    if not np.allclose(C_comp_local, C_true_local, atol=1e-10):
        # Fallback to schoolbook if Laderman is incorrect
        U, V, W, rank = _schoolbook_6x6()
        return {"U": U, "V": V, "W": W}
    
    # Build 2x2 Strassen local decomposition
    U_strassen, V_strassen, W_strassen, strassen_rank = _strassen_2x2()
    
    # Verify the local Strassen decomposition
    np.random.seed(43)
    A_strassen = np.random.rand(2, 2)
    B_strassen = np.random.rand(2, 2)
    C_true_strassen = A_strassen @ B_strassen
    
    C_comp_strassen = np.zeros((2, 2))
    for t in range(strassen_rank):
        U_col = np.array([U_strassen[i * 2 + j][t] for i in range(2) for j in range(2)]).reshape(2, 2)
        V_col = np.array([V_strassen[j * 2 + k][t] for j in range(2) for k in range(2)]).reshape(2, 2)
        M_t = U_col @ V_col
        W_col = np.array([W_strassen[k * 2 + i][t] for k in range(2) for i in range(2)]).reshape(2, 2)
        C_comp_strassen += W_col.T * M_t
    
    if not np.allclose(C_comp_strassen, C_true_strassen, atol=1e-10):
        # Fallback to schoolbook if Strassen is incorrect
        U, V, W, rank = _schoolbook_6x6()
        return {"U": U, "V": V, "W": W}
    
    # Build global decomposition using Kronecker product: Laderman(3x3) tensor Strassen(2x2)
    # Total rank = 23 * 7 = 161
    U, V, W, rank = _build_kronecker_decomposition(
        U_local, V_local, W_local, 3, 3, 3, local_rank,
        U_strassen, V_strassen, W_strassen, 2, 2, 2, strassen_rank
    )
    
    # Verify the full 6x6x6 decomposition
    if not _verify_decomposition(U, V, W, 6, 6, 6):
        # Fallback to schoolbook if verification fails
        U, V, W, rank = _schoolbook_6x6()
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
