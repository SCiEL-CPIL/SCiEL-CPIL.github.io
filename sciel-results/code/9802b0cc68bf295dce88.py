import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get index for A[i,j] in U
    def idx_A(i, j):
        return i * m + j
        
    # Helper to get index for B[j,k] in V
    def idx_B(j, k):
        return j * p + k
        
    # Helper to get index for C[i,k] in W (Note: W[k*n+i])
    def idx_C(i, k):
        return k * n + i

    # We represent linear forms as lists of length 16 (for 4x4 matrices)
    # Index 0..15 corresponds to row-major order
    
    # Define the 7 outer Strassen forms for X and Y
    # X_s and Y_s are 2x2 matrices of linear forms
    # Each linear form is a vector of 16 coefficients
    
    # Initialize A and B block access
    # A blocks:
    # A11: [0, 1, 4, 5]
    # A12: [2, 3, 6, 7]
    # A21: [8, 9, 12, 13]
    # A22: [10, 11, 14, 15]
    
    # B blocks:
    # B11: [0, 1, 4, 5]
    # B12: [2, 3, 6, 7]
    # B21: [8, 9, 12, 13]
    # B22: [10, 11, 14, 15]
    
    def get_block_indices(block):
        # block: 11, 12, 21, 22
        r1, c1 = divmod(block // 10 - 1, 2)
        r2, c2 = divmod(block % 10 - 1, 2)
        # Actually, simpler:
        # 11 -> top-left
        # 12 -> top-right
        # 21 -> bottom-left
        # 22 -> bottom-right
        if block == 11:
            return [0, 1, 4, 5]
        elif block == 12:
            return [2, 3, 6, 7]
        elif block == 21:
            return [8, 9, 12, 13]
        elif block == 22:
            return [10, 11, 14, 15]
            
    def add_form(f1, f2, sign=1):
        res = [0]*16
        for i in range(16):
            res[i] = f1[i] + sign * f2[i]
        return res

    def sub_form(f1, f2):
        return add_form(f1, f2, -1)

    def scale_form(f, s):
        return [x * s for x in f]

    # Pre-compute block forms for A and B
    # A_block[11] is a 2x2 matrix of forms? No, just indices.
    # We need to form X_s and Y_s which are 2x2 matrices of linear forms.
    
    # Let's create a function to get the 4 linear forms (entries) of a block
    def get_block_forms(block, mat_type):
        # Returns 4 forms: 00, 01, 10, 11
        indices = get_block_indices(block)
        forms = []
        for idx in indices:
            form = [0]*16
            form[idx] = 1
            forms.append(form)
        return forms # [00, 01, 10, 11]

    A11 = get_block_forms(11, 'A')
    A12 = get_block_forms(12, 'A')
    A21 = get_block_forms(21, 'A')
    A22 = get_block_forms(22, 'A')
    
    B11 = get_block_forms(11, 'B')
    B12 = get_block_forms(12, 'B')
    B21 = get_block_forms(21, 'B')
    B22 = get_block_forms(22, 'B')
    
    # Construct X_s and Y_s
    # X1 = A11 + A22
    X1 = [add_form(A11[i], A22[i]) for i in range(4)]
    # X2 = A21 + A22
    X2 = [add_form(A21[i], A22[i]) for i in range(4)]
    # X3 = A11
    X3 = A11
    # X4 = A22
    X4 = A22
    # X5 = A11 + A12
    X5 = [add_form(A11[i], A12[i]) for i in range(4)]
    # X6 = A21 - A11
    X6 = [sub_form(A21[i], A11[i]) for i in range(4)]
    # X7 = A12 - A22
    X7 = [sub_form(A12[i], A22[i]) for i in range(4)]
    
    Y1 = [add_form(B11[i], B22[i]) for i in range(4)]
    Y2 = [add_form(B21[i], B22[i]) for i in range(4)]
    Y3 = B11
    Y4 = B22
    Y5 = [add_form(B11[i], B12[i]) for i in range(4)]
    Y6 = [sub_form(B21[i], B11[i]) for i in range(4)]
    Y7 = [sub_form(B12[i], B22[i]) for i in range(4)]
    
    Xs = [X1, X2, X3, X4, X5, X6, X7]
    Ys = [Y1, Y2, Y3, Y4, Y5, Y6, Y7]
    
    # Strassen reconstruction coefficients for M_s entries
    # M00 = m1 + m4 - m5 + m7
    # M01 = m3 + m5
    # M10 = m2 + m4
    # M11 = m1 - m2 + m3 + m6
    
    # Contribution of M_s entries to C entries
    # C11 (0,0) = M1_00 + M4_00 - M5_00 + M7_00
    # C12 (0,1) = M3_01 + M5_01
    # C21 (1,0) = M2_10 + M4_10
    # C22 (1,1) = M1_11 - M2_11 + M3_11 + M6_11
    
    # Map C index (i,k) to W index
    # W index = k*4 + i
    
    # We need to accumulate W contributions
    # For each s (0..6) and each inner product k (0..6):
    # 1. Compute L_A and L_B
    # 2. Determine which M_s entry it is (m_k)
    # 3. Determine which C entries that M_s entry contributes to, and with what sign
    
    # Let's pre-compute the contribution of each m_k (for a fixed s) to C entries
    # m1 -> M00 (+1), M11 (+1)
    # m2 -> M10 (+1), M11 (-1)
    # m3 -> M01 (+1), M11 (+1)
    # m4 -> M00 (+1), M10 (+1)
    # m5 -> M00 (-1), M01 (+1)
    # m6 -> M11 (+1)
    # m7 -> M00 (+1)
    
    # And M_s entries to C:
    # M1_00 -> C11 (+1)
    # M1_11 -> C22 (+1)
    # M2_10 -> C21 (+1)
    # M2_11 -> C22 (-1)
    # M3_01 -> C12 (+1)
    # M3_11 -> C22 (+1)
    # M4_00 -> C11 (+1)
    # M4_10 -> C21 (+1)
    # M5_00 -> C11 (-1)
    # M5_01 -> C12 (+1)
    # M6_11 -> C22 (+1)
    # M7_00 -> C11 (+1)
    
    # Combine:
    # m1: M1_00(+1)->C11(+1), M1_11(+1)->C22(+1)
    # m2: M2_10(+1)->C21(+1), M2_11(-1)->C22(-1)
    # m3: M3_01(+1)->C12(+1), M3_11(+1)->C22(+1)
    # m4: M4_00(+1)->C11(+1), M4_10(+1)->C21(+1)
    # m5: M5_00(-1)->C11(-1), M5_01(+1)->C12(+1)
    # m6: M6_11(+1)->C22(+1)
    # m7: M7_00(+1)->C11(+1)
    
    # C indices:
    # C11: i=0, k=0 -> W_idx = 0
    # C12: i=0, k=1 -> W_idx = 1*4 + 0 = 4
    # C21: i=1, k=0 -> W_idx = 0*4 + 1 = 1
    # C22: i=1, k=1 -> W_idx = 1*4 + 1 = 5
    
    contributions = {
        0: [(0, 1), (5, 1)],   # m1
        1: [(1, 1), (5, -1)],  # m2
        2: [(4, 1), (5, 1)],   # m3
        3: [(0, 1), (1, 1)],   # m4
        4: [(0, -1), (4, 1)],  # m5
        5: [(5, 1)],           # m6
        6: [(0, 1)]            # m7
    }
    
    col = 0
    for s in range(7):
        X = Xs[s]
        Y = Ys[s]
        # X[0]=00, X[1]=01, X[2]=10, X[3]=11
        # Y[0]=00, Y[1]=01, Y[2]=10, Y[3]=11
        
        # Inner Strassen products
        # m1 = (X00+X11)(Y00+Y11)
        # m2 = (X10+X11)Y00
        # m3 = X00(Y01-Y11)
        # m4 = X11(Y10-Y00)
        # m5 = (X00+X01)Y11
        # m6 = (X10-X00)(Y00+Y01)
        # m7 = (X01-X11)(Y10+Y11)
        
        products = []
        
        # m1
        LA = add_form(X[0], X[3])
        LB = add_form(Y[0], Y[3])
        products.append((LA, LB))
        
        # m2
        LA = add_form(X[2], X[3])
        LB = Y[0]
        products.append((LA, LB))
        
        # m3
        LA = X[0]
        LB = sub_form(Y[1], Y[3])
        products.append((LA, LB))
        
        # m4
        LA = X[3]
        LB = sub_form(Y[2], Y[0])
        products.append((LA, LB))
        
        # m5
        LA = add_form(X[0], X[1])
        LB = Y[3]
        products.append((LA, LB))
        
        # m6
        LA = sub_form(X[2], X[0])
        LB = add_form(Y[0], Y[1])
        products.append((LA, LB))
        
        # m7
        LA = sub_form(X[1], X[3])
        LB = add_form(Y[2], Y[3])
        products.append((LA, LB))
        
        for k in range(7):
            LA, LB = products[k]
            # Fill U and V
            for i in range(16):
                if LA[i] != 0:
                    U[i][col] = LA[i]
                if LB[i] != 0:
                    V[i][col] = LB[i]
            
            # Fill W
            for w_idx, sign in contributions[k]:
                W[w_idx][col] += sign
                
            col += 1
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    # Quick verification
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    # Check T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1
    # T is 4x4x4
    # U is 16x49, V is 16x49, W is 16x49
    # We need to map indices correctly
    
    n=4; m=4; p=4
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                # Sum over r
                # U[u_idx, :] * V[v_idx, :] * W[w_idx, :]
                # This is element-wise product sum
                term = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                T[i, j, k] = term
                
    # T should be all 1s
    if not np.allclose(T, 1.0):
        print("Verification Failed")
        print(T)
    else:
        print("Verification Passed")
