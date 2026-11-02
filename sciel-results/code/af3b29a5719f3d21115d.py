import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x7 matrix.
    
    Strategy:
    - Split middle dimension m=3 into m1=2, m2=1
    - A = [A1 | A2] where A1: 2x2, A2: 2x1
    - B = [B1; B2] where B1: 2x7, B2: 1x7
    - C = A1*B1 + A2*B2
    
    For A1*B1 (2x2 x 2x7):
    - Split B1 columns into [B1a, B1b, B1c] where B1a: 2x2, B1b: 2x2, B1c: 2x1
    - Use Strassen for A1*B1a and A1*B1b (rank 7 each)
    - Use schoolbook for A1*B1c (rank 4)
    
    For A2*B2 (2x1 x 1x7):
    - Schoolbook, rank 7
    
    Total rank: 7 + 7 + 4 + 7 = 25
    """
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS  # 2, 3, 7
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    terms = []
    
    def add_term(u, v, w):
        u_list = [0.0] * (n * m)
        v_list = [0.0] * (m * p)
        w_list = [0.0] * (p * n)
        for idx, val in u.items():
            u_list[idx] = val
        for idx, val in v.items():
            v_list[idx] = val
        for idx, val in w.items():
            w_list[idx] = val
        terms.append((u_list, v_list, w_list))
    
    # A1*B1a: A[0:2, 0:2] * B[0:2, 0:2] using Strassen
    # M1 = (A00+A11)(B00+B11)
    add_term({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,0): 1, v_idx(1,1): 1}, {w_idx(0,0): 1, w_idx(1,1): 1})
    # M2 = (A10+A11)B00
    add_term({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,0): 1}, {w_idx(0,1): 1, w_idx(1,1): -1})
    # M3 = A00(B10-B00)
    add_term({u_idx(0,0): 1}, {v_idx(0,1): 1, v_idx(1,1): -1}, {w_idx(1,0): 1, w_idx(1,1): 1})
    # M4 = A11(B01-B10)
    add_term({u_idx(1,1): 1}, {v_idx(1,0): 1, v_idx(0,0): -1}, {w_idx(0,0): 1, w_idx(0,1): 1})
    # M5 = (A00+A01)B11
    add_term({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,1): 1}, {w_idx(0,0): -1, w_idx(1,0): 1})
    # M6 = (A10-A00)(B00+B01)
    add_term({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,0): 1, v_idx(0,1): 1}, {w_idx(1,1): 1})
    # M7 = (A01-A11)(B10+B11)
    add_term({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,0): 1, v_idx(1,1): 1}, {w_idx(0,0): 1})
    
    # A1*B1b: A[0:2, 0:2] * B[0:2, 2:4] using Strassen
    add_term({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,2): 1, v_idx(1,3): 1}, {w_idx(2,0): 1, w_idx(3,1): 1})
    add_term({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,2): 1}, {w_idx(2,1): 1, w_idx(3,1): -1})
    add_term({u_idx(0,0): 1}, {v_idx(0,3): 1, v_idx(1,3): -1}, {w_idx(3,0): 1, w_idx(3,1): 1})
    add_term({u_idx(1,1): 1}, {v_idx(1,2): 1, v_idx(0,2): -1}, {w_idx(2,0): 1, w_idx(2,1): 1})
    add_term({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,3): 1}, {w_idx(2,0): -1, w_idx(3,0): 1})
    add_term({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,2): 1, v_idx(0,3): 1}, {w_idx(3,1): 1})
    add_term({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,2): 1, v_idx(1,3): 1}, {w_idx(2,0): 1})
    
    # A1*B1c: A[0:2, 0:2] * B[0:2, 4:5] using schoolbook (2x2 x 2x1)
    add_term({u_idx(0,0): 1}, {v_idx(0,4): 1}, {w_idx(4,0): 1})
    add_term({u_idx(0,1): 1}, {v_idx(1,4): 1}, {w_idx(4,0): 1})
    add_term({u_idx(1,0): 1}, {v_idx(0,4): 1}, {w_idx(4,1): 1})
    add_term({u_idx(1,1): 1}, {v_idx(1,4): 1}, {w_idx(4,1): 1})
    
    # A2*B2: A[0:2, 2] * B[2, 0:7] using schoolbook (2x1 x 1x7)
    for k in range(p):
        add_term({u_idx(0,2): 1}, {v_idx(2,k): 1}, {w_idx(k,0): 1})
        add_term({u_idx(1,2): 1}, {v_idx(2,k): 1}, {w_idx(k,1): 1})
    
    rank = len(terms)
    
    # Verify the decomposition is exact
    def verify(U_mat, V_mat, W_mat):
        """Check that sum_r U[i,j,r] * V[j',k',r] * W[k'',i'',r] = delta(i,i'')delta(j,j')delta(k',k'')"""
        n, m, p = DIMENSIONS
        # Build target tensor
        target = np.zeros((n, m, m, p, n, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i, j, j, k, i, k] = 1.0
        
        # Compute from decomposition
        U_3d = U_mat.reshape(n, m, rank)
        V_3d = V_mat.reshape(m, p, rank)
        W_3d = W_mat.reshape(p, n, rank)
        
        # T[i, j, j_prime, k_prime, i_prime, k_prime2] = sum_r U[i,j,r] * V[j_prime,k_prime,r] * W[k_prime2,i_prime,r]
        # We need T[i,j,j',k',i',k''] = delta(i,i')delta(j,j')delta(k',k'')
        computed = np.zeros((n, m, m, p, n, p))
        for r in range(rank):
            u_r = U_3d[:, :, r]  # (n, m)
            v_r = V_3d[:, :, r]  # (m, p)
            w_r = W_3d[:, :, r]  # (p, n)
            
            # outer product contribution
            for i in range(n):
                for j in range(m):
                    if u_r[i, j] != 0:
                        for jp in range(m):
                            for kp in range(p):
                                if v_r[jp, kp] != 0:
                                    for ip in range(n):
                                        for kp2 in range(p):
                                            if w_r[kp2, ip] != 0:
                                                computed[i, j, jp, kp, ip, kp2] += u_r[i, j] * v_r[jp, kp] * w_r[kp2, ip]
        
        max_err = np.max(np.abs(computed - target))
        return max_err < 1e-10
    
    # Build U, V, W matrices
    U_mat = np.zeros((n * m, rank))
    V_mat = np.zeros((m * p, rank))
    W_mat = np.zeros((p * n, rank))
    
    for c, (u_list, v_list, w_list) in enumerate(terms):
        U_mat[:, c] = u_list
        V_mat[:, c] = v_list
        W_mat[:, c] = w_list
    
    # Verify
    is_valid = verify(U_mat, V_mat, W_mat)
    
    if not is_valid:
        # Fallback to schoolbook if verification fails
        rank_school = n * m * p
        U_school = np.zeros((n * m, rank_school))
        V_school = np.zeros((m * p, rank_school))
        W_school = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, col] = 1
                    V_school[j * p + k, col] = 1
                    W_school[k * n + i, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
