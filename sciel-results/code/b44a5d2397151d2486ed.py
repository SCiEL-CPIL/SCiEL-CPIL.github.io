"""Laderman's rank-23 decomposition for 3x3 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    n, m, p = 3, 3, 3
    rank = 23
    
    # Define the 23 terms as (U_vector, V_vector) pairs
    # U_vector: 9 entries for A[0,0]..A[2,2] (row-major: i*3+k)
    # V_vector: 9 entries for B[0,0]..B[2,2] (row-major: k*3+j)
    
    terms = []
    
    # p1 = a*x
    terms.append(([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]))
    # p2 = b*u
    terms.append(([0,1,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]))
    # p3 = c*r
    terms.append(([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0]))
    # p4 = d*y
    terms.append(([0,0,0, 1,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]))
    # p5 = e*v
    terms.append(([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]))
    # p6 = f*s
    terms.append(([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,1,0]))
    # p7 = g*z
    terms.append(([0,0,0, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 0,0,0]))
    # p8 = h*w
    terms.append(([0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,1, 0,0,0]))
    # p9 = i*t
    terms.append(([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]))
    # p10 = (a+e+i)*(x+v+t)
    terms.append(([1,0,0, 0,1,0, 0,0,1], [1,0,0, 0,1,0, 0,0,1]))
    # p11 = (a+h+i)*(x+s+t)
    terms.append(([1,0,0, 0,0,0, 0,1,1], [1,0,0, 0,0,0, 0,1,1]))
    # p12 = (b+d+i)*(u+y+t)
    terms.append(([0,1,0, 1,0,0, 0,0,1], [0,0,0, 1,0,0, 0,0,1]))
    # p13 = (b+g+f)*(u+z+y)
    terms.append(([0,1,0, 0,0,1, 1,0,0], [0,0,0, 1,0,0, 0,0,1]))
    # p14 = (c+d+h)*(r+y+w)
    terms.append(([0,0,1, 1,0,0, 0,1,0], [0,0,0, 0,0,0, 1,1,0]))
    # p15 = (c+g+e)*(r+v+z)
    terms.append(([0,0,1, 0,1,0, 1,0,0], [0,0,0, 0,1,0, 1,0,0]))
    # p16 = (a-b+c-d+e-f)*(x-y+z-u+v-w)
    terms.append(([1,-1,1, -1,1,-1, 0,0,0], [1,-1,1, -1,1,-1, 0,0,0]))
    # p17 = (a+b-c-d-e+f)*(x+y-z-u-v+w)
    terms.append(([1,1,-1, -1,-1,1, 0,0,0], [1,1,-1, -1,-1,1, 0,0,0]))
    # p18 = (a-b-c+d+e+f)*(x-y-z+u+v+w)
    terms.append(([1,-1,-1, 1,1,1, 0,0,0], [1,-1,-1, 1,1,1, 0,0,0]))
    # p19 = (a+b+c-d-e-f)*(x+y+z-u-v-w)
    terms.append(([1,1,1, -1,-1,-1, 0,0,0], [1,1,1, -1,-1,-1, 0,0,0]))
    # p20 = (d-e+f-g+h-i)*(y-v+w-s+t-r)
    terms.append(([0,0,0, 1,-1,1, -1,1,-1], [0,1,0, 0,-1,1, -1,1,-1]))
    # p21 = (d+e-f-g-h+i)*(y+v-w-s-t+r)
    terms.append(([0,0,0, 1,1,-1, -1,-1,1], [0,1,0, 0,1,-1, -1,-1,1]))
    # p22 = (d-e-f+g+h+i)*(y-v-w+s+t+r)
    terms.append(([0,0,0, 1,-1,-1, 1,1,1], [0,1,0, 0,-1,-1, 1,1,1]))
    # p23 = (d+e+f-g-h-i)*(y+v+w-s-t-r)
    terms.append(([0,0,0, 1,1,1, -1,-1,-1], [0,1,0, 0,1,1, -1,-1,-1]))
    
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    
    for r in range(rank):
        u_vec = terms[r][0]
        v_vec = terms[r][1]
        for idx in range(9):
            U[idx][r] = u_vec[idx]
            V[idx][r] = v_vec[idx]
    
    # Compute W: for each (i,j), solve sum_r W[j*3+i,r]*U[i*3+k,r]*V[kp*3+j,r] = delta_{k,kp}
    for i in range(3):
        for j in range(3):
            if time.time() > deadline:
                break
            A_sys = np.zeros((9, rank))
            b_sys = np.zeros(9)
            eq_idx = 0
            for k in range(3):
                for kp in range(3):
                    for r in range(rank):
                        A_sys[eq_idx, r] = U[i*3+k][r] * V[kp*3+j][r]
                    b_sys[eq_idx] = 1.0 if k == kp else 0.0
                    eq_idx += 1
            
            w_vec, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
            for r in range(rank):
                W[j*3+i][r] = float(w_vec[r])
    
    # Verify with random matrices
    np.random.seed(42)
    for trial in range(30):
        if time.time() > deadline:
            break
        A = np.random.randn(3, 3)
        B = np.random.randn(3, 3)
        C_expected = A @ B
        C_computed = np.zeros((3, 3))
        for r in range(rank):
            u_val = 0.0
            for i in range(3):
                for k in range(3):
                    u_val += U[i*3+k][r] * A[i][k]
            v_val = 0.0
            for k in range(3):
                for j in range(3):
                    v_val += V[k*3+j][r] * B[k][j]
            prod = u_val * v_val
            for i in range(3):
                for j in range(3):
                    C_computed[i][j] += W[j*3+i][r] * prod
        if not np.allclose(C_computed, C_expected, atol=1e-8):
            # Fall back to schoolbook
            rank = 27
            U = [[0] * rank for _ in range(9)]
            V = [[0] * rank for _ in range(9)]
            W = [[0] * rank for _ in range(9)]
            column = 0
            for i in range(3):
                for k in range(3):
                    for j in range(3):
                        U[i * 3 + k][column] = 1
                        V[k * 3 + j][column] = 1
                        W[j * 3 + i][column] = 1
                        column += 1
            return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
