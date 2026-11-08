import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140  # 10s margin before 150s kill
    
    n, m, p = 2, 3, 3  # A: n x m, B: m x p, C: n x p
    # Output index convention: W[k*n + i, r] = C[i,k]
    # So output row index = k*n + i where k is column of C, i is row of C
    num_outputs = p * n  # 6
    num_inputs_a = n * m  # 6
    num_inputs_b = m * p  # 9
    
    def make_tensor(U, V, W, rank):
        """Compute the multiplication tensor from factorization using einsum"""
        # tensor[ia, ib, out] = sum_r U[ia,r] * V[ib,r] * W[out,r]
        return np.einsum('ir,jr,kr->ijk', U, V, W)
    
    def target_tensor():
        T = np.zeros((num_inputs_a, num_inputs_b, num_outputs))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ia = i * m + j
                    ib = j * p + k
                    out_idx = k * n + i
                    T[ia, ib, out_idx] = 1.0
        return T
    
    target = target_tensor()
    
    # Rank 17 construction
    rank = 17
    U = np.zeros((num_inputs_a, rank))
    V = np.zeros((num_inputs_b, rank))
    W = np.zeros((num_outputs, rank))
    
    # Indices:
    # A: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    # B: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)
    # C: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1), 4:(0,2), 5:(1,2)
    # Note: out_idx = k*n + i.
    # C[0,0] -> 0*2+0=0
    # C[1,0] -> 0*2+1=1
    # C[0,1] -> 1*2+0=2
    # C[1,1] -> 1*2+1=3
    # C[0,2] -> 2*2+0=4
    # C[1,2] -> 2*2+1=5
    
    # Strassen for A[:,:2] * B[:2,:2]
    # M1: (a00+a11)(b00+b11) -> C[0,0](0), C[1,1](3)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (a10+a11)*b00 -> C[1,0](1) +, C[1,1](3) -
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3: a00*(b01-b11) -> C[0,1](2) +, C[1,1](3) +
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4: a11*(b10-b00) -> C[0,0](0) +, C[1,0](1) +
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5: (a00+a01)*b11 -> C[0,0](0) -, C[0,1](2) +
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6: (a10-a00)*(b00+b01) -> C[1,1](3) +
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (a01-a11)*(b10+b11) -> C[0,0](0) +
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # A[:,:2] * B[:2,2] (2x2 * 2x1)
    # r=7: a00*b02 -> C[0,2](4)
    U[0, 7] = 1; V[2, 7] = 1; W[4, 7] = 1
    # r=8: a01*b12 -> C[0,2](4)
    U[1, 8] = 1; V[5, 8] = 1; W[4, 8] = 1
    # r=9: a10*b02 -> C[1,2](5)
    U[3, 9] = 1; V[2, 9] = 1; W[5, 9] = 1
    # r=10: a11*b12 -> C[1,2](5)
    U[4, 10] = 1; V[5, 10] = 1; W[5, 10] = 1
    
    # A[:,2] * B[2,:2] (2x1 * 1x2)
    # r=11: a02*b20 -> C[0,0](0)
    U[2, 11] = 1; V[6, 11] = 1; W[0, 11] = 1
    # r=12: a02*b21 -> C[0,1](2)
    U[2, 12] = 1; V[7, 12] = 1; W[2, 12] = 1
    # r=13: a12*b20 -> C[1,0](1)
    U[5, 13] = 1; V[6, 13] = 1; W[1, 13] = 1
    # r=14: a12*b21 -> C[1,1](3)
    U[5, 14] = 1; V[7, 14] = 1; W[3, 14] = 1
    
    # A[:,2] * B[2,2] (2x1 * 1x1)
    # r=15: a02*b22 -> C[0,2](4)
    U[2, 15] = 1; V[8, 15] = 1; W[4, 15] = 1
    # r=16: a12*b22 -> C[1,2](5)
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
    
    # Verify
    if time.time() - start_time > deadline:
        # Shouldn't happen, but return best effort
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    computed = make_tensor(U, V, W, rank)
    max_err = np.max(np.abs(computed - target))
    
    if max_err < 1e-10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback (should not happen)
    rank = 18
    U = np.zeros((num_inputs_a, rank))
    V = np.zeros((num_inputs_b, rank))
    W = np.zeros((num_outputs, rank))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ia = i * m + j
                ib = j * p + k
                out_idx = k * n + i
                U[ia, r] = 1
                V[ib, r] = 1
                W[out_idx, r] = 1
                r += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
