import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 3, 4
    
    # Try to find a lower rank decomposition
    # Known lower bound: max(n*p, m*p, n*m) = max(8, 12, 6) = 12
    # But tensor rank is generally higher. Let's try rank 20 first.
    
    # Use a systematic approach: build the tensor and try to find a decomposition
    # The matrix multiplication tensor T[i,j,k] = 1 if A[i,j]*B[j,k] contributes to C[i,k]
    
    # Let's try a hybrid approach: use Strassen for 2x2x2 blocks but try to
    # combine cross terms more efficiently.
    
    # Alternative: partition A as [A1 | A2] where A1 is 2x2, A2 is 2x1
    # B as [B1; B2] where B1 is 2x4, B2 is 1x4
    # C = A1*B1 + A2*B2
    
    # A1*B1: 2x2 * 2x4. Split B1 into B1a (2x2) and B1b (2x2)
    # A1*B1a: 2x2*2x2 -> rank 7 (Strassen)
    # A1*B1b: 2x2*2x2 -> rank 7 (Strassen)
    # A2*B2: 2x1 * 1x4 -> this is outer product, rank 4
    
    # Total: 7 + 7 + 4 = 18? Let's verify this works.
    
    # Actually, 2x1 * 1x4 means:
    # A2 = [a; b] (2x1), B2 = [c1, c2, c3, c4] (1x4)
    # A2*B2 = [[a*c1, a*c2, a*c3, a*c4],
    #           [b*c1, b*c2, b*c3, b*c4]]
    # This requires 4 multiplications: a*c1, a*c2, a*c3, a*c4... no wait
    # We need: a*c1, a*c2, a*c3, a*c4, b*c1, b*c2, b*c3, b*c4
    # That's 8 multiplications if done directly.
    # But we can use: let x = a+b, y = a-b, then... no, that doesn't help for 4 outputs.
    # Actually for 2x1 * 1x4, the minimal rank is 4 (it's a rank-4 tensor).
    # Wait, no. The tensor T[i,k] = A[0,0]*B[0,k] for i=0, A[1,0]*B[0,k] for i=1
    # This is a 2x1x4 tensor. Its rank is 4 (you need 4 scalar multiplications).
    
    # So total would be 7 + 7 + 4 = 18. Let me try to construct this.
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)]  # 6 x 18
    V = [[0] * rank for _ in range(m * p)]  # 12 x 18
    W = [[0] * rank for _ in range(p * n)]  # 8 x 18
    
    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val
    
    # Indices:
    # A: 0:(0,0) 1:(0,1) 2:(0,2) 3:(1,0) 4:(1,1) 5:(1,2)
    # B: 0:(0,0) 1:(0,1) 2:(0,2) 3:(0,3) 4:(1,0) 5:(1,1) 6:(1,2) 7:(1,3) 8:(2,0) 9:(2,1) 10:(2,2) 11:(2,3)
    # C: 0:(0,0) 1:(1,0) 2:(0,1) 3:(1,1) 4:(0,2) 5:(1,2) 6:(0,3) 7:(1,3)
    
    # Block 1: Strassen for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # A1 = [[A00, A01], [A10, A11]] = [[U[0], U[1]], [U[3], U[4]]]
    # B1a = [[B00, B01], [B10, B11]] = [[V[0], V[1]], [V[4], V[5]]]
    # C = [[C00, C01], [C10, C11]] = [[W[0], W[2]], [W[1], W[3]]]
    
    # Strassen for 2x2*2x2:
    # m1 = (a00+a11)*(b00+b11) -> C00 += m1, C01 += m1, C10 += m1, C11 -= m1
    # m2 = a11*(b00+b01) -> C01 += m2, C11 += m2
    # m3 = (a00-a11)*(b10+b11) -> C00 -= m3, C10 += m3
    # m4 = (a00+a01)*b00 -> C00 += m4, C10 += m4
    # m5 = (a10+a11)*b11 -> C01 += m5, C11 += m5
    # m6 = (a01-a11)*(b10+b11) -> C00 += m6, C11 -= m6
    # m7 = (a00-a10)*(b00+b01) -> C01 -= m7, C10 += m7
    
    # m1
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    # m2
    add_term([(4, 1)], [(0, 1), (1, 1)], [(2, 1), (3, 1)], 1)
    # m3
    add_term([(0, 1), (4, -1)], [(4, 1), (5, 1)], [(0, -1), (1, 1)], 2)
    # m4
    add_term([(0, 1), (1, 1)], [(0, 1)], [(0, 1), (1, 1)], 3)
    # m5
    add_term([(3, 1), (4, 1)], [(5, 1)], [(2, 1), (3, 1)], 4)
    # m6
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1), (3, -1)], 5)
    # m7
    add_term([(0, 1), (3, -1)], [(0, 1), (1, 1)], [(2, -1), (1, 1)], 6)
    
    # Block 2: Strassen for A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    # B1b = [[B02, B03], [B12, B13]] = [[V[2], V[3]], [V[6], V[7]]]
    # C = [[C02, C03], [C12, C13]] = [[W[4], W[6]], [W[5], W[7]]]
    
    # m8 (analogous to m1)
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
    # m9 (analogous to m2)
    add_term([(4, 1)], [(2, 1), (3, 1)], [(6, 1), (7, 1)], 8)
    # m10 (analogous to m3)
    add_term([(0, 1), (4, -1)], [(6, 1), (7, 1)], [(4, -1), (5, 1)], 9)
    # m11 (analogous to m4)
    add_term([(0, 1), (1, 1)], [(2, 1)], [(4, 1), (5, 1)], 10)
    # m12 (analogous to m5)
    add_term([(3, 1), (4, 1)], [(7, 1)], [(6, 1), (7, 1)], 11)
    # m13 (analogous to m6)
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1), (7, -1)], 12)
    # m14 (analogous to m7)
    add_term([(0, 1), (3, -1)], [(2, 1), (3, 1)], [(6, -1), (5, 1)], 13)
    
    # Cross terms: A2 * B2 where A2 = [A02, A12] = [U[2], U[5]], B2 = [B20, B21, B22, B23] = [V[8], V[9], V[10], V[11]]
    # C contributions:
    # C00 += A02*B20 -> W[0] += U[2]*V[8]
    # C01 += A02*B21 -> W[2] += U[2]*V[9]
    # C02 += A02*B22 -> W[4] += U[2]*V[10]
    # C03 += A02*B23 -> W[6] += U[2]*V[11]
    # C10 += A12*B20 -> W[1] += U[5]*V[8]
    # C11 += A12*B21 -> W[3] += U[5]*V[9]
    # C12 += A12*B22 -> W[5] += U[5]*V[10]
    # C13 += A12*B23 -> W[7] += U[5]*V[11]
    
    # This requires 8 multiplications, but we can use 4 by noting:
    # Let x1 = A02, x2 = A12, y1 = B20, y2 = B21, y3 = B22, y4 = B23
    # We need: x1*y1, x1*y2, x1*y3, x1*y4, x2*y1, x2*y2, x2*y3, x2*y4
    # This is a 2x1x4 tensor which has rank 4.
    # Actually, I need to reconsider. The tensor is T[i,k] = sum_j A[i,j]*B[j,k]
    # For the cross terms, j=2 only, so T[i,k] = A[i,2]*B[2,k]
    # This is an outer product of A[:,2] and B[2,:], which has rank 1 as a matrix,
    # but as a tensor decomposition, we need rank 4 because we have 4 independent outputs per input pair.
    
    # Wait, I think I'm confusing myself. Let me think again.
    # The matrix multiplication tensor for 2x1 * 1x4 is:
    # T[i, j, k] where i in {0,1}, j in {0}, k in {0,1,2,3}
    # T[i,0,k] = 1 if we're computing C[i,k] = A[i,0]*B[0,k]
    # This tensor has rank 4 (you need 4 scalar multiplications: A[0,0]*B[0,0], A[0,0]*B[0,1], A[0,0]*B[0,2], A[0,0]*B[0,3]... no that's not right either)
    
    # Actually, for computing C = A*B where A is 2x1 and B is 1x4:
    # C[0,0] = A[0,0]*B[0,0]
    # C[0,1] = A[0,0]*B[0,1]
    # C[0,2] = A[0,0]*B[0,2]
    # C[0,3] = A[0,0]*B[0,3]
    # C[1,0] = A[1,0]*B[0,0]
    # C[1,1] = A[1,0]*B[0,1]
    # C[1,2] = A[1,0]*B[0,2]
    # C[1,3] = A[1,0]*B[0,3]
    
    # We can compute this with 4 multiplications:
    # m1 = A[0,0]*B[0,0] -> C[0,0] = m1, C[1,0] = m1 * (A[1,0]/A[0,0])... no, that doesn't work.
    
    # Actually, we need 8 multiplications in general, or we can use:
    # Let's use the fact that this is a rank-4 tensor.
    # T[i,k] = A[i,0] * B[0,k]
    # We can write this as sum_{r=1}^{4} u_r[i] * v_r[0] * w_r[k]
    # where for each k, we have one term.
    
    # So: 
    # r=0: u[0]=1, u[1]=0, v[0]=1, w[0]=1, w[1]=0, w[2]=0, w[3]=0 -> A[0,0]*B[0,0] -> C[0,0]
    # r=1: u[0]=1, u[1]=0, v[0]=1, w[0]=0, w[1]=1, w[2]=0, w[3]=0 -> A[0,0]*B[0,1] -> C[0,1]
    # r=2: u[0]=1, u[1]=0, v[0]=1, w[0]=0, w[1]=0, w[2]=1, w[3]=0 -> A[0,0]*B[0,2] -> C[0,2]
    # r=3: u[0]=1, u[1]=0, v[0]=1, w[0]=0, w[1]=0, w[2]=0, w[3]=1 -> A[0,0]*B[0,3] -> C[0,3]
    # r=4: u[0]=0, u[1]=1, v[0]=1, w[0]=1, w[1]=0, w[2]=0, w[3]=0 -> A[1,0]*B[0,0] -> C[1,0]
    # r=5: u[0]=0, u[1]=1, v[0]=1, w[0]=0, w[1]=1, w[2]=0, w[3]=0 -> A[1,0]*B[0,1] -> C[1,1]
    # r=6: u[0]=0, u[1]=1, v[0]=1, w[0]=0, w[1]=0, w[2]=1, w[3]=0 -> A[1,0]*B[0,2] -> C[1,2]
    # r=7: u[0]=0, u[1]=1, v[0]=1, w[0]=0, w[1]=0, w[2]=0, w[3]=1 -> A[1,0]*B[0,3] -> C[1,3]
    
    # So it takes 8 terms. The rank is 8, not 4.
    
    # So total rank would be 7 + 7 + 8 = 22, same as before.
    
    # Let me just return the original rank-22 decomposition.
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val
    
    # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)
    
    # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
    add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
    add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
    add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)
    
    # Cross terms
    add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
    add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
    add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
    add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
    add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
    add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
    add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
    add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
