import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen's algorithm for 2x2 * 2x2 blocks.
    # We can split the 2x12 matrix into 6 blocks of 2x2.
    # For each 2x2 * 2x2 multiplication, Strassen uses rank 7.
    # Since the left matrix (2x2) is shared across all 6 blocks of the right matrix,
    # we can define the 7 "scalar" multiplications (tensors) once and reuse them.
    #
    # Let A = [[a, b], [c, d]] and B = [[e, f], [g, h]].
    # The 7 products are:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    #
    # The result C = [[c11, c12], [c21, c22]] is:
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    #
    # We have 6 such blocks. Let the blocks of B be B_0, ..., B_5.
    # The total rank will be 6 * 7 = 42.
    #
    # Indexing conventions:
    # U rows: i*m + j  (i in 0..n-1, j in 0..m-1) -> 0, 1, 2, 3
    # V rows: j*p + k  (j in 0..m-1, k in 0..p-1) -> 0..23
    # W rows: k*n + i  (k in 0..p-1, i in 0..n-1) -> 0..23
    #
    # For block b (0..5):
    # B block indices:
    # e_{b} is B[0, 2b] -> V row 0*12 + 2b = 2b
    # f_{b} is B[0, 2b+1] -> V row 0*12 + 2b+1 = 2b+1
    # g_{b} is B[1, 2b] -> V row 1*12 + 2b = 12+2b
    # h_{b} is B[1, 2b+1] -> V row 1*12 + 2b+1 = 12+2b+1
    #
    # A indices (same for all blocks):
    # a = A[0,0] -> U row 0
    # b = A[0,1] -> U row 1
    # c = A[1,0] -> U row 2
    # d = A[1,1] -> U row 3
    #
    # Result C indices for block b:
    # C[0, 2b] -> W row (2b)*2 + 0 = 4b
    # C[0, 2b+1] -> W row (2b+1)*2 + 0 = 4b+2
    # C[1, 2b] -> W row (2b)*2 + 1 = 4b+1
    # C[1, 2b+1] -> W row (2b+1)*2 + 1 = 4b+3
    #
    # Rank = 42.
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Define the 7 Strassen tensors for one 2x2 * 2x2 block.
    # Each tensor is defined by its U, V, W vectors.
    # U vector has 4 elements (indices 0..3 for a,b,c,d)
    # V vector has 4 elements (indices 0..3 for e,f,g,h)
    # W vector has 4 elements (indices 0..3 for c11,c12,c21,c22)
    
    # m1 = (a+d)(e+h)
    # U: [1, 0, 0, 1]
    # V: [1, 0, 0, 1]
    # W: [1, 0, 0, 1]  (contributes to c11 and c22)
    
    # m2 = (c+d)e
    # U: [0, 0, 1, 1]
    # V: [1, 0, 0, 0]
    # W: [0, 0, 1, 0]  (contributes to c21)
    
    # m3 = a(f-h)
    # U: [1, 0, 0, 0]
    # V: [0, 1, 0, -1]
    # W: [0, 1, 0, 1]  (contributes to c12 and c22)
    
    # m4 = d(g-e)
    # U: [0, 0, 0, 1]
    # V: [0, 0, 1, -1]  # Wait, g-e means V[2]=1, V[0]=-1
    # V: [-1, 0, 1, 0]
    # W: [1, 0, 1, 0]  (contributes to c11 and c21)
    
    # m5 = (a+b)h
    # U: [1, 1, 0, 0]
    # V: [0, 0, 0, 1]
    # W: [1, 0, 0, 0]  (contributes to c11)
    
    # m6 = (c-a)(e+f)
    # U: [-1, 0, 1, 0]
    # V: [1, 1, 0, 0]
    # W: [0, 0, 0, 1]  (contributes to c22)
    
    # m7 = (b-d)(g+h)
    # U: [0, 1, 0, -1]
    # V: [0, 0, 1, 1]
    # W: [1, 0, 0, 0]  (contributes to c11)

    tensors = [
        # (U_vec, V_vec, W_vec)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # m1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),  # m2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # m3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),  # m4
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]),  # m5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # m6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # m7
    ]

    for b in range(6):
        for r in range(7):
            col = b * 7 + r
            U_vec, V_vec, W_vec = tensors[r]
            
            # Fill U
            for idx in range(4):
                if U_vec[idx] != 0:
                    U[idx][col] += U_vec[idx]
            
            # Fill V
            # V_vec indices: 0->e, 1->f, 2->g, 3->h
            # Global V indices:
            # e: 2b
            # f: 2b+1
            # g: 12+2b
            # h: 12+2b+1
            v_indices = [2*b, 2*b+1, 12+2*b, 12+2*b+1]
            for idx in range(4):
                if V_vec[idx] != 0:
                    V[v_indices[idx]][col] += V_vec[idx]
            
            # Fill W
            # W_vec indices: 0->c11, 1->c12, 2->c21, 3->c22
            # Global W indices for block b:
            # c11: 4b
            # c12: 4b+2
            # c21: 4b+1
            # c22: 4b+3
            w_indices = [4*b, 4*b+2, 4*b+1, 4*b+3]
            for idx in range(4):
                if W_vec[idx] != 0:
                    W[w_indices[idx]][col] += W_vec[idx]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
