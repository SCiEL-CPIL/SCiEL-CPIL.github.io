import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin from 120s limit
    
    n, m, p = DIMENSIONS
    
    # Strategy: Block decomposition
    # We decompose the 2x3x7 tensor into:
    # 1. A part involving the first 2 columns of A and the first 6 columns of B (2x2x6 subproblem)
    #    Wait, A is 2x3. Let's split A into A_0 (2x2, cols 0,1) and A_1 (2x1, col 2).
    #    Let's split B into B_0 (2x6, rows 0,1) and B_1 (1x7, row 2).
    #    
    #    C = A_0 B_0 + A_0 B_1 + A_1 B_0 + A_1 B_1
    #    
    #    Term 1: A_0 (2x2) * B_0 (2x6). 
    #    We can treat B_0 as three 2x2 blocks: B_00 (cols 0,1), B_01 (cols 2,3), B_02 (cols 4,5).
    #    Each 2x2x2 block can be done with rank 7 (Strassen).
    #    So A_0 * B_0 requires 3 * 7 = 21 multiplications.
    #    
    #    Term 2: A_0 (2x2) * B_1 (1x7).
    #    B_1 is a row vector. A_0 B_1 is (2x2)(1x7) -> wait, dimensions don't match for standard matmul.
    #    A_0 is 2x2. B_1 is 1x7. They can't be multiplied directly in the standard sense unless we view them as vectors?
    #    No, the intermediate dimension is m.
    #    A is (n x m) = (2 x 3). B is (m x p) = (3 x 7).
    #    A_0 is rows 0..1, cols 0..1 of A. Size 2x2.
    #    B_0 is rows 0..1, cols 0..6 of B. Size 2x7.
    #    A_1 is rows 0..1, col 2 of A. Size 2x1.
    #    B_1 is row 2, cols 0..6 of B. Size 1x7.
    #    
    #    C = A_0 B_0 + A_1 B_1?
    #    A_0 B_0: (2x2) * (2x7) = 2x7. Valid.
    #    A_1 B_1: (2x1) * (1x7) = 2x7. Valid.
    #    But what about the cross terms?
    #    The full product is sum over k=0..2 of A[:,k] B[k,:].
    #    A[:,0:2] is A_0. B[0:2,:] is B_0.
    #    A[:,2] is A_1. B[2,:] is B_1.
    #    So C = A_0 B_0 + A_1 B_1.
    #    
    #    Let's re-evaluate the rank of A_0 B_0 (2x2 * 2x7).
    #    We can split B_0 into three 2x2 blocks: B_{0,0} (cols 0,1), B_{0,1} (cols 2,3), B_{0,2} (cols 4,5).
    #    Wait, B_0 is 2x7. It has 7 columns.
    #    We can group columns 0,1 -> 2x2 block.
    #    Columns 2,3 -> 2x2 block.
    #    Columns 4,5 -> 2x2 block.
    #    Column 6 -> 2x1 block.
    #    
    #    So A_0 B_0 = A_0 B_{0,0} + A_0 B_{0,1} + A_0 B_{0,2} + A_0 B_{0,3}
    #    where B_{0,3} is the 2x1 block (col 6).
    #    
    #    Rank of 2x2x2 (Strassen) is 7.
    #    So A_0 B_{0,0} costs 7.
    #    A_0 B_{0,1} costs 7.
    #    A_0 B_{0,2} costs 7.
    #    A_0 B_{0,3} is 2x2 * 2x1. This is effectively 3 scalar multiplications?
    #    Let's check 2x2 * 2x1.
    #    A = [[a,b],[c,d]], v = [x,y].
    #    A v = [ax+by, cx+dy].
    #    This is bilinear. Can we do it with rank < 4?
    #    Tensor is 2x2x1. Rank is 3? No, 4?
    #    Actually, 2x2x1 is just 4 scalar products. Is there a shortcut?
    #    (a+c)(x+y) - ... no.
    #    Rank of 2x2x1 is 3?
    #    Let's check. T_{ijk} = delta_{ik} (if we view it as A_{ik} v_k).
    #    Wait, the tensor for 2x2 * 2x1 has indices i (out row), j (A col), k (v col).
    #    T_{i,j,k} = delta_{j,k}.
    #    This is the identity tensor of size 2. Rank of identity tensor of size d is d.
    #    So rank of 2x2x1 is 2?
    #    Let's verify.
    #    We want to represent A v.
    #    u_1 = [1, 0], v_1 = [1, 0], w_1 = [1, 0]?
    #    If we use rank 2:
    #    r1: u=[1,0], v=[1,0] -> A_{00} v_0. Output adds to C_0.
    #    r2: u=[0,1], v=[0,1] -> A_{11} v_1. Output adds to C_1.
    #    This misses A_{01}v_1 and A_{10}v_0.
    #    So rank 2 is not enough.
    #    Rank 3?
    #    It is known that the rank of the 2x2x1 tensor is 3? No, it's 4?
    #    Actually, for matrix multiplication A(n x m) * B(m x p), the rank is at least max(n*m, m*p, n*p).
    #    Here n=2, m=2, p=1. max(4, 2, 2) = 4.
    #    So rank is 4.
    #    
    #    So A_0 B_{0,3} costs 4.
    #    Total for A_0 B_0 = 7 + 7 + 7 + 4 = 25.
    #    
    #    Now A_1 B_1:
    #    A_1 is 2x1. B_1 is 1x7.
    #    This is outer product of a 2-vector and a 7-vector.
    #    Rank is 1?
    #    A_1 = [a, b]^T. B_1 = [c0, ..., c6].
    #    C = [a*c0, ..., a*c6, b*c0, ..., b*c6].
    #    This is a rank 1 tensor?
    #    T_{i,k} = A_{i,0} B_{0,k}.
    #    Yes, this is a rank 1 tensor in the bilinear sense?
    #    Wait, the decomposition is sum of rank 1 tensors u v^T w^T.
    #    Here the "matrix" is just a vector outer product.
    #    The tensor T_{i, j, k} where j is the only index (j=0).
    #    T_{i, 0, k} = A_{i,0} B_{0,k}.
    #    This can be written as u_i * v_0 * w_k?
    #    u = A_1, v = [1], w = B_1.
    #    So yes, rank 1.
    #    
    #    Total rank = 25 + 1 = 26.
    #    
    #    Let's double check if we can do better on A_0 B_0.
    #    A_0 is 2x2. B_0 is 2x7.
    #    We split B_0 into 2x2, 2x2, 2x2, 2x1.
    #    Is there a better way?
    #    Maybe split A_0? No, A_0 is small.
    #    Maybe group B columns differently?
    #    If we group B_0 into 2x7 directly?
    #    Rank of 2x2x7?
    #    Lower bound max(4, 14, 14) = 14.
    #    Strassen gives 7 * ceil(7/2) = 7*4 = 28? No.
    #    The rank of 2x2xp is 3p - 2? (For p even? No).
    #    For 2x2xp, rank is 3p - 2 if p is even?
    #    Let's check p=2. Rank 7. 3(2)-2 = 4. No.
    #    Actually, rank of 2x2xp is 3p - 2 is a known bound?
    #    For p=1, rank 4. 3(1)-2 = 1. No.
    #    For p=2, rank 7.
    #    For p=3, rank 10?
    #    Formula: rank(2x2xp) = 3p - 2 for p >= 2?
    #    Let's check p=2. 3(2)-2 = 4. But Strassen is 7.
    #    Wait, the rank of 2x2x2 is 7.
    #    So 3p-2 is not correct for p=2.
    #    Actually, the rank of 2x2xp is 3p - 2 is for the *complex* rank? Or maybe I'm misremembering.
    #    Let's stick to the block decomposition which is safe.
    #    
    #    So, Rank = 26.
    #    
    #    Let's construct the decomposition.
    
    # We will build U, V, W lists of lists.
    # U shape: (n*m, rank) = (6, 26)
    # V shape: (m*p, rank) = (21, 26)
    # W shape: (p*n, rank) = (14, 26)
    
    # Indices:
    # U[i*m + j] for A[i,j]
    # V[j*p + k] for B[j,k]
    # W[k*n + i] for C[i,k]
    
    U = [[0] * 26 for _ in range(6)]
    V = [[0] * 26 for _ in range(21)]
    W = [[0] * 26 for _ in range(14)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # A block at A rows 0,1 cols 0,1 (A_0)
    # B block at B rows 0,1 cols c0, c0+1
    # C block at C rows 0,1 cols c0, c0+1
    def add_strassen_block(b_col_start):
        nonlocal col
        # Strassen formulas for 2x2 * 2x2
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)h
        # m6 = (c-a)(e+f)
        # m7 = (b-d)(g+h)
        
        # Indices in U (A_0 is fixed at cols 0,1)
        # A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
        u_a = 0*3+0 # 0
        u_b = 0*3+1 # 1
        u_c = 1*3+0 # 3
        u_d = 1*3+1 # 4
        
        # Indices in V (B block rows 0,1, cols b_col_start, b_col_start+1)
        # B[0, b_col_start]=e, B[0, b_col_start+1]=f
        # B[1, b_col_start]=g, B[1, b_col_start+1]=h
        v_e = 0*7 + b_col_start
        v_f = 0*7 + b_col_start + 1
        v_g = 1*7 + b_col_start
        v_h = 1*7 + b_col_start + 1
        
        # Indices in W (C block rows 0,1, cols b_col_start, b_col_start+1)
        # C[0, b_col_start] -> W[b_col_start*2 + 0]
        # C[1, b_col_start] -> W[b_col_start*2 + 1]
        # C[0, b_col_start+1] -> W[(b_col_start+1)*2 + 0]
        # C[1, b_col_start+1] -> W[(b_col_start+1)*2 + 1]
        w_c00 = b_col_start*2 + 0
        w_c10 = b_col_start*2 + 1
        w_c01 = (b_col_start+1)*2 + 0
        w_c11 = (b_col_start+1)*2 + 1
        
        # m1 = (a+d)(e+h)
        U[u_a][col] = 1
        U[u_d][col] = 1
        V[v_e][col] = 1
        V[v_h][col] = 1
        W[w_c00][col] = 1
        W[w_c11][col] = 1
        col += 1
        
        # m2 = (c+d)e
        U[u_c][col] = 1
        U[u_d][col] = 1
        V[v_e][col] = 1
        W[w_c10][col] = 1
        col += 1
        
        # m3 = a(f-h)
        U[u_a][col] = 1
        V[v_f][col] = 1
        V[v_h][col] = -1
        W[w_c01][col] = 1
        col += 1
        
        # m4 = d(g-e)
        U[u_d][col] = 1
        V[v_g][col] = 1
        V[v_e][col] = -1
        W[w_c11][col] = 1
        col += 1
        
        # m5 = (a+b)h
        U[u_a][col] = 1
        U[u_b][col] = 1
        V[v_h][col] = 1
        W[w_c01][col] = 1
        W[w_c11][col] = 1
        col += 1
        
        # m6 = (c-a)(e+f)
        U[u_c][col] = 1
        U[u_a][col] = -1
        V[v_e][col] = 1
        V[v_f][col] = 1
        W[w_c10][col] = 1
        col += 1
        
        # m7 = (b-d)(g+h)
        U[u_b][col] = 1
        U[u_d][col] = -1
        V[v_g][col] = 1
        V[v_h][col] = 1
        W[w_c00][col] = 1
        W[w_c10][col] = 1
        col += 1

    # Add 3 Strassen blocks for B_0 cols (0,1), (2,3), (4,5)
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    # Now handle A_0 * B_{0,3} (2x2 * 2x1)
    # B_{0,3} is B rows 0,1, col 6.
    # This is rank 4.
    # C[0,6] += A[0,0]B[0,6] + A[0,1]B[1,6]
    # C[1,6] += A[1,0]B[0,6] + A[1,1]B[1,6]
    
    # r1: A[0,0] B[0,6] -> C[0,6]
    U[0][col] = 1
    V[0*7+6][col] = 1
    W[6*2+0][col] = 1
    col += 1
    
    # r2: A[0,1] B[1,6] -> C[0,6]
    U[1][col] = 1
    V[1*7+6][col] = 1
    W[6*2+0][col] = 1
    col += 1
    
    # r3: A[1,0] B[0,6] -> C[1,6]
    U[3][col] = 1
    V[0*7+6][col] = 1
    W[6*2+1][col] = 1
    col += 1
    
    # r4: A[1,1] B[1,6] -> C[1,6]
    U[4][col] = 1
    V[1*7+6][col] = 1
    W[6*2+1][col] = 1
    col += 1
    
    # Now handle A_1 * B_1 (2x1 * 1x7)
    # A_1 is A col 2 (indices 2, 5 in U).
    # B_1 is B row 2 (indices 14..20 in V).
    # C is all of C (indices 0..13 in W).
    # Rank 1.
    # C[i,k] += A[i,2] B[2,k]
    # We need one term that covers all i,k?
    # No, rank 1 tensor is u v w.
    # u is vector in U space (size 6).
    # v is vector in V space (size 21).
    # w is vector in W space (size 14).
    # u_i = A[i,2] -> u = [0,0,1,0,0,1]
    # v_j = B[2,j] -> v = [0,0,0,0,0,0,0, 0,0,0,0,0,0,0, 1,1,1,1,1,1,1]
    # w_k = C[i,k] -> w = [1,1,1,1,1,1,1, 1,1,1,1,1,1,1]
    # Wait, w index is k*n+i.
    # For each k, i=0,1.
    # So w should have 1s at all positions?
    # Yes, because C[i,k] gets contribution from A[i,2]B[2,k] for all i,k.
    
    # u
    U[2][col] = 1
    U[5][col] = 1
    # v
    for k in range(7):
        V[2*7+k][col] = 1
    # w
    for k in range(7):
        for i in range(2):
            W[k*2+i][col] = 1
    col += 1
    
    # Total rank should be 26.
    # 3*7 = 21
    # 4
    # 1
    # Total 26.
    
    if time.time() > deadline:
        pass

    return {"U": U, "V": V, "W": W}
