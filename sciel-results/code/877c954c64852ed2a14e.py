import numpy as np
import time
from math import exp, log, sqrt, pi

def construct():
    n, m, p = 3, 4, 8
    start_time = time.time()
    deadline = start_time + 115

    # Build the target tensor
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                target[(a, b, c)] = 1.0

    U_cols = []
    V_cols = []
    W_cols = []

    def add_rank1(U_vec, V_vec, W_vec):
        U_cols.append(U_vec)
        V_cols.append(V_vec)
        W_cols.append(W_vec)

    # DFT-based decomposition for 1x4x8 matrix multiplication
    # For a single row i, the computation is: C_k = sum_{j=0}^{3} A_j * B_{j,k}
    # We can use DFT to diagonalize the 4x4 subproblem for each pair of k values.
    # The key insight: for fixed k, the map j -> (j,k) can be seen as a 4-dimensional
    # inner product. We use DFT to decompose this into 4 rank-1 terms.
    
    # For the 8 output dimensions, we can view them as two groups of 4:
    # Group 0: k = 0,1,2,3
    # Group 1: k = 4,5,6,7
    # Each group uses 4 rank-1 terms via DFT, so total 8 terms per row.
    
    # DFT matrix for size 4: F[l,j] = exp(2*pi*i*l*j/4) = exp(i*pi/2 * l*j)
    # F[0] = [1, 1, 1, 1]
    # F[1] = [1, i, -1, -i]
    # F[2] = [1, -1, 1, -1]
    # F[3] = [1, -i, -1, i]
    
    # For exact decomposition with half-integers, we use the real-valued DFT structure.
    # Actually, let's use a simpler approach: the 4x4 identity can be decomposed using
    # the Kronecker product structure.
    
    # Alternative: Use the fact that 1x4x8 multiplication can be done with rank 16
    # by treating the 8 outputs as 2x4 and using DFT on the 4-dimension.
    
    # Let's use a direct approach: for each row i (0,1,2), we need to compute
    # 8 outputs. We'll use 16 terms per row (2 groups of 4).
    
    # For row i, output k: C_k = sum_j A_{i,j} * B_{j,k}
    # We decompose this using DFT on the j-index.
    
    # Define DFT coefficients (using complex numbers, but we'll convert to real)
    # Actually, let's use a real-valued decomposition.
    
    # For 1x4x4 multiplication, the minimal rank is 4 (it's already diagonal).
    # For 1x4x8, we can do it in 8 terms (schoolbook).
    # But the directive suggests DFT might help. Let's try a different structure.
    
    # Actually, let's stick with the Strassen approach but optimize it.
    # The current Strassen gives rank 56 (28 for rows 0,1 + 32 for row 2... wait, let me recalculate)
    # Part 1: 4 k-pairs * 7 Strassen terms = 28
    # Part 2: 4 k-pairs * 7 Strassen terms = 28
    # Part 3: 4 j * 8 k = 32
    # Total: 28 + 28 + 32 = 88
    
    # Let's try a better decomposition. For the 2x4x8 part (rows 0,1),
    # we can use a 2x4x8 decomposition directly.
    
    # For 2x4x8: We can view this as 2 independent 1x4x8 problems, each of rank 8, total 16.
    # Or we can use Strassen-like methods.
    
    # Let's try: decompose 2x4x8 as 4 blocks of 2x1x2 using Strassen.
    # Each 2x1x2 block has rank 3 (Strassen for 2x2x2 gives 7 for 2x2x2, so 2x1x2 should be less).
    # Actually, 2x1x2 multiplication: C_k = sum_j A_{i,j} * B_{j,k} for i=0,1 and k=0,1.
    # This is 4 outputs, 4 inputs. Minimal rank is 4 (schoolbook) or can we do better?
    # For 2x1x2, the tensor is: T[(i,j),(j,k),(k,i)] = 1
    # This is equivalent to 2 independent 1x1x2 problems? No, it's 2x1x2.
    # Actually, 2x1x2 can be done in rank 3 using:
    # M1 = A0*B0, M2 = A1*B1, M3 = (A0+A1)*(B0+B1)
    # C00 = M1, C01 = M1, C10 = M2, C11 = M2... no that's not right.
    
    # Let me reconsider. The 2x4x8 problem:
    # For each k (0..7), we compute C_{0,k} and C_{1,k} = sum_j A_{i,j} * B_{j,k}
    # We can group k into 4 pairs: (0,1), (2,3), (4,5), (6,7)
    # For each pair (k0,k1) and each j-pair, we can use Strassen.
    
    # Actually, let's just use the schoolbook for the whole thing and see if we can improve.
    # Schoolbook: 3*4*8 = 96 terms.
    
    # Let's try a different approach: use the fact that 8 = 2*4 and 4 = 2*2.
    # Decompose as 3x4x2 * 2x2x4 (Kronecker product structure).
    
    # For 3x4x2: rank 3*4*2 = 24 (schoolbook) or better?
    # For 2x2x4: rank 2*2*4 = 16 (schoolbook) or better?
    
    # This is getting complicated. Let's just use the current Strassen approach
    # and verify it's correct. If it is, we'll keep it. If not, we'll fallback to schoolbook.
    
    # Strassen 2x2x2 rank-7 decomposition
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 1, 0]),
        ([0, 0, 0, 1], [1, 1, 0, -1], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([1, 0, -1, 0], [1, 0, 1, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 1, 0, 1], [1, 0, 0, 1]),
    ]

    # Part 1: Rows {0,1}, j in {0,1}, k in pairs -> 2x2x2 Strassen (4 k-pairs)
    for k_pair in range(4):
        k0 = 2 * k_pair
        k1 = 2 * k_pair + 1
        for (u_c, v_c, w_c) in strassen_terms:
            U_vec = [0.0] * (n*m)
            U_vec[0] = float(u_c[0])   # i=0,j=0
            U_vec[1] = float(u_c[1])   # i=0,j=1
            U_vec[4] = float(u_c[2])   # i=1,j=0
            U_vec[5] = float(u_c[3])   # i=1,j=1

            V_vec = [0.0] * (m*p)
            V_vec[k0] = float(v_c[0])       # j=0,k=k0
            V_vec[k1] = float(v_c[1])       # j=0,k=k1
            V_vec[8 + k0] = float(v_c[2])   # j=1,k=k0
            V_vec[8 + k1] = float(v_c[3])   # j=1,k=k1

            W_vec = [0.0] * (p*n)
            W_vec[3*k0] = float(w_c[0])       # k=k0,i=0
            W_vec[3*k1] = float(w_c[1])       # k=k1,i=0
            W_vec[3*k0 + 1] = float(w_c[2])   # k=k0,i=1
            W_vec[3*k1 + 1] = float(w_c[3])   # k=k1,i=1

            add_rank1(U_vec, V_vec, W_vec)

    # Part 2: Rows {0,1}, j in {2,3}, k in pairs -> 2x2x2 Strassen (4 k-pairs)
    for k_pair in range(4):
        k0 = 2 * k_pair
        k1 = 2 * k_pair + 1
        for (u_c, v_c, w_c) in strassen_terms:
            U_vec = [0.0] * (n*m)
            U_vec[2] = float(u_c[0])   # i=0,j=2
            U_vec[3] = float(u_c[1])   # i=0,j=3
            U_vec[6] = float(u_c[2])   # i=1,j=2
            U_vec[7] = float(u_c[3])   # i=1,j=3

            V_vec = [0.0] * (m*p)
            V_vec[16 + k0] = float(v_c[0])   # j=2,k=k0
            V_vec[16 + k1] = float(v_c[1])   # j=2,k=k1
            V_vec[24 + k0] = float(v_c[2])   # j=3,k=k0
            V_vec[24 + k1] = float(v_c[3])   # j=3,k=k1

            W_vec = [0.0] * (p*n)
            W_vec[3*k0] = float(w_c[0])
            W_vec[3*k1] = float(w_c[1])
            W_vec[3*k0 + 1] = float(w_c[2])
            W_vec[3*k1 + 1] = float(w_c[3])

            add_rank1(U_vec, V_vec, W_vec)

    # Part 3: Row {2}, all j in {0,1,2,3}, k in {0..7} -> schoolbook 1x4x8
    for j in range(4):
        for k in range(8):
            U_vec = [0.0] * (n*m)
            U_vec[2*4 + j] = 1.0

            V_vec = [0.0] * (m*p)
            V_vec[j*8 + k] = 1.0

            W_vec = [0.0] * (p*n)
            W_vec[k*3 + 2] = 1.0

            add_rank1(U_vec, V_vec, W_vec)

    rank = len(U_cols)

    # Convert to arrays
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T

    # Verify correctness
    error = 0.0
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                val = np.dot(U[a] * V[b], W[c])
                true_val = target.get((a, b, c), 0.0)
                error += (val - true_val)**2

    if error > 1e-6:
        # Fallback to schoolbook
        U_cols = []
        V_cols = []
        W_cols = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_vec = [0.0] * (n*m)
                    U_vec[i*m+j] = 1.0
                    V_vec = [0.0] * (m*p)
                    V_vec[j*p+k] = 1.0
                    W_vec = [0.0] * (p*n)
                    W_vec[k*n+i] = 1.0
                    add_rank1(U_vec, V_vec, W_vec)
        rank = len(U_cols)
        U = np.array(U_cols).T
        V = np.array(V_cols).T
        W = np.array(W_cols).T

    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]

    return {"U": U_list, "V": V_list, "W": W_list}
