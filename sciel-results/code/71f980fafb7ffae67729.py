import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # We decompose the problem into:
    # 1. Multiplying A (4x4) by B_L (4x4) -> C_L (4x4) using Strassen (Rank 49)
    # 2. Multiplying A (4x4) by B_R (4x3) -> C_R (4x3) using Schoolbook (Rank 48)
    # Total Rank = 49 + 48 = 97
    
    rank_square = 49
    rank_rect = 48
    total_rank = rank_square + rank_rect
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Helper to get Strassen 2x2x2 decomposition tensors
    # U, V, W are 4x7 matrices for the local 2x2x2 problem
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    def get_strassen_2x2x2():
        # 7 multiplications
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Local U (4x7), V (4x7), W (4x7)
        # A indices: 0,1,2,3 (A11=0, A12=1, A21=2, A22=3)
        # B indices: 0,1,2,3 (B11=0, B12=1, B21=2, B22=3)
        # C indices: 0,1,2,3 (C11=0, C12=1, C21=2, C22=3)
        
        u = [[0]*7 for _ in range(4)]
        v = [[0]*7 for _ in range(4)]
        w = [[0]*7 for _ in range(4)]
        
        # M1
        u[0][0] = 1; u[3][0] = 1
        v[0][0] = 1; v[3][0] = 1
        w[0][0] = 1; w[3][0] = -1; w[2][0] = 1; w[1][0] = -1 # C11=M1, C22=M1, C21=-M1, C12=-M1 ?? No.
        # Strassen Output:
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M7
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's re-fill W based on this.
        # W[c_idx][m_idx]
        
        # M1
        u[0][0] = 1; u[3][0] = 1
        v[0][0] = 1; v[3][0] = 1
        w[0][0] = 1 # C11
        w[3][0] = 1 # C22
        
        # M2
        u[2][1] = 1; u[3][1] = 1
        v[0][1] = 1
        w[2][1] = 1 # C21
        
        # M3
        u[0][2] = 1
        v[1][2] = 1; v[3][2] = -1
        w[1][2] = 1 # C12
        
        # M4
        u[3][3] = 1
        v[2][3] = 1; v[0][3] = -1
        w[0][3] = -1 # C11
        w[2][3] = 1 # C21
        
        # M5
        u[0][4] = 1; u[1][4] = 1
        v[3][4] = 1
        w[0][4] = 1 # C11
        
        # M6
        u[2][5] = 1; u[0][5] = -1
        v[0][5] = 1; v[1][5] = 1
        w[3][5] = 1 # C22
        
        # M7
        u[1][6] = 1; u[3][6] = -1
        v[2][6] = 1; v[3][6] = 1
        w[0][6] = -1 # C11
        w[1][6] = 1 # C12
        
        return u, v, w

    # 1. Fill in the 4x4x4 part (Rank 49)
    # This involves 4 blocks of 2x2x2.
    # We need to map the global indices.
    # A is 4x4. Indices 0-15.
    # B_L is 4x4 (cols 0-3). Indices 0-15.
    # C_L is 4x4 (cols 0-3). Indices 0-15 (W index is k*n+i).
    
    # Global A index: i*4 + j
    # Global B index: j*7 + k (k in 0..3 for this part)
    # Global C index: k*4 + i
    
    u_s, v_s, w_s = get_strassen_2x2x2()
    
    # The 4x4 multiplication is composed of 4 interleaved 2x2 multiplications?
    # No, Strassen on 4x4 blocks means we treat the 4x4 matrices as 2x2 blocks of 2x2 matrices.
    # So we have 7 "big" multiplications of 2x2 matrices.
    # Each 2x2 multiplication uses 7 scalar multiplications.
    # Total 49 scalar multiplications.
    
    # The structure is:
    # Let A = [[A11, A12], [A21, A22]] where Ai are 2x2.
    # Let B = [[B11, B12], [B21, B22]] where Bj are 2x2.
    # C11 = M1 - M4 + M5 - M7
    # ... where M's are 2x2 matrices resulting from multiplying 2x2 block combinations.
    
    # We can construct the 49-rank decomposition by iterating through the 7 Strassen steps
    # and for each, iterating through the 7 internal Strassen steps of the 2x2 block multiplication.
    
    # Let's define the 7 Strassen combinations for the 4x4 level.
    # S1 = A11 + A22, T1 = B11 + B22 -> M1 = S1 * T1
    # S2 = A21 + A22, T2 = B11 -> M2 = S2 * T2
    # S3 = A11, T3 = B12 - B22 -> M3 = S3 * T3
    # S4 = A22, T4 = B21 - B11 -> M4 = S4 * T4
    # S5 = A11 + A12, T5 = B22 -> M5 = S5 * T5
    # S6 = A21 - A11, T6 = B11 + B12 -> M6 = S6 * T6
    # S7 = A12 - A22, T7 = B21 + B22 -> M7 = S7 * T7
    
    # And C blocks:
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M7
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2 matrix product. We use the rank-7 decomposition for each Mi.
    # Let's map the "big" indices to "small" indices.
    # A11 corresponds to A[0:2, 0:2] -> Global U indices 0,1,4,5
    # A12 corresponds to A[0:2, 2:4] -> Global U indices 2,3,6,7
    # A21 corresponds to A[2:4, 0:2] -> Global U indices 8,9,12,13
    # A22 corresponds to A[2:4, 2:4] -> Global U indices 10,11,14,15
    
    # B11 (cols 0:2) -> Global V indices 0,1, 7,8 (j=0,1 k=0,1) -> j*7+k.
    # j=0: 0,1. j=1: 7,8.
    # B12 (cols 2:4) -> j=0: 2,3. j=1: 9,10.
    # B21 (cols 0:2) -> j=2: 14,15. j=3: 21,22.
    # B22 (cols 2:4) -> j=2: 16,17. j=3: 23,24.
    
    # C11 (cols 0:2) -> k=0: i=0..3 -> 0,1,2,3. k=1: i=0..3 -> 4,5,6,7.
    # C12 (cols 2:4) -> k=2: 8,9,10,11. k=3: 12,13,14,15.
    # C21 ... same k as C11/C12? No, C is 4x7.
    # C index in W is k*n + i.
    # For C_L (cols 0-3):
    # C11 (top-left 2x2 of C_L? No, C_L is 4x4.
    # C_L indices:
    # i=0,1, j=0,1,2,3 -> W[k*4+i]
    # Block C11 (i=0,1; k=0,1) -> W[0,1,4,5]
    # Block C12 (i=0,1; k=2,3) -> W[8,9,12,13]
    # Block C21 (i=2,3; k=0,1) -> W[2,3,6,7]
    # Block C22 (i=2,3; k=2,3) -> W[10,11,14,15]
    
    # We will iterate col = 0 to 47.
    # For each "big" Strassen step (1..7) and "small" Strassen step (1..7):
    # Determine the coefficients for U, V, W.
    
    # To keep it simple and fast, we can pre-compute the 49-rank tensors for 4x4x4.
    # But generating it on the fly is safer for correctness.
    
    # Let's define the mapping for the 4x4 Strassen.
    # For each big step b in 1..7:
    #   We have linear combinations of A-blocks and B-blocks.
    #   For each small step s in 1..7:
    #     We have linear combinations of A-elements and B-elements within those blocks.
    #     We add these to the global U, V, W.
    
    # This is complex to code dynamically. Let's use a fixed list of tuples.
    # Or, simpler: Just use the schoolbook for the 4x4x4 part? No, 49 is known.
    
    # Alternative: Use the explicit 49-rank decomposition found in literature or derive it.
    # Given time, I will construct the 49-rank by nesting the 7x7.
    
    # Let's write a small helper to add a rank-1 tensor to U, V, W.
    def add_rank1(U, V, W, col_idx, u_vec, v_vec, w_vec):
        # u_vec is list of (global_idx, coeff)
        # v_vec is list of (global_idx, coeff)
        # w_vec is list of (global_idx, coeff)
        for idx, coeff in u_vec:
            if coeff != 0:
                U[idx][col_idx] += coeff
        for idx, coeff in v_vec:
            if coeff != 0:
                V[idx][col_idx] += coeff
        for idx, coeff in w_vec:
            if coeff != 0:
                W[idx][col_idx] += coeff

    # Define the 7 Strassen steps for 2x2x2 (local)
    # u, v, w are lists of (local_idx, coeff)
    # Local A indices: 0,1,2,3
    # Local B indices: 0,1,2,3
    # Local C indices: 0,1,2,3
    
    # Step 1: (A0+A3)(B0+B3)
    u1 = [(0,1), (3,1)]
    v1 = [(0,1), (3,1)]
    w1 = [(0,1), (3,1)] # C0=C11, C3=C22
    
    # Step 2: (A2+A3)B0
    u2 = [(2,1), (3,1)]
    v2 = [(0,1)]
    w2 = [(2,1)] # C2=C21
    
    # Step 3: A0(B1-B3)
    u3 = [(0,1)]
    v3 = [(1,1), (3,-1)]
    w3 = [(1,1)] # C1=C12
    
    # Step 4: A3(B2-B0)
    u4 = [(3,1)]
    v4 = [(2,1), (0,-1)]
    w4 = [(0,-1), (2,1)] # C0=-1, C2=1
    
    # Step 5: (A0+A1)B3
    u5 = [(0,1), (1,1)]
    v5 = [(3,1)]
    w5 = [(0,1)] # C0=1
    
    # Step 6: (A2-A0)(B0+B1)
    u6 = [(2,1), (0,-1)]
    v6 = [(0,1), (1,1)]
    w6 = [(3,1)] # C3=1
    
    # Step 7: (A1-A3)(B2+B3)
    u7 = [(1,1), (3,-1)]
    v7 = [(2,1), (3,1)]
    w7 = [(0,-1), (1,1)] # C0=-1, C1=1
    
    strassen_steps = [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7)
    ]
    
    # Now map to 4x4x4.
    # Big Steps:
    # B1: S1=A11+A22, T1=B11+B22. M1=S1*T1.
    #    S1 elements: A11(0,1,4,5) + A22(10,11,14,15)
    #    T1 elements: B11(0,1,7,8) + B22(16,17,23,24)
    #    M1 is a 2x2 matrix. Its elements are computed by Strassen on the "scalar" elements of S1 and T1.
    #    This is where it gets tricky. The "scalar" elements of the block product are linear combinations of the original matrix elements.
    
    # Actually, the rank-49 decomposition of 4x4x4 is NOT simply the Kronecker product of the 7-step Strassen tensors.
    # The rank of the Kronecker product is $7 \times 7 = 49$.
    # Yes, the tensor $T_{4,4,4} = T_{2,2,2} \otimes T_{2,2,2}$.
    # And $\text{rank}(T \otimes T) \le \text{rank}(T) \text{rank}(T)$.
    # So we can just take the Kronecker product of the U, V, W matrices of the 2x2x2 Strassen decomposition!
    
    # U_4 = U_2 \otimes I_2? No.
    # If $T = \sum u_i \otimes v_i \otimes w_i$, then $T \otimes T = \sum (u_i \otimes u_j) \otimes (v_i \otimes v_j) \otimes (w_i \otimes w_j)$.
    # So we can generate the 49 tensors by iterating $i, j \in \{1..7\}$.
    
    # Global indices for Kronecker product:
    # A (4x4) index $a = a_1 \cdot 2 + a_0$.
    # U_4 index $u = u_1 \cdot 4 + u_0$? No, U is flattened.
    # U_2 has indices 0..3.
    # U_4 has indices 0..15.
    # Index $u_{4} = u_1 \cdot 2 + u_0$? No.
    # The element $A_{ij}$ in 4x4 is $A_{i_1 i_0, j_1 j_0}$.
    # The "block" index is $(i_1, j_1)$ and "element" index is $(i_0, j_0)$.
    # In the Kronecker product, the first index of $U_4$ is $u_1$ (from $U_2$) and $u_0$ (from $U_2$).
    # Wait, $U_2$ maps $A_{ij} \to$ linear combo.
    # $U_4$ maps $A_{ij} \to$ linear combo.
    # The index in $U_4$ corresponds to the pair $(i, j)$.
    # In Kronecker, index $k$ in $U_4$ corresponds to $(i, j) = (k // 2, k \% 2)$?
    # No, standard Kronecker indexing:
    # If $U_2$ has rows $r \in \{0,1,2,3\}$, $U_4$ has rows $R \in \{0..15\}$.
    # $R = r_1 \cdot 4 + r_0$?
    # Let's check the tensor definition.
    # $T[i,j,k]$.
    # $T_4[i,j,k] = T_2[i_1,j_1,k_1] T_2[i_0,j_0,k_0]$ where $i=2i_1+i_0$, etc.
    # The decomposition is $\sum_{a,b} U_{4}[i, ab] V_{4}[j, ab] W_{4}[k, ab]$.
    # $U_4[i, ab] = U_2[i_1, a] U_2[i_0, b]$.
    # Here $i$ is the row index of U. For A, $i$ is $i_A$.
    # So $U_4$ has size $16 \times 49$.
    # Index $r$ in $U_4$ corresponds to $A_{r}$.
    # $r = 2 r_1 + r_0$.
    # $U_4[r, c] = U_2[r_1, c_1] U_2[r_0, c_0]$ where $c = 7 c_1 + c_0$.
    
    # So we can construct U_4, V_4, W_4 easily.
    
    col = 0
    for c1 in range(7):
        for c0 in range(7):
            c = c1 * 7 + c0
            # We need U_2, V_2, W_2 for step c1 and c0.
            # We have strassen_steps list.
            u_s1, v_s1, w_s1 = strassen_steps[c1]
            u_s0, v_s0, w_s0 = strassen_steps[c0]
            
            # Build U_4 row for this column? No, U_4 is a matrix.
            # We just add to the global U, V, W.
            # For each r1 in 0..3, r0 in 0..3:
            #   r = 2*r1 + r0
            #   val = (coeff of r1 in u_s1) * (coeff of r0 in u_s0)
            #   U[r][col] += val
            
            # To do this efficiently, we can precompute the vector for U_4.
            u4_vec = [0]*16
            for idx1, coeff1 in u_s1:
                for idx0, coeff0 in u_s0:
                    r = 2*idx1 + idx0
                    u4_vec[r] += coeff1 * coeff0
            
            v4_vec = [0]*28 # m*p = 4*7
            for idx1, coeff1 in v_s1:
                for idx0, coeff0 in v_s0:
                    # V index for 4x7?
                    # V_2 is for 2x2. V_4 is for 4x4.
                    # But we are only filling the first 4 columns of B (B_L).
                    # So B index $j$ (0..3), $k$ (0..3).
                    # Global V index = $j \cdot 7 + k$.
                    # $j = 2 j_1 + j_0$, $k = 2 k_1 + k_0$.
                    # $j_1, k_1$ from $V_2$ step $c1$. $j_0, k_0$ from $V_2$ step $c0$.
                    # In V_2, index is $j \cdot 2 + k$.
                    # So $idx1 = 2 j_1 + k_1$. $j_1 = idx1 // 2, k_1 = idx1 \% 2$.
                    # $idx0 = 2 j_0 + k_0$. $j_0 = idx0 // 2, k_0 = idx0 \% 2$.
                    # Global V index = $(2 j_1 + j_0) \cdot 7 + (2 k_1 + k_0)$.
                    j1 = idx1 // 2; k1 = idx1 % 2
                    j0 = idx0 // 2; k0 = idx0 % 2
                    gv = (2*j1 + j0)*7 + (2*k1 + k0)
                    v4_vec[gv] += coeff1 * coeff0
            
            w4_vec = [0]*28 # p*n = 7*4
            for idx1, coeff1 in w_s1:
                for idx0, coeff0 in w_s0:
                    # W index for C_L (4x4).
                    # C index $k \cdot 4 + i$.
                    # $k = 2 k_1 + k_0$, $i = 2 i_1 + i_0$.
                    # In W_2, index is $k \cdot 2 + i$.
                    # $idx1 = 2 k_1 + i_1$. $k_1 = idx1 // 2, i_1 = idx1 \% 2$.
                    # $idx0 = 2 k_0 + i_0$. $k_0 = idx0 // 2, i_0 = idx0 \% 2$.
                    # Global W index = $(2 k_1 + k_0) \cdot 4 + (2 i_1 + i_0)$.
                    k1 = idx1 // 2; i1 = idx1 % 2
                    k0 = idx0 // 2; i0 = idx0 % 2
                    gw = (2*k1 + k0)*4 + (2*i1 + i0)
                    w4_vec[gv] += coeff1 * coeff0 # Error: should be w4_vec[gv]
                    # Wait, I used gv for W. Let's fix.
                    # gv is for V. Let's use gw.
                    
            # Correct W loop
            for idx1, coeff1 in w_s1:
                for idx0, coeff0 in w_s0:
                    k1 = idx1 // 2; i1 = idx1 % 2
                    k0 = idx0 // 2; i0 = idx0 % 2
                    gw = (2*k1 + k0)*4 + (2*i1 + i0)
                    w4_vec[gw] += coeff1 * coeff0
            
            # Add to global
            for r, val in enumerate(u4_vec):
                if val != 0:
                    U[r][col] += val
            for r, val in enumerate(v4_vec):
                if val != 0:
                    V[r][col] += val
            for r, val in enumerate(w4_vec):
                if val != 0:
                    W[r][col] += val
            
            col += 1
    
    # 2. Fill in the 4x4x3 part (Rank 48) using Schoolbook
    # A (4x4), B_R (4x3, cols 4,5,6), C_R (4x3, cols 4,5,6)
    # Schoolbook: sum over j of A[i,j] B[j,k]
    for i in range(4):
        for j in range(4):
            for k in range(3): # cols 4,5,6
                col_idx = col # 49 + offset
                # U index: i*4 + j
                # V index: j*7 + (4+k)
                # W index: (4+k)*4 + i
                U[i*4 + j][col_idx] = 1
                V[j*7 + 4 + k][col_idx] = 1
                W[(4+k)*4 + i][col_idx] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
