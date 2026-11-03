import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Global U: (n*m, Rank), V: (m*p, Rank), W: (p*n, Rank)
    U = [[0]*0 for _ in range(n*m)]
    V = [[0]*0 for _ in range(m*p)]
    W = [[0]*0 for _ in range(p*n)]
    
    def add_block(sn, sm, sp, U_sub, V_sub, W_sub, i_off, j_off, k_off):
        r = len(U_sub[0])
        # Extend global matrices
        for row in U:
            row.extend([0]*r)
        for row in V:
            row.extend([0]*r)
        for row in W:
            row.extend([0]*r)
            
        # Fill U
        # U_sub indices: i in [0,sn), j in [0,sm)
        # Global index: (i + i_off)*m + (j + j_off)
        for i in range(sn):
            for j in range(sm):
                u_idx = i*sm + j
                g_u_idx = (i + i_off)*m + (j + j_off)
                for col in range(r):
                    U[g_u_idx][len(U[g_u_idx])-r+col] = U_sub[u_idx][col]
                    
        # Fill V
        # V_sub indices: j in [0,sm), k in [0,sp)
        # Global index: (j + j_off)*p + (k + k_off)
        for j in range(sm):
            for k in range(sp):
                v_idx = j*sp + k
                g_v_idx = (j + j_off)*p + (k + k_off)
                for col in range(r):
                    V[g_v_idx][len(V[g_v_idx])-r+col] = V_sub[v_idx][col]
                    
        # Fill W
        # W_sub indices: k in [0,sp), i in [0,sn)
        # Global index: (k + k_off)*n + (i + i_off)
        for k in range(sp):
            for i in range(sn):
                w_idx = k*sn + i
                g_w_idx = (k + k_off)*n + (i + i_off)
                for col in range(r):
                    W[g_w_idx][len(W[g_w_idx])-r+col] = W_sub[w_idx][col]

    # Strassen 2x2x2 (Rank 7)
    # U (4x7), V (4x7), W (4x7)
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    U_s = [[0]*7 for _ in range(4)]
    V_s = [[0]*7 for _ in range(4)]
    W_s = [[0]*7 for _ in range(4)]
    
    # M1
    U_s[0][0] = 1; U_s[3][0] = 1
    V_s[0][0] = 1; V_s[3][0] = 1
    W_s[0][0] = 1; W_s[3][0] = 1
    
    # M2
    U_s[2][1] = 1; U_s[3][1] = 1
    V_s[0][1] = 1
    W_s[2][1] = 1; W_s[3][1] = -1
    
    # M3
    U_s[0][2] = 1
    V_s[1][2] = 1; V_s[3][2] = -1
    W_s[1][2] = 1; W_s[3][2] = 1
    
    # M4
    U_s[3][3] = 1
    V_s[2][3] = 1; V_s[0][3] = -1
    W_s[0][3] = 1; W_s[2][3] = 1
    
    # M5
    U_s[0][4] = 1; U_s[1][4] = 1
    V_s[3][4] = 1
    W_s[0][4] = -1; W_s[1][4] = 1
    
    # M6
    U_s[2][5] = 1; U_s[0][5] = -1
    V_s[0][5] = 1; V_s[1][5] = 1
    W_s[3][5] = 1
    
    # M7
    U_s[1][6] = 1; U_s[3][6] = -1
    V_s[2][6] = 1; V_s[3][6] = 1
    W_s[0][6] = 1; W_s[3][6] = -1
    
    # 2x1x2 (Rank 1)
    # C_ik = A_i B_k
    # U: 2x1, V: 2x1, W: 4x1
    U_212 = [[1], [1]]
    V_212 = [[1], [0]] # V[0] corresponds to B_00, V[1] to B_01? No, V is m*p = 1*2 = 2.
    # V index j*p+k. j=0. k=0,1.
    # We need coeff of A_i B_k to be 1.
    # Sum_r U_i V_{0k} W_k = A_i B_k.
    # If V_{00}=1, V_{01}=1? No.
    # The term is A_i * B_k.
    # In the sum, for a fixed k, the coefficient of A_i is sum_r U_i V_{0k} W_k.
    # This must be 1 for all i, k?
    # No, the tensor is T_{i, 0, k} = 1.
    # So sum_r U_i V_{0,r} W_k = 1? No, W is also indexed by r.
    # Sum_r U_i V_{0,r} W_k = 1? No.
    # The decomposition is: C_ik = sum_r U_i V_j W_k A_ij B_jk.
    # Here j=0.
    # C_ik = A_i0 B_0k.
    # So we need sum_r U_i V_0r W_k = 1?
    # Wait, the product is A_i0 * B_0k.
    # The coefficient is 1.
    # So sum_r U_i V_0r W_k = 1 is not quite right because A and B are separate.
    # The identity is: sum_r U_{i,r} V_{j,r} W_{k,r} = 1 if (i,j,k) is a valid multiplication term.
    # For 2x1x2, valid terms are (i, 0, k).
    # So sum_r U_{i,r} V_{0,r} W_{k,r} = 1.
    # Let r=0.
    # U_i = 1, V_0 = 1, W_k = 1?
    # Then sum is 1*1*1 = 1.
    # But V has 2 rows (k=0,1).
    # V is indexed by j*p+k.
    # j=0, k=0 -> V[0].
    # j=0, k=1 -> V[1].
    # In the formula, V is the factor for B.
    # B is 1x2. Entries B_00, B_01.
    # V corresponds to B.
    # So V[0] is coeff of B_00, V[1] is coeff of B_01.
    # We need coeff of A_i B_0k to be 1.
    # Term in sum: A_i B_0k U_i V_{0k} W_k.
    # This doesn't look right.
    # Standard decomposition:
    # C_ik = sum_j A_ij B_jk.
    # Sum_r U_{i,r} V_{j,r} W_{k,r} A_{i,j'} B_{j',k'}?
    # No.
    # The tensor T_{i,j,k} is 1 if i,k are indices of C and j is the inner index.
    # We want T = sum_r U^{(r)} \otimes V^{(r)} \otimes W^{(r)}.
    # T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}.
    # For 2x1x2:
    # i in 0,1. j=0. k in 0,1.
    # T_{i,0,k} = 1.
    # So sum_r U_{i,r} V_{0,r} W_{k,r} = 1.
    # Let r=0.
    # U_i = 1.
    # V_0 = 1.
    # W_k = 1.
    # Then 1*1*1 = 1.
    # V_1 (corresponding to j=1? No, j is fixed to 0 in the subproblem, but V matrix has rows for all j in the subproblem).
    # Subproblem j is 0..sm-1. Here sm=1. So only j=0.
    # V has sm*sp = 1*2 = 2 rows.
    # Row 0 corresponds to j=0, k=0? No.
    # V index is j*sp+k.
    # j=0, k=0 -> index 0.
    # j=0, k=1 -> index 1.
    # So V[0] is coeff for B_00? No, V is a matrix of shape (m*p, r).
    # It maps B to the intermediate products.
    # In the tensor identity, V_{j,r} is the coefficient for the r-th term's j-part.
    # The j-part of B is indexed by j (inner) and k (outer).
    # Wait, B is m x p.
    # In the subproblem, B is sm x sp.
    # So B_{jk}.
    # V_{jk, r} is the coefficient.
    # In the sum, we have A_{ij} V_{jk, r} W_{ki, r}?
    # No, the standard form is:
    # C_{ik} = sum_j sum_r U_{i,r} V_{j,r} W_{k,r} A_{ij} B_{jk} is wrong.
    # It is C_{ik} = sum_j A_{ij} B_{jk}.
    # Decomposition: A_{ij} -> U_{i,r} A_{?}
    # The tensor is T_{i,j,k}.
    # T = sum_r u^{(r)} v^{(r)} w^{(r)}.
    # T_{i,j,k} = u_i^{(r)} v_j^{(r)} w_k^{(r)}.
    # Here v_j is for the middle dimension.
    # But B is 2D.
    # The middle dimension of the tensor is j (size m).
    # So v is a vector of size m.
    # But V matrix is (m*p, r).
    # This implies the tensor is actually T_{i, (j,k), ?}
    # No.
    # The task says: "output factor follows the official trace convention W[k*n+i,r]".
    # U: (n*m, r). V: (m*p, r). W: (p*n, r).
    # This corresponds to flattening A (n*m), B (m*p), C (p*n).
    # So the tensor is T_{i, j, k} where i is A-index, j is B-index, k is C-index?
    # No, the multiplication is bilinear in A and B, linear in C.
    # C = A B.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # Flatten A to a, B to b, C to c.
    # c_{k*n+i} = sum_j a_{j*n+i} b_{k*p+j}.
    # So the tensor T relates a, b, c.
    # T_{a_idx, b_idx, c_idx} = 1 if c_idx corresponds to a_idx and b_idx multiplication.
    # Specifically, if a_idx = i*m+j, b_idx = j*p+k, c_idx = k*n+i.
    # Then T_{i*m+j, j*p+k, k*n+i} = 1.
    # So V is indexed by b_idx = j*p+k.
    # In the subproblem, we have a block of A (sn x sm), B (sm x sp), C (sp x sn).
    # The indices are local.
    # For 2x1x2:
    # A is 2x1. i in 0,1. j=0.
    # B is 1x2. j=0. k in 0,1.
    # C is 2x2. k in 0,1. i in 0,1.
    # Local indices:
    # a_idx = i*1+0 = i. (size 2)
    # b_idx = 0*2+k = k. (size 2)
    # c_idx = k*2+i. (size 4)
    # T_{i, k, k*2+i} = 1.
    # Decomposition: T = sum_r u_r v_r w_r.
    # u_i v_k w_{k*2+i} = 1.
    # Let r=0.
    # u_i = 1.
    # v_k = 1.
    # w_{k*2+i} = 1.
    # This works.
    # So U_212 = [[1], [1]].
    # V_212 = [[1], [1]].
    # W_212 = [[1], [1], [1], [1]].
    
    U_212 = [[1], [1]]
    V_212 = [[1], [1]]
    W_212 = [[1], [1], [1], [1]]
    
    # 1x2x2 (Rank 2)
    # A 1x2. i=0. j in 0,1.
    # B 2x2. j in 0,1. k in 0,1.
    # C 1x2. k in 0,1. i=0.
    # a_idx = j. (size 2)
    # b_idx = j*2+k. (size 4)
    # c_idx = k. (size 2)
    # T_{j, j*2+k, k} = 1.
    # We need rank 2.
    # r=0: j=0, k=0.
    # u_0=1, u_1=0.
    # v_0=1, v_1=0, v_2=0, v_3=0. (b_idx 0 is j=0,k=0)
    # w_0=1, w_1=0.
    # r=1: j=1, k=1?
    # Wait, j and k are independent in B?
    # T_{j, j*2+k, k} = 1.
    # This is not separable into rank 1 easily if j and k are linked in b_idx.
    # Let's check schoolbook rank.
    # Terms:
    # j=0, k=0: a_0, b_0, c_0.
    # j=0, k=1: a_0, b_1, c_1.
    # j=1, k=0: a_1, b_2, c_0.
    # j=1, k=1: a_1, b_3, c_1.
    # We can group by k?
    # k=0: a_j b_{2j} c_0.
    # k=1: a_j b_{2j+1} c_1.
    # This looks like rank 2 if we can factor a_j and b_{2j} or b_{2j+1}.
    # But a_j is different for j=0,1.
    # Let's try:
    # r=0: k=0 part.
    # u_j = 1.
    # v_{2j} = 1.
    # w_0 = 1.
    # This covers j=0, k=0 and j=1, k=0?
    # For j=0: u_0=1, v_0=1, w_0=1 -> 1. Correct.
    # For j=1: u_1=1, v_2=1, w_0=1 -> 1. Correct.
    # So r=0 handles k=0.
    # r=1: k=1 part.
    # u_j = 1.
    # v_{2j+1} = 1.
    # w_1 = 1.
    # For j=0: u_0=1, v_1=1, w_1=1 -> 1. Correct.
    # For j=1: u_1=1, v_3=1, w_1=1 -> 1. Correct.
    # So U_122 = [[1, 1], [1, 1]].
    # V_122 = [[1, 0], [0, 1], [1, 0], [0, 1]].
    # W_122 = [[1, 0], [0, 1]].
    
    U_122 = [[1, 1], [1, 1]]
    V_122 = [[1, 0], [0, 1], [1, 0], [0, 1]]
    W_122 = [[1, 0], [0, 1]]
    
    # 1x1x2 (Rank 1)
    # A 1x1. i=0, j=0.
    # B 1x2. j=0, k=0,1.
    # C 1x2. k=0,1. i=0.
    # a_idx = 0.
    # b_idx = k.
    # c_idx = k.
    # T_{0, k, k} = 1.
    # Rank 1.
    # u_0 = 1.
    # v_k = 1.
    # w_k = 1.
    
    U_112 = [[1]]
    V_112 = [[1], [1]]
    W_112 = [[1], [1]]
    
    # 1x2x1 (Rank 1)
    # A 1x2. i=0, j=0,1.
    # B 2x1. j=0,1. k=0.
    # C 1x1. k=0. i=0.
    # a_idx = j.
    # b_idx = j.
    # c_idx = 0.
    # T_{j, j, 0} = 1.
    # Rank 1?
    # u_j = 1.
    # v_j = 1.
    # w_0 = 1.
    # Sum u_j v_j w_0 = 1*1*1 = 1.
    # Yes.
    
    U_121 = [[1], [1]]
    V_121 = [[1], [1]]
    W_121 = [[1]]
    
    # 2x2x1 (Rank 2)
    # A 2x2. i=0,1. j=0,1.
    # B 2x1. j=0,1. k=0.
    # C 2x1. k=0. i=0,1.
    # a_idx = i*2+j.
    # b_idx = j.
    # c_idx = i.
    # T_{i*2+j, j, i} = 1.
    # Rank 2?
    # Group by i?
    # i=0: j=0,1.
    # a_0, b_0, c_0.
    # a_1, b_1, c_0.
    # i=1: j=0,1.
    # a_2, b_0, c_1.
    # a_3, b_1, c_1.
    # r=0: i=0.
    # u_0=1, u_1=1, u_2=0, u_3=0.
    # v_0=1, v_1=1.
    # w_0=1, w_1=0.
    # r=1: i=1.
    # u_0=0, u_1=0, u_2=1, u_3=1.
    # v_0=1, v_1=1.
    # w_0=0, w_1=1.
    
    U_221 = [[1, 0], [1, 0], [0, 1], [0, 1]]
    V_221 = [[1, 1], [1, 1]]
    W_221 = [[1, 0], [0, 1]]
    
    # 2x1x1 (Rank 1)
    # A 2x1. i=0,1. j=0.
    # B 1x1. j=0. k=0.
    # C 2x1. k=0. i=0,1.
    # a_idx = i.
    # b_idx = 0.
    # c_idx = i.
    # T_{i, 0, i} = 1.
    # u_i = 1.
    # v_0 = 1.
    # w_i = 1.
    
    U_211 = [[1], [1]]
    V_211 = [[1]]
    W_211 = [[1], [1]]
    
    # 1x1x1 (Rank 1)
    U_111 = [[1]]
    V_111 = [[1]]
    W_111 = [[1]]
    
    # Helper to get decomp
    def get_decomp(sn, sm, sp):
        if sn == 2 and sm == 2 and sp == 2:
            return U_s, V_s, W_s
        elif sn == 2 and sm == 1 and sp == 2:
            return U_212, V_212, W_212
        elif sn == 1 and sm == 2 and sp == 2:
            return U_122, V_122, W_122
        elif sn == 1 and sm == 1 and sp == 2:
            return U_112, V_112, W_112
        elif sn == 1 and sm == 2 and sp == 1:
            return U_121, V_121, W_121
        elif sn == 2 and sm == 2 and sp == 1:
            return U_221, V_221, W_221
        elif sn == 2 and sm == 1 and sp == 1:
            return U_211, V_211, W_211
        elif sn == 1 and sm == 1 and sp == 1:
            return U_111, V_111, W_111
        else:
            # Schoolbook
            rank = sn * sm * sp
            U = [[0]*rank for _ in range(sn*sm)]
            V = [[0]*rank for _ in range(sm*sp)]
            W = [[0]*rank for _ in range(sp*sn)]
            col = 0
            for i in range(sn):
                for j in range(sm):
                    for k in range(sp):
                        U[i*sm+j][col] = 1
                        V[j*sp+k][col] = 1
                        W[k*sn+i][col] = 1
                        col += 1
            return U, V, W

    # Blocks for 3x5x6
    # i in 0,1,2. j in 0,1,2,3,4. k in 0,1,2,3,4,5.
    
    # Row i=0,1 (2 rows)
    # Col j=0,1 (2 cols), k=0,1 (2 cols) -> 2x2x2
    add_block(2, 2, 2, *get_decomp(2, 2, 2), 0, 0, 0)
    # Col j=0,1, k=2,3 -> 2x2x2
    add_block(2, 2, 2, *get_decomp(2, 2, 2), 0, 0, 2)
    # Col j=0,1, k=4,5 -> 2x2x2
    add_block(2, 2, 2, *get_decomp(2, 2, 2), 0, 0, 4)
    
    # Col j=2,3, k=0,1 -> 2x2x2
    add_block(2, 2, 2, *get_decomp(2, 2, 2), 0, 2, 0)
    # Col j=2,3, k=2,3 -> 2x2x2
    add_block(2, 2, 2, *get_decomp(2, 2, 2), 0, 2, 2)
    # Col j=2,3, k=4,5 -> 2x2x2
    add_block(2, 2, 2, *get_decomp(2, 2, 2), 0, 2, 4)
    
    # Col j=4 (1 col), k=0,1 -> 2x1x2
    add_block(2, 1, 2, *get_decomp(2, 1, 2), 0, 4, 0)
    # Col j=4, k=2,3 -> 2x1x2
    add_block(2, 1, 2, *get_decomp(2, 1, 2), 0, 4, 2)
    # Col j=4, k=4,5 -> 2x1x2
    add_block(2, 1, 2, *get_decomp(2, 1, 2), 0, 4, 4)
    
    # Row i=2 (1 row)
    # Col j=0,1, k=0,1 -> 1x2x2
    add_block(1, 2, 2, *get_decomp(1, 2, 2), 2, 0, 0)
    # Col j=0,1, k=2,3 -> 1x2x2
    add_block(1, 2, 2, *get_decomp(1, 2, 2), 2, 0, 2)
    # Col j=0,1, k=4,5 -> 1x2x2
    add_block(1, 2, 2, *get_decomp(1, 2, 2), 2, 0, 4)
    
    # Col j=2,3, k=0,1 -> 1x2x2
    add_block(1, 2, 2, *get_decomp(1, 2, 2), 2, 2, 0)
    # Col j=2,3, k=2,3 -> 1x2x2
    add_block(1, 2, 2, *get_decomp(1, 2, 2), 2, 2, 2)
    # Col j=2,3, k=4,5 -> 1x2x2
    add_block(1, 2, 2, *get_decomp(1, 2, 2), 2, 2, 4)
    
    # Col j=4, k=0,1 -> 1x1x2
    add_block(1, 1, 2, *get_decomp(1, 1, 2), 2, 4, 0)
    # Col j=4, k=2,3 -> 1x1x2
    add_block(1, 1, 2, *get_decomp(1, 1, 2), 2, 4, 2)
    # Col j=4, k=4,5 -> 1x1x2
    add_block(1, 1, 2, *get_decomp(1, 1, 2), 2, 4, 4)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
    # Verification
    n, m, p = DIMENSIONS
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    # Construct tensor from U, V, W
    T = np.zeros((n*m, m*p, p*n))
    for r in range(len(U[0])):
        T += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    
    # Construct multiplication tensor
    T_true = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i*m + j
                b_idx = j*p + k
                c_idx = k*n + i
                T_true[a_idx, b_idx, c_idx] = 1
                
    if np.allclose(T, T_true):
        print("Verification: PASS")
    else:
        print("Verification: FAIL")
        print("Diff:", np.sum(np.abs(T - T_true)))
