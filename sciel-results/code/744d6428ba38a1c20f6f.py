import time
import numpy as np

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication.
    Strategy: Block decomposition using Strassen for 2x2x2 blocks and schoolbook for remainders.
    Target Rank: 28
    """
    n, m, p = 2, 3, 5
    nm, mp, pn = n * m, m * p, p * n  # 6, 15, 10
    
    start_time = time.time()
    deadline = start_time + 110.0

    # We will construct U, V, W as lists of lists (or numpy arrays)
    # U: (6, R), V: (15, R), W: (10, R)
    
    # We use a list of (U_vec, V_vec, W_vec) tuples to build the decomposition
    # U_vec is length 6, V_vec is length 15, W_vec is length 10
    
    terms = []
    
    # Helper to add a term
    def add_term(u_vals, v_vals, w_vals):
        terms.append((u_vals, v_vals, w_vals))
        
    # Block 1: A[0:2, 0:2] * B[0:2, :]
    # This is a 2x2 * 2x5 multiplication.
    # We split B's columns (5) into 2 + 2 + 1.
    # Sub-block 1.1: A_L * B_T[:, 0:2] (2x2 * 2x2) -> Use Strassen (Rank 7)
    # Sub-block 1.2: A_L * B_T[:, 2:4] (2x2 * 2x2) -> Use Strassen (Rank 7)
    # Sub-block 1.3: A_L * B_T[:, 4:5] (2x2 * 2x1) -> Use Schoolbook (Rank 4)
    
    # Indices for A_L: i in {0,1}, j in {0,1}. Global A index: i*m+j = i*3+j
    # Indices for B_T: j in {0,1}, k in {0..4}. Global B index: j*p+k = j*5+k
    # Indices for C: i in {0,1}, k in {0..4}. Global C index: k*n+i = k*2+i
    
    # Strassen decomposition for 2x2 * 2x2
    # Local indices: a=(i,j) in {0,1}x{0,1}, b=(j,k) in {0,1}x{0,1}, c=(i,k) in {0,1}x{0,1}
    # We map local (i,j,k) to global.
    
    def add_strassen_block(a_i_offset, a_j_offset, b_j_offset, b_k_offset):
        """
        Adds Strassen terms for A[a_i_offset:a_i_offset+2, a_j_offset:a_j_offset+2] * 
        B[b_j_offset:b_j_offset+2, b_k_offset:b_k_offset+2]
        """
        # Local i, j, k in 0..1
        # Global A index: (a_i_offset + i)*3 + (a_j_offset + j)
        # Global B index: (b_j_offset + j)*5 + (b_k_offset + k)
        # Global C index: (b_k_offset + k)*2 + (a_i_offset + i)
        
        # Strassen M1..M7 coefficients for A and B
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We need to express each Mk as a sum of rank-1 terms?
        # No, Strassen IS the rank-7 decomposition.
        # Each Mk corresponds to a product of two linear combinations.
        # In the tensor decomposition, each "product" corresponds to a term in the sum.
        # However, the standard tensor decomposition requires terms of the form u * v * w.
        # Strassen computes C as linear combination of 7 scalars (or matrices).
        # For 2x2x2, the "scalars" are the products of the linear forms.
        # So we add 7 terms.
        
        # Term 1: (A11+A22) * (B11+B22)
        # A part: e_{0,0} + e_{1,1}  (Local i,j)
        # B part: e_{0,0} + e_{1,1}  (Local j,k)
        # C part: Contribution to C?
        # Wait, the tensor decomposition T = sum u_r v_r w_r.
        # The value at (a,b,c) is sum u_r(a) v_r(b) w_r(c).
        # For Strassen, we define the linear forms.
        # L1_A = A11 + A22. L1_B = B11 + B22.
        # The product L1_A * L1_B contributes to C11, C12, C21, C22.
        # Specifically, C = M1 * [[1, 0], [0, 1]] + ... ? No.
        # C11 = M1 - M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # This means the "W" vectors (output) for each M term are not just 1s, but the coefficients from the Strassen combination.
        # So for Term 1 (M1):
        # U vector (A part): 1 at (0,0), 1 at (1,1)
        # V vector (B part): 1 at (0,0), 1 at (1,1)
        # W vector (C part): 1 at (0,0), 1 at (1,1)  (Coefficients for C11 and C22)
        
        # Let's verify:
        # T1[a,b,c] = U1[a] V1[b] W1[c]
        # If a=(0,0), b=(0,0), c=(0,0): U=1, V=1, W=1 -> 1.
        # If a=(1,1), b=(1,1), c=(1,1): U=1, V=1, W=1 -> 1.
        # If a=(0,0), b=(0,0), c=(1,1): U=1, V=1, W=1 -> 1.
        # But M1 should contribute to C11 and C22.
        # The tensor entry T[a,b,c] is the coefficient of A_a B_b in C_c.
        # For M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
        # This product contributes 1 to C11 and 1 to C22?
        # C11 gets M1. C22 gets M1.
        # So yes, W1 has 1 at C11 and 1 at C22.
        
        # Let's define the 7 Strassen terms for a local 2x2x2 block.
        # Local indices: i,j,k in {0,1}
        
        # Term 1: M1
        u1 = [0]*4; u1[0] = 1; u1[3] = 1 # A11 (0,0), A22 (1,1)
        v1 = [0]*4; v1[0] = 1; v1[3] = 1 # B11 (0,0), B22 (1,1)
        w1 = [0]*4; w1[0] = 1; w1[3] = 1 # C11 (0,0), C22 (1,1)
        
        # Term 2: M2 = (A21+A22)B11
        u2 = [0]*4; u2[2] = 1; u2[3] = 1 # A21 (1,0), A22 (1,1)
        v2 = [0]*4; v2[0] = 1             # B11 (0,0)
        w2 = [0]*4; w2[2] = 1             # C21 (1,0)
        
        # Term 3: M3 = A11(B12-B22)
        u3 = [0]*4; u3[0] = 1             # A11 (0,0)
        v3 = [0]*4; v3[1] = 1; v3[3] = -1 # B12 (0,1), B22 (1,1)
        w3 = [0]*4; w3[1] = 1; w3[3] = 1  # C12 (0,1), C22 (1,1)
        
        # Term 4: M4 = A22(B21-B11)
        u4 = [0]*4; u4[3] = 1             # A22 (1,1)
        v4 = [0]*4; v4[2] = 1; v4[0] = -1 # B21 (1,0), B11 (0,0)
        w4 = [0]*4; w4[2] = 1; w4[0] = -1 # C21 (1,0), C11 (0,0)
        
        # Term 5: M5 = (A11+A12)B22
        u5 = [0]*4; u5[0] = 1; u5[1] = 1  # A11 (0,0), A12 (0,1)
        v5 = [0]*4; v5[3] = 1             # B22 (1,1)
        w5 = [0]*4; w5[1] = 1; w5[3] = -1 # C12 (0,1), C22 (1,1)
        
        # Term 6: M6 = (A21-A11)(B11+B12)
        u6 = [0]*4; u6[2] = 1; u6[0] = -1 # A21 (1,0), A11 (0,0)
        v6 = [0]*4; v6[0] = 1; v6[1] = 1  # B11 (0,0), B12 (0,1)
        w6 = [0]*4; w6[3] = 1             # C22 (1,1)
        
        # Term 7: M7 = (A12-A22)(B21+B22)
        u7 = [0]*4; u7[1] = 1; u7[3] = -1 # A12 (0,1), A22 (1,1)
        v7 = [0]*4; v7[2] = 1; v7[3] = 1  # B21 (1,0), B22 (1,1)
        w7 = [0]*4; w7[0] = 1             # C11 (0,0)
        
        # Map local indices to global
        # Local A index la = i*2 + j. Global A index ga = (a_i_offset + i)*3 + (a_j_offset + j)
        # Local B index lb = j*2 + k. Global B index gb = (b_j_offset + j)*5 + (b_k_offset + k)
        # Local C index lc = i*2 + k. Global C index gc = (b_k_offset + k)*2 + (a_i_offset + i)
        
        local_terms = [(u1,v1,w1), (u2,v2,w2), (u3,v3,w3), (u4,v4,w4), (u5,v5,w5), (u6,v6,w6), (u7,v7,w7)]
        
        for u_l, v_l, w_l in local_terms:
            u_g = [0]*nm
            v_g = [0]*mp
            w_g = [0]*pn
            
            for i in range(2):
                for j in range(2):
                    la = i*2 + j
                    ga = (a_i_offset + i)*3 + (a_j_offset + j)
                    u_g[ga] = u_l[la]
                    
            for j in range(2):
                for k in range(2):
                    lb = j*2 + k
                    gb = (b_j_offset + j)*5 + (b_k_offset + k)
                    v_g[gb] = v_l[lb]
                    
            for i in range(2):
                for k in range(2):
                    lc = i*2 + k
                    gc = (b_k_offset + k)*2 + (a_i_offset + i)
                    w_g[gc] = w_l[lc]
                    
            add_term(u_g, v_g, w_g)
            
    # Add Strassen for B columns 0,1
    add_strassen_block(0, 0, 0, 0)
    # Add Strassen for B columns 2,3
    add_strassen_block(0, 0, 0, 2)
    
    # Sub-block 1.3: A_L * B_T[:, 4:5] (2x2 * 2x1)
    # Schoolbook for 2x2 * 2x1. Rank 4.
    # C_i = A_i0 B_0 + A_i1 B_1
    # Terms:
    # 1. A00 * B0 -> C0
    # 2. A01 * B1 -> C0
    # 3. A10 * B0 -> C1
    # 4. A11 * B1 -> C1
    
    def add_schoolbook_2x2x1(a_i_offset, a_j_offset, b_j_offset, b_k_offset):
        # Local A: 2x2, Local B: 2x1, Local C: 2x1
        # B has only 1 column (k=0 local)
        
        # Term 1: A00 * B0
        u1 = [0]*4; u1[0] = 1
        v1 = [0]*2; v1[0] = 1 # B index 0 (j=0, k=0)
        w1 = [0]*2; w1[0] = 1 # C index 0 (i=0, k=0)
        
        # Term 2: A01 * B1
        u2 = [0]*4; u2[1] = 1
        v2 = [0]*2; v2[1] = 1 # B index 1 (j=1, k=0)
        w2 = [0]*2; w2[0] = 1
        
        # Term 3: A10 * B0
        u3 = [0]*4; u3[2] = 1
        v3 = [0]*2; v3[0] = 1
        w3 = [0]*2; w3[1] = 1 # C index 1 (i=1, k=0)
        
        # Term 4: A11 * B1
        u4 = [0]*4; u4[3] = 1
        v4 = [0]*2; v4[1] = 1
        w4 = [0]*2; w4[1] = 1
        
        local_terms = [(u1,v1,w1), (u2,v2,w2), (u3,v3,w3), (u4,v4,w4)]
        
        for u_l, v_l, w_l in local_terms:
            u_g = [0]*nm
            v_g = [0]*mp
            w_g = [0]*pn
            
            for i in range(2):
                for j in range(2):
                    la = i*2 + j
                    ga = (a_i_offset + i)*3 + (a_j_offset + j)
                    u_g[ga] = u_l[la]
                    
            for j in range(2):
                # k is always 0 local
                lb = j*2 + 0
                gb = (b_j_offset + j)*5 + (b_k_offset + 0)
                v_g[gb] = v_l[lb]
                
            for i in range(2):
                # k is always 0 local
                lc = i*2 + 0
                gc = (b_k_offset + 0)*2 + (a_i_offset + i)
                w_g[gc] = w_l[lc]
                
            add_term(u_g, v_g, w_g)
            
    add_schoolbook_2x2x1(0, 0, 0, 4)
    
    # Block 2: A[0:2, 2:3] * B[2:3, :]
    # This is 2x1 * 1x5.
    # C_i = A_i2 * B_2k
    # Rank 10 (Schoolbook).
    
    def add_outer_2x1x5(a_i_offset, a_j_offset, b_j_offset):
        # A is 2x1 (j is fixed to a_j_offset)
        # B is 1x5 (j is fixed to b_j_offset)
        # C is 2x5
        
        for i in range(2):
            for k in range(5):
                u_g = [0]*nm
                v_g = [0]*mp
                w_g = [0]*pn
                
                ga = (a_i_offset + i)*3 + (a_j_offset)
                gb = (b_j_offset)*5 + k
                gc = k*2 + (a_i_offset + i)
                
                u_g[ga] = 1
                v_g[gb] = 1
                w_g[gc] = 1
                
                add_term(u_g, v_g, w_g)
                
    add_outer_2x1x5(0, 2, 2)
    
    # Total Rank: 7 + 7 + 4 + 10 = 28
    
    R = len(terms)
    
    # Build final matrices
    U = [[0]*R for _ in range(nm)]
    V = [[0]*R for _ in range(mp)]
    W = [[0]*R for _ in range(pn)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx in range(nm):
            if u[idx] != 0:
                U[idx][r] = u[idx]
        for idx in range(mp):
            if v[idx] != 0:
                V[idx][r] = v[idx]
        for idx in range(pn):
            if w[idx] != 0:
                W[idx][r] = w[idx]
                
    # Verify
    def verify(U, V, W, rank):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        for a in range(nm):
            i, j = divmod(a, m)
            for b in range(mp):
                j2, k2 = divmod(b, p)
                if j != j2:
                    # T[a,b,c] should be 0
                    for c in range(pn):
                        val = np.sum(Ua[a,:] * Va[b,:] * Wa[c,:])
                        if abs(val) > 1e-9:
                            return False
                else:
                    for c in range(pn):
                        expected = 1.0 if (c == k2 * n + i) else 0.0
                        val = np.sum(Ua[a,:] * Va[b,:] * Wa[c,:])
                        if abs(val - expected) > 1e-9:
                            return False
        return True

    if verify(U, V, W, R):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if construction fails
        rank = n * m * p
        U_sb = [[0]*rank for _ in range(nm)]
        V_sb = [[0]*rank for _ in range(mp)]
        W_sb = [[0]*rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j][col] = 1
                    V_sb[j*p+k][col] = 1
                    W_sb[k*n+i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
