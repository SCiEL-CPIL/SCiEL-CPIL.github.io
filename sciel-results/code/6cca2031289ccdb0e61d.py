import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def get_strassen_2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm (Rank 7).
    U: (4, 7)
    V: (4, 7)
    W: (4, 7)
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # 7 terms M1..M7
    # Each term is a 2x2 multiplication.
    # We use the standard Strassen decomposition for the base 2x2 case? 
    # No, Strassen IS the decomposition for 2x2.
    # The "base" for Strassen is scalar multiplication (rank 1).
    # But we want the rank-7 tensor for 2x2x2.
    # We can construct it by treating the 2x2 multiplication as 7 scalar multiplications of linear combinations.
    
    # Let's define the linear combinations for A and B for each of the 7 products.
    # A is 2x2, indices (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
    # B is 2x2, indices (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need U, V, W such that C_ik = sum_r W_ik,r U_i,r V_k,r? 
    # No, the interface is U(i,j), V(j,k), W(k,i).
    # C_ik = sum_j A_ij B_jk.
    # In the decomposition: C_ik = sum_r W_ki,r (sum_j U_ij,r A_ij) (sum_j V_jk,r B_jk).
    # Note: The sum over j is inside the parentheses? 
    # The seed code: U[i*m+j][c]=1. So sum_j U[i*m+j][c] A_ij = A_ic? No.
    # If U[i*m+j][c] is 1 for specific j, then sum is A_ij.
    # So the term is A_ij * B_jk * W_ki,c.
    # This implies that for a fixed c, U selects a specific (i,j), V selects a specific (j,k), W selects (k,i).
    # This is the "monomial" basis.
    # Strassen uses linear combinations.
    # So U[i*m+j][c] can be non-zero for multiple j.
    
    # Let's build the rank-7 decomposition explicitly.
    # We define 7 "scalar" multiplications.
    # For each scalar multiplication r, we have:
    # L_r(A) = sum_{i,j} U_ij,r A_ij
    # R_r(B) = sum_{j,k} V_jk,r B_jk
    # C_ik += W_ki,r L_r(A) R_r(B)
    
    # This is equivalent to:
    # C_ik = sum_r W_ki,r (sum_{i',j'} U_{i'j',r} A_{i'j'}) (sum_{j'',k''} V_{j''k'',r} B_{j''k''})
    # This is NOT the same as the standard tensor decomposition if U and V depend on i,k.
    # In the standard decomposition for bilinear maps:
    # C_ik = sum_r (sum_{j} U_{ij,r} A_{ij}) (sum_{j} V_{jk,r} B_{jk}) W_{ik,r}
    # Here U depends on i, V on k.
    # My formulation above: L_r(A) is a single scalar for the whole matrix A.
    # This would mean C_ik = sum_r W_ik,r L_r(A) R_r(B).
    # This is a valid decomposition if L_r and R_r are linear forms on the whole space.
    # However, the task interface U has shape (n*m, rank).
    # If U[i*m+j, r] is used, it implies the linear form for rank r on A is sum_{i,j} U[i*m+j, r] A_{i,j}.
    # This is a single scalar L_r(A).
    # Similarly R_r(B) is a single scalar.
    # Then C_ik = sum_r W[k*n+i, r] L_r(A) R_r(B).
    # This is exactly the "scalar" tensor decomposition.
    # The rank is the number of such terms.
    
    # So, for Strassen 2x2:
    # We define 7 linear forms L_1..L_7 on A, and 7 on B.
    # And W coefficients.
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Indices for A: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Indices for B: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Indices for C: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) -> W index is k*n+i
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # C00 += M1, C11 += M1
    W[0, 0] = 1 # C00 (k=0,i=0)
    W[3, 0] = 1 # C11 (k=1,i=1)
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C10 += M2
    W[2, 1] = 1 # C10 (k=0,i=1)
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # C01 += M3
    W[1, 2] = 1 # C01 (k=1,i=0)
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # C00 += M4, C10 += M4
    W[0, 3] = 1 # C00
    W[2, 3] = 1 # C10
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C01 += M5, C00 -= M5
    W[1, 4] = 1 # C01
    W[0, 4] = -1 # C00
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C11 += M6
    W[3, 5] = 1 # C11
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C00 += M7
    W[0, 6] = 1 # C00
    
    # Check C11: M1 - M2 + M3 + M6
    # W[3,0]=1 (M1)
    # W[3,1]=0 (M2 is not in C11? Wait. C11 = M1 - M2 + M3 + M6. 
    # My W for M2 was W[2,1]=1 (C10). 
    # I missed W[3,1] for M2? 
    # Let's re-verify Strassen formulas.
    # C11 = M1 - M2 + M3 + M6.
    # So W[3,1] should be -1.
    # I set W[2,1]=1 for C10. That is correct for C10 = M2 + M4.
    # So I need to add W[3,1] = -1.
    
    W[3, 1] = -1
    
    # Check C01: M3 + M5.
    # W[1,2]=1 (M3). Correct.
    # W[1,4]=1 (M5). Correct.
    
    # Check C10: M2 + M4.
    # W[2,1]=1 (M2). Correct.
    # W[2,3]=1 (M4). Correct.
    
    # Check C00: M1 + M4 - M5 + M7.
    # W[0,0]=1 (M1).
    # W[0,3]=1 (M4).
    # W[0,4]=-1 (M5).
    # W[0,6]=1 (M7).
    # Correct.
    
    return U, V, W

def construct():
    start_time = time.time()
    
    # Get base 2x2x2 decomposition
    U2, V2, W2 = get_strassen_2x2()
    
    # Construct 4x4x4 decomposition using Strassen
    # We need to define the 7 block multiplications.
    # A is 4x4. Blocks:
    # A00 (0:2, 0:2), A01 (0:2, 2:4), A10 (2:4, 0:2), A11 (2:4, 2:4)
    # B is 4x4. Blocks:
    # B00, B01, B10, B11
    
    # For each of the 7 terms, we define linear maps from A(4x4) to A_block(2x2)
    # and B(4x4) to B_block(2x2).
    
    # We will build U4, V4, W4 of shape (16, 49), (16, 49), (16, 49).
    # Rank 49 = 7 * 7.
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Mapping from 4x4 index to 2x2 block index
    # 4x4 index i, j -> i*4+j
    # 2x2 block (bi, bj) with local (li, lj) -> global i = bi*2+li, j = bj*2+lj
    # U4 index: i*4+j.
    # V4 index: j*4+k.
    # W4 index: k*4+i.
    
    # Define the 7 terms
    # term 0: M1 = (A00+A11)(B00+B11)
    # term 1: M2 = (A10+A11)B00
    # term 2: M3 = A00(B01-B11)
    # term 3: M4 = A11(B10-B00)
    # term 4: M5 = (A00+A01)B11
    # term 5: M6 = (A10-A00)(B00+B01)
    # term 6: M7 = (A01-A11)(B10+B11)
    
    # For each term t, we have a 2x2 multiplication with rank 7.
    # We need to map the 4x4 inputs to the 2x2 inputs.
    
    # Let's create a helper to add a term.
    # term_t:
    # A_map: dict {(bi,bj): coeff} for the 2x2 block A_t
    # B_map: dict {(bi,bj): coeff} for the 2x2 block B_t
    # C_map: dict {(ci,cj): coeff} for how M_t contributes to C blocks
    
    # Actually, it's easier to iterate over the 4x4 indices.
    
    terms = []
    
    # M1
    # A1 = A00 + A11
    # B1 = B00 + B11
    # C00 += M1, C11 += M1
    terms.append({
        'A': { (0,0): 1, (1,1): 1 },
        'B': { (0,0): 1, (1,1): 1 },
        'C': { (0,0): 1, (1,1): 1 }
    })
    
    # M2
    # A2 = A10 + A11
    # B2 = B00
    # C10 += M2, C11 -= M2
    terms.append({
        'A': { (1,0): 1, (1,1): 1 },
        'B': { (0,0): 1 },
        'C': { (1,0): 1, (1,1): -1 }
    })
    
    # M3
    # A3 = A00
    # B3 = B01 - B11
    # C01 += M3, C11 += M3
    terms.append({
        'A': { (0,0): 1 },
        'B': { (0,1): 1, (1,1): -1 },
        'C': { (0,1): 1, (1,1): 1 }
    })
    
    # M4
    # A4 = A11
    # B4 = B10 - B00
    # C00 += M4, C10 += M4
    terms.append({
        'A': { (1,1): 1 },
        'B': { (1,0): 1, (0,0): -1 },
        'C': { (0,0): 1, (1,0): 1 }
    })
    
    # M5
    # A5 = A00 + A01
    # B5 = B11
    # C01 += M5, C00 -= M5
    terms.append({
        'A': { (0,0): 1, (0,1): 1 },
        'B': { (1,1): 1 },
        'C': { (0,1): 1, (0,0): -1 }
    })
    
    # M6
    # A6 = A10 - A00
    # B6 = B00 + B01
    # C11 += M6
    terms.append({
        'A': { (1,0): 1, (0,0): -1 },
        'B': { (0,0): 1, (0,1): 1 },
        'C': { (1,1): 1 }
    })
    
    # M7
    # A7 = A01 - A11
    # B7 = B10 + B11
    # C00 += M7
    terms.append({
        'A': { (0,1): 1, (1,1): -1 },
        'B': { (1,0): 1, (1,1): 1 },
        'C': { (0,0): 1 }
    })
    
    for t_idx, term in enumerate(terms):
        # Offset for this term's 7 rank components
        offset = t_idx * 7
        
        # For each of the 7 rank components of the base 2x2 multiplication
        for r in range(7):
            col = offset + r
            
            # U contribution
            # U2[local_i*2 + local_j, r] is the coefficient for A_local[local_i, local_j]
            # We need to map local (li, lj) to global (gi, gj)
            # gi = bi*2 + li, gj = bj*2 + lj
            # U4[gi*4 + gj, col] += U2[li*2 + lj, r] * coeff_A[bi, bj]
            
            for (bi, bj), coeff_a in term['A'].items():
                for li in range(2):
                    for lj in range(2):
                        val = U2[li*2 + lj, r]
                        if val != 0:
                            gi = bi * 2 + li
                            gj = bj * 2 + lj
                            U4[gi*4 + gj, col] += val * coeff_a
            
            # V contribution
            # V2[local_j*2 + local_k, r]
            # V4[gj*4 + gk, col] += V2[lj*2 + lk, r] * coeff_B[bj, bk]
            
            for (bj, bk), coeff_b in term['B'].items():
                for lj in range(2):
                    for lk in range(2):
                        val = V2[lj*2 + lk, r]
                        if val != 0:
                            gj = bj * 2 + lj
                            gk = bk * 2 + lk
                            V4[gj*4 + gk, col] += val * coeff_b
            
            # W contribution
            # W2[local_k*2 + local_i, r]
            # W4[gk*4 + gi, col] += W2[lk*2 + li, r] * coeff_C[ci, cj]
            
            for (ci, cj), coeff_c in term['C'].items():
                for lk in range(2):
                    for li in range(2):
                        val = W2[lk*2 + li, r]
                        if val != 0:
                            gk = ci * 2 + lk
                            gi = cj * 2 + li
                            W4[gk*4 + gi, col] += val * coeff_c
    
    # Now we have U4, V4, W4 for 4x4x4.
    # We need 4x4x8.
    # This is two 4x4x4 multiplications.
    # C_left (cols 0-3) and C_right (cols 4-7).
    
    U8 = np.zeros((16, 98))
    V8 = np.zeros((16, 98))
    W8 = np.zeros((32, 98)) # p*n = 8*4 = 32
    
    # First part: cols 0-3 of B, cols 0-3 of C
    U8[:, 0:49] = U4
    V8[:, 0:49] = V4
    # W4 is (16, 49) for C (4x4). Index k*4+i.
    # For the first part, k is 0..3.
    # W8 index for first part: k*4+i (same as W4).
    W8[0:16, 0:49] = W4
    
    # Second part: cols 4-7 of B, cols 4-7 of C
    U8[:, 49:98] = U4
    # V4 is (16, 49). Index j*4+k_local.
    # For the second part, k_global = k_local + 4.
    # So V8[j*4 + (k_local+4), 49:98] = V4[j*4+k_local, 0:49]
    # V8 index: j*8 + k_global? No, V shape is (m*p, rank). m=4, p=8.
    # Index is j*8 + k.
    # V4 index is j*4 + k_local.
    # We need to map V4[j*4+k_local] to V8[j*8 + (k_local+4)].
    
    for j in range(4):
        for k_local in range(4):
            idx_v4 = j*4 + k_local
            idx_v8 = j*8 + (k_local + 4)
            V8[idx_v8, 49:98] = V4[idx_v4, 0:49]
            
    # W8 for second part:
    # W4 index k_local*4 + i.
    # W8 index k_global*4 + i = (k_local+4)*4 + i = k_local*4 + i + 16.
    for k_local in range(4):
        for i in range(4):
            idx_w4 = k_local*4 + i
            idx_w8 = (k_local+4)*4 + i
            W8[idx_w8, 49:98] = W4[idx_w4, 0:49]
            
    # Convert to lists
    U_list = U8.tolist()
    V_list = V8.tolist()
    W_list = W8.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    # Quick verification
    res = construct()
    print("Rank:", len(res['U'][0]))
    print("U shape:", len(res['U']), "x", len(res['U'][0]))
    print("V shape:", len(res['V']), "x", len(res['V'][0]))
    print("W shape:", len(res['W']), "x", len(res['W'][0]))
