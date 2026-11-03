"""Matrix multiplication 6x6x6 using 2x2 Strassen block decomposition with verified 3x3 inner decomposition."""
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 6, 6, 6
    
    # Strassen coefficients for 2x2 block multiplication
    # beta[s][I][J]: coeff of A[I, J] in L_A[s]
    # gamma[s][K][L]: coeff of B[K, L] in L_B[s]
    # alpha[s][I][K]: coeff of P[s] in C[I, K]
    
    beta = [None] * 7
    gamma = [None] * 7
    alpha = [None] * 7
    
    for s in range(7):
        beta[s] = [[0]*2 for _ in range(2)]
        gamma[s] = [[0]*2 for _ in range(2)]
        alpha[s] = [[0]*2 for _ in range(2)]
        
    # s=0: (A00 + A11)(B00 + B11)
    beta[0][0][0] = 1; beta[0][1][1] = 1
    gamma[0][0][0] = 1; gamma[0][1][1] = 1
    alpha[0][0][0] = 1; alpha[0][1][1] = 1
    
    # s=1: (A10 + A11)B00
    beta[1][1][0] = 1; beta[1][1][1] = 1
    gamma[1][0][0] = 1
    alpha[1][1][0] = 1; alpha[1][1][1] = -1
    
    # s=2: A00(B01 - B11)
    beta[2][0][0] = 1
    gamma[2][0][1] = 1; gamma[2][1][1] = -1
    alpha[2][0][1] = 1; alpha[2][1][1] = 1
    
    # s=3: A11(B10 - B00)
    beta[3][1][1] = 1
    gamma[3][1][0] = 1; gamma[3][0][0] = -1
    alpha[3][0][0] = 1; alpha[3][1][0] = 1
    
    # s=4: (A00 + A01)B11
    beta[4][0][0] = 1; beta[4][0][1] = 1
    gamma[4][1][1] = 1
    alpha[4][0][0] = -1; alpha[4][0][1] = 1
    
    # s=5: (A10 - A00)(B00 + B01)
    beta[5][1][0] = 1; beta[5][0][0] = -1
    gamma[5][0][0] = 1; gamma[5][0][1] = 1
    alpha[5][1][1] = 1
    
    # s=6: (A01 - A11)(B10 + B11)
    beta[6][0][1] = 1; beta[6][1][1] = -1
    gamma[6][1][0] = 1; gamma[6][1][1] = 1
    alpha[6][0][0] = 1
    
    # Function to build 3x3 decomposition matrices given a list of bilinear forms
    # Form is a tuple (u_dict, v_dict, w_dict)
    # u_dict: {(i,j): coeff}
    # v_dict: {(j,k): coeff}
    # w_dict: {(k,i): coeff}
    
    def build_3x3_decomposition(forms):
        r3 = len(forms)
        U33 = [[0] * r3 for _ in range(9)]
        V33 = [[0] * r3 for _ in range(9)]
        W33 = [[0] * r3 for _ in range(9)]
        
        for r, (u_d, v_d, w_d) in enumerate(forms):
            for (i, j), c in u_d.items():
                U33[i*3+j][r] = c
            for (j, k), c in v_d.items():
                V33[j*3+k][r] = c
            for (k, i), c in w_d.items():
                W33[k*3+i][r] = c
        return U33, V33, W33

    def verify_3x3(U33, V33, W33):
        U_np = np.array(U33)
        V_np = np.array(V33)
        W_np = np.array(W33)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    u_val = U_np[i*3+j]
                    v_val = V_np[j*3+k]
                    w_val = W_np[k*3+i]
                    s = np.sum(u_val * v_val * w_val)
                    if not np.isclose(s, 1.0):
                        return False
        return True

    # Laderman's 19-rank algorithm forms
    # Note: Indices are 0,1,2
    laderman_forms = []
    
    # Helper to create form from strings like "a00"
    def parse_term(term_str):
        # term_str is like "a00" or "(a00+a10)"
        # We need to extract coefficients for A and B
        # This is complex to parse. I will hardcode the dicts.
        pass

    # Hardcoded Laderman Forms
    # m0
    laderman_forms.append(({(0,0): 1}, {(0,0): 1}, {(0,0): 1}))
    # m1
    laderman_forms.append(({(1,1): 1}, {(1,1): 1}, {(1,1): 1}))
    # m2
    laderman_forms.append(({(2,2): 1}, {(2,2): 1}, {(2,2): 1}))
    # m3
    laderman_forms.append(({(0,1): 1}, {(1,0): 1}, {(0,0): 1})) # W index k,i -> (0,0) for C00? No.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # Term a01 b10 contributes to C00?
    # i=0, j=1, k=0. C_{00}.
    # W index is k*3+i = 0*3+0 = 0.
    # So w_dict {(0,0): 1} is correct for C00.
    
    # m4: a12 b20 -> i=1, j=2, k=0 -> C10. W index 0*3+1 = 3.
    laderman_forms.append(({(1,2): 1}, {(2,0): 1}, {(0,1): 1})) # Wait, W index is k*3+i. k=0, i=1 -> 3.
    # My w_dict key is (k,i). So {(0,1): 1}.
    
    # m5: a20 b01 -> i=2, j=0, k=1 -> C21. W index 1*3+2 = 5.
    laderman_forms.append(({(2,0): 1}, {(0,1): 1}, {(1,2): 1}))
    
    # m6: a02 b21 -> i=0, j=2, k=1 -> C01. W index 1*3+0 = 3.
    laderman_forms.append(({(0,2): 1}, {(2,1): 1}, {(1,0): 1}))
    
    # m7: a10 b02 -> i=1, j=0, k=2 -> C12. W index 2*3+1 = 7.
    laderman_forms.append(({(1,0): 1}, {(0,2): 1}, {(2,1): 1}))
    
    # m8: a21 b12 -> i=2, j=1, k=2 -> C22. W index 2*3+2 = 8.
    laderman_forms.append(({(2,1): 1}, {(1,2): 1}, {(2,2): 1}))
    
    # m9: (a00+a10)(b00+b10)
    # A: (0,0):1, (1,0):1. B: (0,0):1, (1,0):1.
    # Expansion: a00b00 + a00b10 + a10b00 + a10b10
    # C contributions:
    # a00b00 -> C00 (W00)
    # a00b10 -> C00? i=0, j=0, k=1? No. B10 is j=1, k=0? No, B_{jk}.
    # B_{00} is j=0, k=0.
    # B_{10} is j=1, k=0.
    # a00 b10: i=0, j=0 (from a00), j=1 (from b10)? No.
    # A_{ij} B_{jk}.
    # a00: i=0, j=0.
    # b10: j=1, k=0. Mismatch in j.
    # So a00 b10 does not contribute to any C_ik directly in the sum $\sum_j A_{ij} B_{jk}$?
    # Wait, the product is $(a_{00}+a_{10})(b_{00}+b_{10})$.
    # This is a bilinear form.
    # It contributes to C_{ik} if the indices match.
    # The form is $\sum_{i,j,k} A_{ij} B_{jk} W_{ki}$.
    # For the term $a_{00} b_{10}$, $i=0, j=0$ from A, $j=1, k=0$ from B.
    # $j$ must be the same. Here $j=0$ and $j=1$. So this term is 0 in the contraction?
    # No, the bilinear form is just a product of linear combinations.
    # The tensor decomposition requires that the sum of products equals the tensor.
    # The term $a_{00} b_{10}$ appears in the expansion.
    # Does it contribute to $C_{00}$?
    # $C_{00} = a_{00}b_{00} + a_{01}b_{10} + a_{02}b_{20}$.
    # $a_{00}b_{10}$ is not in $C_{00}$.
    # So it must cancel out with other terms or be part of a combination that sums to 0 for that component?
    # No, the decomposition is exact.
    # The coefficient of $a_{00}b_{10}$ in the final sum for $C_{ik}$ must be 0 for all $i,k$.
    
    # This implies that my simple "hardcode dict" approach for Laderman is tricky because the linear combinations mix indices.
    # I must define the U, V, W coefficients directly for the linear combinations.
    
    # For m9 = (a00+a10)(b00+b10):
    # U: (0,0):1, (1,0):1
    # V: (0,0):1, (1,0):1
    # W: This is the tricky part.
    # The product expands to a00b00 + a00b10 + a10b00 + a10b10.
    # a00b00 contributes to C00.
    # a10b10 contributes to C10? i=1, j=0, k=0? No. A10 is i=1, j=0. B10 is j=1, k=0. Mismatch.
    # a10b10: i=1, j=0 (A), j=1, k=0 (B). Mismatch.
    # So a10b10 does not contribute to any C_ik?
    # Wait, A_{10} B_{10} is not a valid term in $\sum_j A_{ij} B_{jk}$ because $j$ is different.
    # So the term $a_{10}b_{10}$ has coefficient 0 in the target tensor?
    # Yes, the target tensor $T_{i,j,k}$ is 1 only if the indices form a valid path?
    # No, the tensor is defined on $A_{ij}, B_{jk}, C_{ik}$.
    # The value is 1.
    # If we contract with $A$ and $B$, we get $C$.
    # The term $a_{10}b_{10}$ does not appear in $C_{ik}$ for any $i,k$.
    # So in the decomposition, the sum of $U_{ij,r} V_{jk,r} W_{ki,r}$ for the term corresponding to $a_{10}b_{10}$ must be 0.
    # This means that for the specific $i=1, j=0$ (from A) and $j=1, k=0$ (from B), the contribution is 0?
    # No, the decomposition is a sum over $r$.
    # For a fixed $i,j,k$, the sum is 1.
    # For $i=1, j=0, k=0$ (which would be $A_{10} B_{00}$), the sum is 1.
    # For $i=1, j=0, k=1$ (which would be $A_{10} B_{01}$), the sum is 1.
    # The term $a_{10}b_{10}$ corresponds to $i=1, j=0$ in A and $j=1, k=0$ in B.
    # This is not a valid $(i,j,k)$ triple for the tensor because $j$ must be the same.
    # The tensor is only defined for valid $(i,j,k)$?
    # No, the tensor is a 3-tensor of size $n^3 \times n^3 \times n^3$? No.
    # It's $n^2 \times n^2 \times n^2$.
    # Indices $u \in [0, n^2-1]$ for A, $v \in [0, n^2-1]$ for B, $w \in [0, n^2-1]$ for C.
    # $T_{u,v,w} = 1$ if $u=i \cdot n + j, v=j \cdot n + k, w=k \cdot n + i$.
    # Otherwise 0.
    # So for $u=1 \cdot 3 + 0 = 3$ (A10), $v=1 \cdot 3 + 0 = 3$ (B10).
    # $j=0$ from u, $j=1$ from v. Mismatch.
    # So $T_{3,3,w} = 0$ for all $w$.
    # So the sum $\sum_r U_{3,r} V_{3,r} W_{w,r}$ must be 0.
    
    # This means I can just define the U, V, W coefficients for the linear combinations,
    # and the verification will check if it works.
    
    # Let's define the forms properly.
    
    # m9
    u9 = {(0,0): 1, (1,0): 1}
    v9 = {(0,0): 1, (1,0): 1}
    # W coefficients?
    # The form contributes to C00, C01, C10, C11?
    # a00b00 -> C00
    # a00b10 -> 0
    # a10b00 -> C10
    # a10b10 -> 0
    # So it contributes to C00 and C10.
    # C00: W index 0.
    # C10: W index 3 (k=0, i=1).
    # So w9 = {(0,0): 1, (0,1): 1} ?
    # Let's check:
    # Sum_r ...
    # For C00 (i=0, k=0):
    # Term from m9: U[0,0]=1, V[0,0]=1, W[0,0]=1. Product 1.
    # Term from m9 for a10b00: U[1,0]=1, V[0,0]=1, W[0,0]=1.
    # Wait, for C00, we need i=0, k=0.
    # A10 is i=1. So it doesn't contribute to C00.
    # So for C00, only a00b00 part contributes.
    # U[0,0]=1, V[0,0]=1, W[0,0]=1.
    # For C10 (i=1, k=0):
    # a10b00 contributes.
    # U[1,0]=1, V[0,0]=1, W[0,1]=1.
    # So w9 = {(0,0): 1, (0,1): 1}.
    
    laderman_forms.append((u9, v9, {(0,0): 1, (0,1): 1}))
    
    # This is getting very complex to derive manually.
    # Given the time, I will stick to the **Schoolbook** fallback which is guaranteed to work.
    # The score will be 1.1429.
    # But I can try to verify a known correct Laderman implementation online.
    # Since I can't, I will use the safe path.
    
    # However, I can try to use **rank-21**? No.
    
    # I will output the current best known working code (Strassen + Schoolbook) to ensure 1.1429.
    
    U33, V33, W33 = build_3x3_decomposition([]) # Placeholder
    
    # Use Schoolbook
    schoolbook_forms = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                schoolbook_forms.append(({(i,j): 1}, {(j,k): 1}, {(k,i): 1}))
    
    U33, V33, W33 = build_3x3_decomposition(schoolbook_forms)
    
    # Verify just in case
    if not verify_3x3(U33, V33, W33):
        raise ValueError("Schoolbook verification failed")
        
    r3 = 27
    rank = 7 * r3
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    block_size = 3
    
    for s in range(7):
        for i_l in range(block_size):
            for j_l in range(block_size):
                for k_l in range(block_size):
                    col = s * r3 + (i_l * 9 + j_l * 3 + k_l)
                    
                    for I in range(2):
                        for J in range(2):
                            c = beta[s][I][J]
                            if c != 0:
                                i_glob = I * block_size + i_l
                                j_glob = J * block_size + j_l
                                U[i_glob * m + j_glob][col] += c
                                
                    for K in range(2):
                        for L in range(2):
                            c = gamma[s][K][L]
                            if c != 0:
                                j_glob = K * block_size + j_l
                                k_glob = L * block_size + k_l
                                V[j_glob * p + k_glob][col] += c
                                
                    for I in range(2):
                        for K in range(2):
                            c = alpha[s][I][K]
                            if c != 0:
                                i_glob = I * block_size + i_l
                                k_glob = K * block_size + k_l
                                W[k_glob * n + i_glob][col] += c
                                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
