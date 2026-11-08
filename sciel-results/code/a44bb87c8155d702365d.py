import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = 9 x R
    # V: (m*p, rank) = 21 x R
    # W: (p*n, rank) = 21 x R
    
    # We don't know R exactly until we count, but we can estimate or use a list of dicts
    # Or just build lists of coefficients for each row.
    
    # Let's use a list of columns. Each column is a dict or tuple of (U_idx, V_idx, W_idx, coeff)
    # Actually, U, V, W are matrices.
    # We can accumulate the coefficients for each row in each matrix.
    
    # U rows: 0..8
    # V rows: 0..20
    # W rows: 0..20
    
    # We will store the terms as a list of (u_idx, v_idx, w_idx, coeff)
    # Then we can build the matrices.
    
    terms = []
    
    # Part 1: k=2 (index 2)
    # A[i, 2] B[2, j] -> C[i, j]
    # U idx: i*3 + 2
    # V idx: 2*7 + j = 14 + j
    # W idx: j*3 + i
    for i in range(3):
        for j in range(7):
            u_idx = i * 3 + 2
            v_idx = 14 + j
            w_idx = j * 3 + i
            terms.append((u_idx, v_idx, w_idx, 1))
            
    # Part 2b: i=2, k in {0,1}
    # A[2, k] B[k, j] -> C[2, j]
    # U idx: 2*3 + k = 6 + k
    # V idx: k*7 + j
    # W idx: j*3 + 2
    for k in range(2):
        for j in range(7):
            u_idx = 6 + k
            v_idx = k * 7 + j
            w_idx = j * 3 + 2
            terms.append((u_idx, v_idx, w_idx, 1))
            
    # Part 2a-ii: i in {0,1}, k in {0,1}, j in {3,4,5,6}
    # Schoolbook
    # U idx: i*3 + k
    # V idx: k*7 + j
    # W idx: j*3 + i
    for i in range(2):
        for k in range(2):
            for j in range(3, 7):
                u_idx = i * 3 + k
                v_idx = k * 7 + j
                w_idx = j * 3 + i
                terms.append((u_idx, v_idx, w_idx, 1))
                
    # Part 2a-i: i in {0,1}, k in {0,1}, j in {0,1,2}
    # Strassen for j in {0,1}
    # A indices: 0,1,3,4
    # B indices: 0,1,7,8
    # C indices: 0,1,3,4
    
    # P1: (A00+A11)(B00+B11) -> C00, C11
    terms.append((0, 0, 0, 1)) # A00 B00 C00
    terms.append((4, 0, 0, 1)) # A11 B00 C00
    terms.append((0, 8, 0, 1)) # A00 B11 C00
    terms.append((4, 8, 0, 1)) # A11 B11 C00
    # Wait, Strassen combines them.
    # The decomposition is a sum of rank 1 tensors.
    # Strassen uses 7 multiplications.
    # Each multiplication is one term in the rank decomposition?
    # No, the rank decomposition is a sum of $R$ terms $u_r \otimes v_r \otimes w_r$.
    # Strassen gives $C = \sum_{r=1}^7 L_A(r) L_B(r) S(r)$.
    # This can be written as $\sum_{r=1}^7 \sum_{i,j} S_{ij,r} L_A(r) L_B(r) E_{ij}$.
    # This is not a sum of rank 1 tensors in the original A,B,C space directly unless we expand.
    # But we can treat each Strassen multiplication as a "virtual" variable.
    # However, the task requires U, V, W with integer/half-integer entries.
    # We can simply add the terms corresponding to the Strassen formula expanded?
    # No, that would increase rank.
    # The rank of the $2 \times 2 \times 2$ tensor is 7.
    # This means there exist 7 vectors $u_r, v_r, w_r$ such that $T = \sum u_r v_r w_r$.
    # The Strassen algorithm provides these vectors.
    # For each $r$, we have a linear form $L_A(r)$ and $L_B(r)$.
    # And a linear form $S(r)$ on C.
    # So for each $r$, we add one term to the decomposition?
    # No, $L_A(r)$ is a vector in $A$ space. $L_B(r)$ is a vector in $B$ space.
    # $S(r)$ is a vector in $C$ space.
    # The term is $L_A(r) \otimes L_B(r) \otimes S(r)$.
    # This is a rank 1 tensor.
    # So we add 7 terms.
    
    # Let's re-verify.
    # $T_{ijk} = \sum_r U_{i,r} V_{j,r} W_{k,r}$.
    # If we set $U_{i,r} = (L_A(r))_i$, $V_{j,r} = (L_B(r))_j$, $W_{k,r} = (S(r))_k$.
    # Then $\sum_r (L_A(r))_i (L_B(r))_j (S(r))_k$.
    # This matches the Strassen expansion.
    # So yes, we add 7 terms, each with coefficients from the linear forms.
    
    # Let's define the Strassen terms properly.
    # A indices: 0:00, 1:01, 3:10, 4:11
    # B indices: 0:00, 1:01, 7:10, 8:11
    # C indices: 0:00, 1:10, 3:01, 4:11 (Note: W index is j*n+i. C_{ij} -> j*3+i)
    # C00 -> 0*3+0=0
    # C01 -> 1*3+0=3
    # C10 -> 0*3+1=1
    # C11 -> 1*3+1=4
    
    # P1: A: 0,4 (1,1). B: 0,8 (1,1). C: 0,4 (1,1)
    terms.append((0, 0, 0, 1))
    terms.append((4, 0, 0, 1))
    terms.append((0, 8, 0, 1))
    terms.append((4, 8, 0, 1))
    # Wait, this adds 4 terms. But Strassen is 1 multiplication.
    # The multiplication is $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$.
    # This single product contributes to $C_{00}$ and $C_{11}$.
    # In the tensor decomposition, this corresponds to ONE term $r=1$.
    # $U[0,1]=1, U[4,1]=1$.
    # $V[0,1]=1, V[8,1]=1$.
    # $W[0,1]=1, W[4,1]=1$.
    # This single term generates 4 non-zero entries in the expanded tensor?
    # No, the term is a rank 1 tensor.
    # $T_1 = u_1 \otimes v_1 \otimes w_1$.
    # $T_1[i,j,k] = U[i,1] V[j,1] W[k,1]$.
    # So we just add ONE entry to U, V, W lists for this term.
    
    # My previous `terms.append` approach was adding individual monomials.
    # We need to add the Strassen rank-1 tensors.
    
    # Let's clear the Strassen part and redo.
    # We will define a helper to add a term.
    # But we are building a list of terms.
    # Each term is (u_dict, v_dict, w_dict)?
    # No, the final output is matrices.
    # It is easier to accumulate into U, V, W matrices directly.
    
    # Let's restart the construction logic inside the function.
    
    U = [[0.0] * 62 for _ in range(9)]
    V = [[0.0] * 62 for _ in range(21)]
    W = [[0.0] * 62 for _ in range(21)]
    
    col = 0
    
    # Part 1: 21 terms
    for i in range(3):
        for j in range(7):
            u_idx = i * 3 + 2
            v_idx = 14 + j
            w_idx = j * 3 + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    # Part 2b: 14 terms
    for k in range(2):
        for j in range(7):
            u_idx = 6 + k
            v_idx = k * 7 + j
            w_idx = j * 3 + 2
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    # Part 2a-ii: 16 terms
    for i in range(2):
        for k in range(2):
            for j in range(3, 7):
                u_idx = i * 3 + k
                v_idx = k * 7 + j
                w_idx = j * 3 + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    # Part 2a-i: Strassen (7 terms) + Schoolbook (4 terms)
    
    # Strassen Terms
    # 1. (A00+A11)(B00+B11) -> C00, C11
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[8][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # 2. (A10+A11)B00 -> C10, C11(-1)
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    # 3. A00(B01-B11) -> C01, C11
    U[0][col] = 1
    V[1][col] = 1; V[8][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # 4. A11(B10-B00) -> C00, C10
    U[4][col] = 1
    V[7][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # 5. (A00+A10)B11 -> C00(-1), C01
    U[0][col] = 1; U[3][col] = 1
    V[8][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    # 6. (A11-A00)(B00+B01) -> C11
    U[4][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # 7. (A01-A11)(B10+B11) -> C00
    U[1][col] = 1; U[4][col] = -1
    V[7][col] = 1; V[8][col] = 1
    W[0][col] = 1
    col += 1
    
    # Schoolbook for j=2 (4 terms)
    # A00 B02 -> C02
    U[0][col] = 1
    V[2][col] = 1
    W[6][col] = 1
    col += 1
    
    # A01 B12 -> C02
    U[1][col] = 1
    V[9][col] = 1
    W[6][col] = 1
    col += 1
    
    # A10 B02 -> C12
    U[3][col] = 1
    V[2][col] = 1
    W[7][col] = 1
    col += 1
    
    # A11 B12 -> C12
    U[4][col] = 1
    V[9][col] = 1
    W[7][col] = 1
    col += 1
    
    # col should be 62
    assert col == 62, f"Rank mismatch: {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
