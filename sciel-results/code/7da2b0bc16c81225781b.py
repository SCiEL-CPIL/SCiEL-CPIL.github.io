import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # A is 4x5, B is 5x5, C is 4x5
    # We split A into A1 (4x4) and A2 (4x1)
    # We split B into B1 (5x4) and B2 (5x1)
    # C = A1 B1 + A2 B1 + A1 B2 + A2 B2
    
    # Part 1: A1 B1 (4x4 * 4x4) Rank 49
    # Part 2: A2 B1 (4x1 * 1x4) Rank 16
    # Part 3: A1 B2 (4x4 * 4x1) Rank 16
    # Part 4: A2 B2 (4x1 * 1x1) Rank 4
    # Total Rank 85
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff_u=1, coeff_v=1, coeff_w=1):
        U_rows.append([coeff_u if u_idx == r else 0 for r in range(n*m)])
        V_rows.append([coeff_v if v_idx == r else 0 for r in range(m*p)])
        W_rows.append([coeff_w if w_idx == r else 0 for r in range(p*n)])
        
    # Wait, U, V, W are matrices of shape (n*m, rank), (m*p, rank), (p*n, rank)
    # It is more efficient to build them as lists of columns (terms) and then transpose/stack.
    # Or just append to lists of length rank.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_idx, v_idx, w_idx, coeff_u=1, coeff_v=1, coeff_w=1):
        u_vec = [0.0] * (n*m)
        u_vec[u_idx] = coeff_u
        v_vec = [0.0] * (m*p)
        v_vec[v_idx] = coeff_v
        w_vec = [0.0] * (p*n)
        w_vec[w_idx] = coeff_w
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # --- Part 1: A1 B1 (4x4 * 4x4) ---
    # A1 indices: i in 0..3, j in 0..3. Index in A: i*5 + j
    # B1 indices: j in 0..3, k in 0..3. Index in B: j*5 + k
    # C1 indices: i in 0..3, k in 0..3. Index in C: k*4 + i
    
    # Strassen's algorithm for 4x4
    # Split A1 into 2x2 blocks of 2x2 matrices
    # A11 (0,0)-(1,1), A12 (0,2)-(1,3), A21 (2,0)-(3,1), A22 (2,2)-(3,3)
    
    # We need to generate the 7 Strassen multiplications for each of the 7 blocks.
    # Actually, Strassen's algorithm computes C = A B where A, B are 2x2 blocks of 2x2 matrices.
    # The 7 multiplications are M1..M7.
    # Each Mr is a 2x2 matrix product P_r Q_r.
    # P_r and Q_r are 2x2 matrices with entries that are linear combinations of A and B blocks.
    
    # Let's define the linear forms for P_r and Q_r entries.
    # P_r is 2x2, Q_r is 2x2.
    # P_r[0,0] is a linear form in A11, A12, etc? No, P_r is a matrix of linear forms.
    # Actually, P_r is a 2x2 matrix where each entry is a linear combination of 4 entries of A.
    # Let's list the 7 Strassen multiplications.
    
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # For each Mr, we perform a 2x2 Strassen multiplication.
    # This requires 7 scalar multiplications.
    # We need to map the entries of P_r and Q_r to the original A and B indices.
    
    # Helper to get A index
    def a_idx(i, j): return i * 5 + j
    def b_idx(j, k): return j * 5 + k
    def c_idx(i, k): return k * 4 + i
    
    # Define the 7 block multiplications
    # Each block multiplication is defined by P_r and Q_r.
    # P_r is 2x2, Q_r is 2x2.
    # We represent P_r and Q_r as 2x2 arrays of lists of (index, coeff) pairs.
    
    # A11: i in 0..1, j in 0..1
    # A12: i in 0..1, j in 2..3
    # A21: i in 2..3, j in 0..1
    # A22: i in 2..3, j in 2..3
    
    # B11: j in 0..1, k in 0..1
    # B12: j in 0..1, k in 2..3
    # B21: j in 2..3, k in 0..1
    # B22: j in 2..3, k in 2..3
    
    # This is getting very detailed. Given the time constraint, I will implement the 85 rank decomposition using the simpler parts and a placeholder for the 49 rank part if I can't derive it fully, but I should try to derive it.
    # Actually, I can just use the 85 rank decomposition with the 49 rank part implemented as Strassen recursion.
    
    # Let's implement the 49 rank part properly.
    
    # Strassen 2x2 multiplication for matrices P and Q.
    # P, Q are 2x2 matrices.
    # We return 7 terms (u_idx, v_idx, w_idx) for the product C = P Q.
    # But here P and Q are not scalar matrices, they are 2x2 matrices of linear forms.
    # So each "scalar" multiplication in Strassen's algorithm is actually a multiplication of two linear forms.
    # Wait, no. Strassen's algorithm for 2x2 matrices uses 7 scalar multiplications.
    # If we apply it to 2x2 matrices of linear forms, each "scalar" is a linear form.
    # So we get 7 products of linear forms.
    # Each product of linear forms is a bilinear form.
    # We can decompose each bilinear form into rank 1 terms?
    # No, a product of two linear forms $\sum a_i x_i \sum b_j y_j$ is a sum of rank 1 terms $a_i b_j x_i y_j$.
    # So each of the 7 Strassen multiplications expands into multiple rank 1 terms.
    # This is not efficient.
    
    # The rank 49 algorithm for 4x4 is NOT simply Strassen's algorithm applied to 2x2 blocks where each block multiplication is rank 7.
    # It IS that, but the "scalar" multiplications in the outer Strassen are the 2x2 matrix multiplications.
    # So we have 7 outer multiplications, each of which is a 2x2 matrix multiplication.
    # Each 2x2 matrix multiplication has rank 7.
    # So total rank 49.
    # The 7 inner multiplications for each outer multiplication are scalar multiplications of linear forms.
    # So we just need to generate the 49 scalar multiplications.
    
    # Let's define the 7 outer multiplications M1..M7.
    # For each Mr, define P_r and Q_r.
    # Then for each Mr, apply Strassen's 7 formulas to P_r and Q_r.
    # This gives 7 scalar multiplications.
    # Each scalar multiplication is of the form (linear form in A) * (linear form in B).
    # We add these to U, V, W.
    
    # This is feasible.
    
    # I will implement this in the code.
    
    pass # Placeholder for the actual implementation in the final code block
