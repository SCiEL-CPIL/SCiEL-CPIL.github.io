import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    n, m, p = DIMENSIONS
    # Initial schoolbook decomposition
    rank0 = n * m * p
    U = np.zeros((n * m, rank0))
    V = np.zeros((m * p, rank0))
    W = np.zeros((p * n, rank0))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1
                
    # Verify initial (should always be true)
    # T = sum U[:,r] V[:,r]^T W[:,r]^T
    # We check if T can be represented with fewer terms.
    
    # Strategy:
    # 1. Try to find a decomposition of rank R < rank0.
    # 2. We can do this by checking if the tensor T is in the variety of tensors of rank <= R.
    #    A simpler heuristic for small reductions:
    #    Check if any term is linearly dependent in a specific way.
    #    
    #    Actually, a more robust "structural" change is to use the **Border Rank** or **Approximate Rank** 
    #    and then snap to integers? No, must be exact.
    
    #    Let's try a greedy elimination approach.
    #    If we remove term r, can the rest express T?
    #    T - sum_{c != r} U_c V_c^T W_c^T = 0?
    #    This is equivalent to checking if U_r V_r^T W_r^T is in the span of {U_c V_c^T W_c^T | c != r}.
    #    This is hard to check directly.
    
    #    Alternative: Use the fact that for small matrices, we can try to find a decomposition 
    #    with rank = n*m*p - 1, etc., by solving a system of linear equations for the coefficients 
    #    of a fixed set of basis tensors?
    
    #    Let's try a different angle: **Strassen-style block decomposition**.
    #    For 3x4 * 4x6, can we use a block structure?
    #    3x4 can be seen as blocks of 2x2? No, 3 is prime.
    #    4x6 can be seen as 2x3 blocks?
    
    #    Let's look at the known bounds.
    #    R(3,4,6) <= ?
    #    Schoolbook: 72.
    #    If we split 4 into 2+2:
    #    A (3x4) = [A1 A2] (3x2, 3x2)
    #    B (4x6) = [B1; B2] (2x6, 2x6)
    #    AB = A1B1 + A2B2.
    #    Cost: 2 * R(3,2,6).
    #    R(3,2,6) <= 3*2*6 = 36. Total 72. No gain.
    
    #    Split 6 into 3+3:
    #    A (3x4), B (4x6) = [B1 B2] (4x3, 4x3)
    #    AB = [A B1, A B2].
    #    Cost: 2 * R(3,4,3).
    #    R(3,4,3) <= 36. Total 72.
    
    #    Split 3 into 1+2:
    #    A (3x4) = [A1; A2] (1x4, 2x4)
    #    B (4x6)
    #    AB = [A1B; A2B].
    #    Cost: R(1,4,6) + R(2,4,6).
    #    R(1,4,6) = 6.
    #    R(2,4,6) <= 2*4*6 = 48. Total 54.
    #    Wait, R(2,4,6) can be better?
    #    Split 4 into 2+2:
    #    A2 (2x4) = [A21 A22] (2x2, 2x2)
    #    B (4x6) = [B1; B2] (2x6, 2x6)
    #    A2B = A21B1 + A22B2.
    #    Cost: 2 * R(2,2,6).
    #    R(2,2,6) <= 8. Total 16.
    #    So R(2,4,6) <= 16.
    #    Total R(3,4,6) <= 6 + 16 = 22.
    
    #    Let's verify this decomposition.
    #    1. Compute C1 = A1 B. A1 is 1x4, B is 4x6. C1 is 1x6.
    #       This requires 6 scalar multiplications.
    #       U1, V1, W1 for this part.
    #       U1 has 1 non-zero per term?
    #       Let's construct it explicitly.
        
    #    2. Compute C2 = A2 B. A2 is 2x4, B is 4x6. C2 is 2x6.
    #       Split A2 into A21 (2x2) and A22 (2x2).
    #       Split B into B1 (2x6) and B2 (2x6).
    #       C2 = A21 B1 + A22 B2.
    #       D1 = A21 B1 (2x2 * 2x6 -> 2x6).
    #       D2 = A22 B2 (2x2 * 2x6 -> 2x6).
    #       R(2,2,6) schoolbook is 2*2*6 = 24? No.
    #       R(2,2,6) is min rank for 2x2 * 2x6.
    #       2x2 * 2x6.
    #       Schoolbook: 2*2*6 = 24.
    #       Can we do better?
    #       2x2 * 2x6 = (2x2) * (2x2 * 2x3)?
    #       Let's just use schoolbook for the sub-blocks for now to ensure correctness, 
    #       then optimize if time permits.
    #       Actually, R(2,2,6) is likely 12? 
    #       (2x2) * (2x6). 
    #       Let A be 2x2, B be 2x6.
    #       AB is 2x6.
    #       We can compute 2x2 * 2x2 (4 terms) and 2x2 * 2x2 (4 terms) for the two 2x3 blocks of B?
    #       No, B is 2x6.
    #       Let B = [B_a B_b] where B_a, B_b are 2x3.
    #       AB = [A B_a, A B_b].
    #       Cost: 2 * R(2,2,3).
    #       R(2,2,3) <= 12.
    #       So R(2,2,6) <= 24.
    #       This matches schoolbook.
        
    #    So the bound 22 is:
    #    R(3,4,6) <= R(1,4,6) + R(2,4,6)
    #    R(1,4,6) = 6.
    #    R(2,4,6) <= R(2,2,6) + R(2,2,6) = 24 + 24 = 48? 
    #    Wait, my previous calc:
    #    A2 (2x4) = [A21 A22] (2x2, 2x2).
    #    B (4x6) = [B1; B2] (2x6, 2x6).
    #    A2B = A21 B1 + A22 B2.
    #    R(2,2,6) is the cost for 2x2 * 2x6.
    #    Schoolbook R(2,2,6) = 2*2*6 = 24.
    #    So R(2,4,6) <= 24 + 24 = 48.
    #    Total 6 + 48 = 54.
    
    #    Can we do better on R(2,2,6)?
    #    2x2 * 2x6.
    #    Let A = [[a,b],[c,d]], B = [[e,f,g,h,i,j],[k,l,m,n,o,p]].
    #    AB = [[ae+bk, af+bl, ag+bm, ah+bn, ai+bo, aj+bp],
    #         [[ce+dk, cf+dl, cg+dm, ch+dn, ci+do, cj+dp]].
    #    This is 12 outputs.
    #    Each output is a sum of 2 products.
    #    Total 24 products.
    #    Can we share?
    #    e.g. (a+b)(e+k) = ae + ak + be + bk.
    #    We have ae+bk. We don't have ak or be.
    #    It is known that R(2,2,2)=7 (Strassen).
    #    R(2,2,3)?
    #    If we use Strassen for 2x2 * 2x2, cost 7.
    #    B is 2x6 = 3 blocks of 2x2.
    #    Cost 3 * 7 = 21.
    #    So R(2,2,6) <= 21.
    #    Then R(2,4,6) <= 21 + 21 = 42.
    #    Total R(3,4,6) <= 6 + 42 = 48.
    
    #    Let's implement the 48-rank decomposition.
    #    Structure:
    #    Part 1: A1 (1x4) * B (4x6). Rank 6.
    #    Part 2: A21 (2x2) * B1 (2x6). Rank 21 (3 Strassen blocks).
    #    Part 3: A22 (2x2) * B2 (2x6). Rank 21 (3 Strassen blocks).
    #    Total Rank 48.
    
    #    We need to map this to the global indices.
    #    Global U shape: (12, R)
    #    Global V shape: (24, R)
    #    Global W shape: (18, R)
    
    #    Indices:
    #    A (3x4): rows 0,1,2. cols 0,1,2,3.
    #    B (4x6): rows 0,1,2,3. cols 0,1,2,3,4,5.
    #    C (3x6): rows 0,1,2. cols 0,1,2,3,4,5.
    
    #    Split A:
    #    A1 = row 0 of A. (1x4). Indices in A: (0, 0..3).
    #    A2 = rows 1,2 of A. (2x4). Indices in A: (1, 0..3), (2, 0..3).
    #    A21 = cols 0,1 of A2. (2x2). Indices in A: (1,0), (1,1), (2,0), (2,1).
    #    A22 = cols 2,3 of A2. (2x2). Indices in A: (1,2), (1,3), (2,2), (2,3).
    
    #    Split B:
    #    B is used in Part 1 fully.
    #    For Part 2, we split B into B1 (rows 0,1) and B2 (rows 2,3).
    #    B1 is 2x6.
    #    B2 is 2x6.
    
    #    Part 1: A1 * B
    #    A1 is 1x4. B is 4x6.
    #    Term (j, k) where j in 0..3, k in 0..5.
    #    U index: i=0, j=j. -> 0*m + j = j.
    #    V index: j=j, k=k. -> j*p + k = j*6 + k.
    #    W index: k=k, i=0. -> k*n + 0 = k*3.
    #    6 terms.
    
    #    Part 2: A21 * B1
    #    A21 is 2x2. Rows 0,1 (local). Cols 0,1 (local).
    #    Global A rows: 1,2. Global A cols: 0,1.
    #    B1 is 2x6. Rows 0,1 (local). Cols 0..5 (local).
    #    Global B rows: 0,1. Global B cols: 0..5.
    #    Output is 2x6. Global C rows: 1,2. Global C cols: 0..5.
    #    
    #    We use 3 Strassen multiplications for 2x2 * 2x2 blocks of B1.
    #    B1 cols 0,1 -> Block 0.
    #    B1 cols 2,3 -> Block 1.
    #    B1 cols 4,5 -> Block 2.
    #    
    #    Strassen for 2x2 * 2x2:
    #    M1 = (A00 + A11)(B00 + B11)
    #    M2 = (A10 + A11)B00
    #    M3 = A00(B01 - B11)
    #    M4 = A11(B10 - B00)
    #    M5 = (A00 + A01)B11
    #    M6 = (A10 - A00)(B00 + B01)
    #    M7 = (A01 - A11)(B10 + B11)
    #    
    #    C00 = M1 + M4 - M5 + M7
    #    C01 = M3 + M5
    #    C10 = M2 + M4
    #    C11 = M1 - M2 + M3 + M6
    
    #    Let's generate the terms for one 2x2 * 2x2 Strassen block.
    #    Local A indices: (0,0), (0,1), (1,0), (1,1).
    #    Local B indices: (0,0), (0,1), (1,0), (1,1).
    #    Local C indices: (0,0), (0,1), (1,0), (1,1).
    
    #    We will write a helper to add terms.
    
    U = []
    V = []
    W = []
    
    def add_term(u_idx, v_idx, w_idx, coeff_u, coeff_v, coeff_w):
        # u_idx, v_idx, w_idx are global indices
        # coeff are multipliers for the basis vectors
        # Since we are building sparse matrices, we just append to lists?
        # No, we need fixed rank.
        # We know total rank is 6 + 21 + 21 = 48.
        pass
        
    # Let's build the matrices directly.
    Rank = 48
    U_mat = np.zeros((12, Rank))
    V_mat = np.zeros((24, Rank))
    W_mat = np.zeros((18, Rank))
    
    col = 0
    
    # Part 1: A1 * B
    # A1 is row 0 of A.
    # B is full 4x6.
    for j in range(4):
        for k in range(6):
            # U: A[0, j]
            U_mat[0 * 4 + j, col] = 1
            # V: B[j, k]
            V_mat[j * 6 + k, col] = 1
            # W: C[k, 0]  (since i=0)
            W_mat[k * 3 + 0, col] = 1
            col += 1
            
    # Part 2: A21 * B1
    # A21: A[1:3, 0:2]
    # B1: B[0:2, 0:6]
    # C: C[1:3, 0:6]
    
    # Helper for Strassen 2x2 * 2x2
    # a_idx: global row indices for A (2 rows)
    # a_col: global col indices for A (2 cols)
    # b_idx: global row indices for B (2 rows)
    # b_col: global col indices for B (2 cols)
    # c_idx: global row indices for C (2 rows)
    # c_col: global col indices for C (2 cols)
    
    def add_strassen(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        nonlocal col
        # Local 00, 01, 10, 11
        a00 = a_rows[0] * 4 + a_cols[0]
        a01 = a_rows[0] * 4 + a_cols[1]
        a10 = a_rows[1] * 4 + a_cols[0]
        a11 = a_rows[1] * 4 + a_cols[1]
        
        b00 = b_rows[0] * 6 + b_cols[0]
        b01 = b_rows[0] * 6 + b_cols[1]
        b10 = b_rows[1] * 6 + b_cols[0]
        b11 = b_rows[1] * 6 + b_cols[1]
        
        c00 = c_rows[0] * 6 + c_cols[0]
        c01 = c_rows[0] * 6 + c_cols[1]
        c10 = c_rows[1] * 6 + c_cols[0]
        c11 = c_rows[1] * 6 + c_cols[1]
        
        # M1 = (A00 + A11)(B00 + B11)
        U_mat[a00, col] = 1
        U_mat[a11, col] = 1
        V_mat[b00, col] = 1
        V_mat[b11, col] = 1
        W_mat[c00, col] = 1
        W_mat[c11, col] = 1
        col += 1
        
        # M2 = (A10 + A11)B00
        U_mat[a10, col] = 1
        U_mat[a11, col] = 1
        V_mat[b00, col] = 1
        W_mat[c10, col] = 1
        col += 1
        
        # M3 = A00(B01 - B11)
        U_mat[a00, col] = 1
        V_mat[b01, col] = 1
        V_mat[b11, col] = -1
        W_mat[c01, col] = 1
        W_mat[c11, col] = 1
        col += 1
        
        # M4 = A11(B10 - B00)
        U_mat[a11, col] = 1
        V_mat[b10, col] = 1
        V_mat[b00, col] = -1
        W_mat[c00, col] = 1
        W_mat[c10, col] = 1
        col += 1
        
        # M5 = (A00 + A01)B11
        U_mat[a00, col] = 1
        U_mat[a01, col] = 1
        V_mat[b11, col] = 1
        W_mat[c00, col] = -1
        W_mat[c01, col] = 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U_mat[a10, col] = 1
        U_mat[a00, col] = -1
        V_mat[b00, col] = 1
        V_mat[b01, col] = 1
        W_mat[c11, col] = 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U_mat[a01, col] = 1
        U_mat[a11, col] = -1
        V_mat[b10, col] = 1
        V_mat[b11, col] = 1
        W_mat[c00, col] = 1
        W_mat[c11, col] = -1
        col += 1
        
    # Apply Strassen for B1 blocks
    # B1 is B[0:2, 0:6]
    # Block 0: Cols 0,1
    add_strassen(
        a_rows=[1, 2], a_cols=[0, 1],
        b_rows=[0, 1], b_cols=[0, 1],
        c_rows=[1, 2], c_cols=[0, 1]
    )
    # Block 1: Cols 2,3
    add_strassen(
        a_rows=[1, 2], a_cols=[0, 1],
        b_rows=[0, 1], b_cols=[2, 3],
        c_rows=[1, 2], c_cols=[2, 3]
    )
    # Block 2: Cols 4,5
    add_strassen(
        a_rows=[1, 2], a_cols=[0, 1],
        b_rows=[0, 1], b_cols=[4, 5],
        c_rows=[1, 2], c_cols=[4, 5]
    )
    
    # Part 3: A22 * B2
    # A22: A[1:3, 2:4]
    # B2: B[2:4, 0:6]
    # C: C[1:3, 0:6]
    
    # Block 0: Cols 0,1
    add_strassen(
        a_rows=[1, 2], a_cols=[2, 3],
        b_rows=[2, 3], b_cols=[0, 1],
        c_rows=[1, 2], c_cols=[0, 1]
    )
    # Block 1: Cols 2,3
    add_strassen(
        a_rows=[1, 2], a_cols=[2, 3],
        b_rows=[2, 3], b_cols=[2, 3],
        c_rows=[1, 2], c_cols=[2, 3]
    )
    # Block 2: Cols 4,5
    add_strassen(
        a_rows=[1, 2], a_cols=[2, 3],
        b_rows=[2, 3], b_cols=[4, 5],
        c_rows=[1, 2], c_cols=[4, 5]
    )
    
    # Check if we hit the deadline
    if time.time() > deadline:
        # Return current best (which is 48)
        return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
        
    # Verify correctness quickly
    # T = sum U_c V_c^T W_c^T
    # We can check a few random entries or the whole thing if fast.
    # 48 terms, 12x24x18 tensor.
    # 48 * 12 * 24 * 18 ~ 250k ops. Fast.
    
    T = np.zeros((12, 24, 18))
    for r in range(Rank):
        if np.any(U_mat[:, r] != 0):
            T += np.einsum('i,j,k->ijk', U_mat[:, r], V_mat[:, r], W_mat[:, r])
            
    # Check against schoolbook
    A = np.eye(12).reshape(3, 4) # This is not right for verification.
    # We need to verify that the decomposition represents the multiplication tensor.
    # The tensor T_{i,j,k} should be 1 if i,j,k correspond to a valid product term?
    # No, the tensor IS the multiplication map.
    # T_{i,j,k} = sum_c U_{i,c} V_{j,c} W_{k,c}.
    # For matrix multiplication, T_{i,j,k} is 1 if j is the inner index?
    # Wait, the tensor indices are:
    # U: (n*m) x rank. Index i*m + j.
    # V: (m*p) x rank. Index j*p + k.
    # W: (p*n) x rank. Index k*n + i.
    # The tensor entry T_{i,j,k} in the (n*m) x (m*p) x (p*n) space?
    # No, the tensor is defined such that C = AB.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor T has indices (i, j) for A, (j, k) for B, (k, i) for C?
    # The contract says:
    # U: (n*m, rank). Index i*m + j.
    # V: (m*p, rank). Index j*p + k.
    # W: (p*n, rank). Index k*n + i.
    # This implies the tensor T is of size (n*m) x (m*p) x (p*n).
    # T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}.
    # For schoolbook, T_{i*m+j, j*p+k, k*n+i} = 1.
    # So we check if our T matches this pattern.
    
    T_school = np.zeros((12, 24, 18))
    for i in range(3):
        for j in range(4):
            for k in range(6):
                T_school[i*4+j, j*6+k, k*3+i] = 1
                
    if not np.allclose(T, T_school):
        # Fallback to schoolbook if decomposition is wrong
        # This should not happen, but safety first.
        U_f = np.zeros((12, 72))
        V_f = np.zeros((24, 72))
        W_f = np.zeros((18, 72))
        c = 0
        for i in range(3):
            for j in range(4):
                for k in range(6):
                    U_f[i*4+j, c] = 1
                    V_f[j*6+k, c] = 1
                    W_f[k*3+i, c] = 1
                    c += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
        
    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}

# EVOLVE-BLOCK-END
