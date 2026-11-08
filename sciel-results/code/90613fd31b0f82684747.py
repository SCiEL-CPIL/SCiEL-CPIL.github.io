"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Constructs a tensor decomposition for matrix multiplication C = A @ B
    where A is n x m and B is m x p.
    
    Decomposition form:
    C_k_i = sum_l W_{k,i,l} * ( sum_{i',j'} U_{i',j',l} A_{i',j'} ) * ( sum_{j'',k'} V_{j'',k',l} B_{j'',k'} )
    
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)  <-- Index is k*n + i
    """
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    
    # Deadline guard
    start_time = time.time()
    timeout = 100.0  # Keep well within 120s limit

    # We will use a block structure.
    # A is 2x4, B is 4x5.
    # Split A into A_L (2x2) and A_R (2x2).
    # Split B into B_1 (2x5) and B_2 (2x5) along the first dimension (rows).
    # B_1: rows 0,1 (2x5). B_2: rows 2,3 (2x5).
    # C = A_L @ B_1 + A_R @ B_2.
    # Both terms are 2x2 @ 2x5 -> 2x5.
    
    # So we need to solve two instances of M_{2,2,5}.
    # What is the rank of M_{2,2,5}?
    # We can split B (2x5) into B_a (2x2), B_b (2x2), B_c (2x1).
    # Term = A @ B_a + A @ B_b + A @ B_c.
    # M_{2,2,2} has rank 7 (Strassen).
    # M_{2,2,2} has rank 7.
    # M_{2,2,1} (output 2x1) -> This is like vector-matrix product.
    #    Output y (2x1). y = A x.
    #    y0 = a00 x0 + a01 x1
    #    y1 = a10 x0 + a11 x1
    #    This can be done with 4 multiplications (rank 4).
    #    Can we do better?
    #    Tensor for 2x2x1.
    #    Basis vectors e_i, e_j, e_k.
    #    T = e_0 e_0 e_0 + e_1 e_0 e_0 ... wait indices.
    #    Usually defined as sum over intermediate index.
    #    Here B is 2x1, A is 2x2.
    #    Result C is 2x1.
    #    C_0 = A_00 B_00 + A_01 B_10
    #    C_1 = A_10 B_00 + A_11 B_10
    #    4 terms. Rank 4 is likely minimal. (Bilipski 1976 says rank 2x2x1 is 4).
    #
    # So rank(M_{2,2,5}) <= 7 + 7 + 4 = 18.
    #
    # Total rank = 2 * 18 = 36.
    #
    # Can we do better than 18 for M_{2,2,5}?
    # Maybe combine the two 2x2 blocks of B?
    # B is 2x4. A is 2x2.
    # M_{2,2,4}.
    # Rank 14? (7+7).
    # Then add M_{2,2,1} rank 4. Total 18.
    #
    # Is there a known decomposition for M_{2,2,5} with rank < 18?
    # Search space is small, but manual derivation is hard.
    # Standard algorithms give 18.
    #
    # Let's check if we can combine A_L and A_R.
    # We have A_L B_1 and A_R B_2.
    # Maybe define intermediate sums?
    # This is equivalent to finding a decomposition for the full 2x4x5 tensor.
    # The standard construction from sub-problems gives 36.
    #
    # Is there a "Strassen" trick for the whole thing?
    # A (2x4) -> 2x2 blocks.
    # B (4x5) -> 2x5 blocks.
    #
    # Let's stick to 36 as it's the standard bound.
    # However, if I can find a rank 35 solution, that would be an improvement.
    #
    # Let's try to be smart about the 2x2x1 part.
    # In M_{2,2,5}, we have two 2x2x2 parts and one 2x2x1 part.
    # The 2x2x1 part involves A's columns and B's specific column.
    # For A_L @ B_1, the 2x2x1 part is A_L @ B_1[:, 4] (last col of B_1, which is col 4 of B).
    # For A_R @ B_2, the 2x2x1 part is A_R @ B_2[:, 4] (last col of B_2, which is col 4 of B).
    #
    # Let B_c1 = B[:, 4] (2x1) for the first block? No, B_1 is rows 0,1.
    # B_1[:, 4] is B[0:2, 4].
    # B_2[:, 4] is B[2:4, 4].
    #
    # Let u = B[0:2, 4] (vector of size 2).
    # Let v = B[2:4, 4] (vector of size 2).
    #
    # Term 1 (from A_L): C[:, 4] += A_L @ u.
    # Term 2 (from A_R): C[:, 4] += A_R @ v.
    #
    # C_0,4 = A_L[0,:] @ u + A_R[0,:] @ v
    # C_1,4 = A_L[1,:] @ u + A_R[1,:] @ v
    #
    # This sums to:
    # C_0,4 = A_L[0,:] u + A_R[0,:] v
    # C_1,4 = A_L[1,:] u + A_R[1,:] v
    #
    # This looks like two independent 2D dot products added together.
    #
    # Let's see if we can merge the "Strassen" parts for the first 4 columns.
    # For columns 0-3 of B:
    # B_left is 4x4.
    # We compute A_L @ B_L1 + A_R @ B_L2.
    #
    # Let's construct the 36 rank solution explicitly to ensure it's correct,
    # then see if we can prune.
    
    rank = 36
    # Shapes
    # U: (n*m, rank) = (8, 36)
    # V: (m*p, rank) = (20, 36)
    # W: (p*n, rank) = (10, 36)
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    def u_idx(i, j): return i*m + j
    def v_idx(j, k): return j*p + k
    def w_idx(k, i): return k*n + i
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 decomposition
    # A_block: 2x2 at A rows i0,i1 and cols j0,j1
    # B_block: 2x2 at B rows j0,j1 and cols k0,k1
    # C_block: 2x2 at C rows k0,k1 and cols i0,i1
    # w_idx is k*n+i.
    def add_strassen_2x2x2(ai0, ai1, aj0, aj1, bj0, bj1, bk0, bk1):
        nonlocal col
        # Strassen constants
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        a00 = u_idx(ai0, aj0)
        a01 = u_idx(ai0, aj1)
        a10 = u_idx(ai1, aj0)
        a11 = u_idx(ai1, aj1)
        
        b00 = v_idx(bj0, bk0)
        b01 = v_idx(bj0, bk1)
        b10 = v_idx(bj1, bk0)
        b11 = v_idx(bj1, bk1)
        
        c00 = w_idx(bk0, ai0) # k=bk0, i=ai0
        c01 = w_idx(bk1, ai0) # k=bk1, i=ai0
        c10 = w_idx(bk0, ai1) # k=bk0, i=ai1
        c11 = w_idx(bk1, ai1) # k=bk1, i=ai1
        
        # M1
        U[a00][col] += 1; U[a11][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1
        
        # M2
        U[a10][col] += 1; U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1; W[c11][col] -= 1
        col += 1
        
        # M3
        U[a00][col] += 1
        V[b01][col] += 1; V[b11][col] -= 1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1
        
        # M4
        U[a11][col] += 1
        V[b10][col] += 1; V[b00][col] -= 1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1
        
        # M5
        U[a00][col] += 1; U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1; W[c01][col] += 1
        col += 1
        
        # M6
        U[a10][col] += 1; U[a00][col] -= 1
        V[b00][col] += 1; V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7
        U[a01][col] += 1; U[a11][col] -= 1
        V[b10][col] += 1; V[b11][col] += 1
        W[c00][col] += 1
        col += 1
        
    # Helper to add a 2x2 @ 2x1 decomposition (Rank 4)
    # A_block: 2x2 at A rows i0,i1 cols j0,j1
    # B_vec: 2x1 at B rows j0,j1 col k
    # C_vec: 2x1 at C rows k, cols i0,i1
    def add_vec_mat_2x2x1(ai0, ai1, aj0, aj1, bj0, bj1, bk):
        nonlocal col
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        
        a00 = u_idx(ai0, aj0)
        a01 = u_idx(ai0, aj1)
        a10 = u_idx(ai1, aj0)
        a11 = u_idx(ai1, aj1)
        
        b0 = v_idx(bj0, bk)
        b1 = v_idx(bj1, bk)
        
        c0 = w_idx(bk, ai0)
        c1 = w_idx(bk, ai1)
        
        # M1: A00 B0 -> C0
        U[a00][col] = 1; V[b0][col] = 1; W[c0][col] += 1; col += 1
        # M2: A01 B1 -> C0
        U[a01][col] = 1; V[b1][col] = 1; W[c0][col] += 1; col += 1
        # M3: A10 B0 -> C1
        U[a10][col] = 1; V[b0][col] = 1; W[c1][col] += 1; col += 1
        # M4: A11 B1 -> C1
        U[a11][col] = 1; V[b1][col] = 1; W[c1][col] += 1; col += 1
        
    # Block 1: A_L (rows 0,1 cols 0,1) @ B_11 (rows 0,1 cols 0,1) -> C_11 (rows 0,1 cols 0,1)
    # A_L is A[0:2, 0:2]. B_11 is B[0:2, 0:2]. Result C[0:2, 0:2].
    # w_idx: k in 0,1. i in 0,1.
    add_strassen_2x2x2(0, 1, 0, 1, 0, 1, 0, 1)
    
    # Block 2: A_L @ B_12 (rows 0,1 cols 2,3) -> C_12 (rows 2,3 cols 0,1)
    # A_L same. B_12 is B[0:2, 2:4]. Result C[2:4, 0:2].
    # w_idx: k in 2,3. i in 0,1.
    add_strassen_2x2x2(0, 1, 0, 1, 0, 1, 2, 3)
    
    # Block 3: A_L @ B_13 (rows 0,1 col 4) -> C_13 (row 4 cols 0,1)
    # A_L same. B_13 is B[0:2, 4]. Result C[4, 0:2].
    add_vec_mat_2x2x1(0, 1, 0, 1, 0, 1, 4)
    
    # Block 4: A_R (rows 0,1 cols 2,3) @ B_21 (rows 2,3 cols 0,1) -> C_21 (rows 0,1 cols 0,1)
    # A_R is A[0:2, 2:4]. B_21 is B[2:4, 0:2]. Result C[0:2, 0:2].
    add_strassen_2x2x2(0, 1, 2, 3, 2, 3, 0, 1)
    
    # Block 5: A_R @ B_22 (rows 2,3 cols 2,3) -> C_22 (rows 2,3 cols 0,1)
    # A_R same. B_22 is B[2:4, 2:4]. Result C[2:4, 0:2].
    add_strassen_2x2x2(0, 1, 2, 3, 2, 3, 2, 3)
    
    # Block 6: A_R @ B_23 (rows 2,3 col 4) -> C_23 (row 4 cols 0,1)
    # A_R same. B_23 is B[2:4, 4]. Result C[4, 0:2].
    add_vec_mat_2x2x1(0, 1, 2, 3, 2, 3, 4)
    
    # Total columns used: 7*4 + 4*2 = 28 + 8 = 36.
    # Matches rank=36.
    
    # Check if we are running out of time
    if time.time() - start_time > timeout:
        return {"U": U, "V": V, "W": W}

    # Attempt to reduce rank by checking for linear dependencies
    # We can use SVD or QR decomposition to find the true rank
    # and potentially reduce the number of terms if some are linearly dependent
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # We want to see if any column is a linear combination of others
    # This is equivalent to checking the rank of the tensor
    # A simpler heuristic: check if any column in U, V, or W is zero
    # If a column is zero in any of U, V, or W, it contributes nothing
    
    new_rank = rank
    keep_cols = []
    
    for c in range(rank):
        if np.all(U_np[:, c] == 0) or np.all(V_np[:, c] == 0) or np.all(W_np[:, c] == 0):
            continue
        keep_cols.append(c)
    
    if len(keep_cols) < new_rank:
        new_rank = len(keep_cols)
        U_np = U_np[:, keep_cols]
        V_np = V_np[:, keep_cols]
        W_np = W_np[:, keep_cols]
        
        # Convert back to lists
        U = U_np.tolist()
        V = V_np.tolist()
        W = W_np.tolist()
        
        # Update rank
        rank = new_rank

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
