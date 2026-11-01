"""Matrix multiplication 2x3x9 using a block decomposition with Strassen for the 2x2x2 core."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose M(2,3,9) into M(2,2,9) + M(2,1,9).
    # M(2,2,9) is split into three M(2,2,3) blocks.
    # Each M(2,2,3) is split into M(2,2,2) + M(2,2,1).
    # M(2,2,2) uses Strassen (rank 7).
    # M(2,2,1) uses schoolbook (rank 4).
    # So each M(2,2,3) block has rank 11.
    # Total rank for M(2,2,9) is 3 * 11 = 33.
    # M(2,1,9) uses schoolbook rank 2*1*9 = 18? No, rank of 2x1x9 is 2*9=18?
    # Wait, M(n,m,p) schoolbook rank is n*m*p.
    # M(2,1,9) rank is 2*1*9 = 18.
    # Total rank = 33 + 18 = 51.
    
    # Let's verify if we can do better on M(2,1,9).
    # M(2,1,9) is just outer product of vector u (len 2), scalar v (len 1), vector w (len 9).
    # This is a rank-1 tensor in the space of matrices? No, it's a 3-way tensor.
    # The tensor is T_{i,j,k} = A_{i,j} B_{j,k} C_{k,i}.
    # For j=1 fixed, T_{i,1,k} = A_{i,1} B_{1,k}.
    # This is a matrix of rank 1 (outer product of col A_1 and row B_1).
    # The tensor is decomposable?
    # T = sum_{r=1}^{R} U_r V_r W_r.
    # U is 2xR, V is 1xR, W is 9xR.
    # T_{i,k} = U_{i,r} V_{1,r} W_{k,r}.
    # This is a matrix of rank at most R.
    # Since T_{i,k} has rank 1, we can do R=1?
    # If R=1, T_{i,k} = u_i v_1 w_k. This is an outer product.
    # Yes, M(2,1,9) has rank 1?
    # Let's check.
    # A is 2x1, B is 1x9. C is 9x2.
    # C_{ik} = A_{i1} B_{1k}.
    # We want to express C_{ik} = sum_r U_{ir} V_{1r} W_{kr}.
    # Since there is only one j, V is 1xR.
    # Let V_{1r} = 1 for all r.
    # Then C_{ik} = sum_r U_{ir} W_{kr}.
    # This is a matrix multiplication of 2xR and Rx9.
    # The matrix C_{ik} has rank 1.
    # So we can choose R=1.
    # U_{i1} = A_{i1}, W_{k1} = B_{1k}, V_{11} = 1.
    # So M(2,1,9) has rank 1.
    
    # So total rank = 33 + 1 = 34.
    
    rank = 34
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Process the 3 blocks of M(2,2,3)
    # Block b corresponds to k indices: 3*b, 3*b+1, 3*b+2
    for b in range(3):
        k0 = 3 * b
        
        # Sub-block M(2,2,2) for k0, k0+1
        # Strassen decomposition for 2x2x2
        # A is 2x2 (rows i=0,1; cols j=0,1)
        # B is 2x2 (rows j=0,1; cols k=k0, k0+1)
        # C is 2x2 (rows i=0,1; cols k=k0, k0+1)
        
        # Strassen intermediates:
        # M1 = A00 * (B00 + B11)
        # M2 = (A00 + A10) * B11
        # M3 = (A01 + A11) * B00
        # M4 = A11 * (B10 + B01)
        # M5 = (A00 - A11) * (B10 + B11)
        # M6 = (A01 + A10) * (B00 - B10)
        # M7 = (A01 - A10) * (B11 + B00)
        
        # Indices for Strassen terms
        # U indices: A_{i,j} -> i*m + j. Here m=2 for the sub-block?
        # No, global m=3.
        # For this block, we are using j=0,1.
        # A_{i,0}, A_{i,1}.
        # Global U index: i*3 + j.
        # Global V index: j*9 + k.
        # Global W index: k*2 + i.
        
        # Term 1: A00 * (B00 + B11)
        # U: A00 -> U[0*3+0] = 1
        # V: B00 -> V[0*9+k0] = 1, B11 -> V[1*9+k0+1] = 1
        # W: C00 -> W[k0*2+0] = 1, C11 -> W[(k0+1)*2+1] = 1
        U[0][col] += 1
        V[k0][col] += 1
        V[9 + k0 + 1][col] += 1
        W[k0][col] += 1
        W[2*(k0+1) + 1][col] += 1
        col += 1
        
        # Term 2: (A00 + A10) * B11
        # U: A00 -> 1, A10 -> 1
        # V: B11 -> V[9 + k0 + 1] = 1
        # W: C10 -> W[k0*2+1] = 1, C11 -> W[(k0+1)*2+1] = 1?
        # C_{ij} from M2:
        # C_{10} = M2 (wait, C indices i=1, k=k0)
        # C_{11} = M2?
        # Let's re-verify Strassen output formulas.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M6
        # C11 = M1 - M3 + M6 + M7
        #
        # So M2 contributes to C10 and C11.
        # C10 corresponds to i=1, k=k0. W index: k0*2 + 1.
        # C11 corresponds to i=1, k=k0+1. W index: (k0+1)*2 + 1.
        
        U[0][col] += 1
        U[3][col] += 1  # A10 is i=1, j=0 -> 1*3+0
        V[9 + k0 + 1][col] += 1
        W[k0 * 2 + 1][col] += 1
        W[(k0 + 1) * 2 + 1][col] += 1
        col += 1
        
        # Term 3: (A01 + A11) * B00
        # U: A01 (i=0, j=1) -> 0*3+1=1, A11 (i=1, j=1) -> 1*3+1=4
        # V: B00 (j=0, k=k0) -> V[k0]
        # W: C01 (i=0, k=k0+1) -> W[(k0+1)*2+0], C11 (i=1, k=k0+1) -> W[(k0+1)*2+1]
        # M3 contributes to C01 and C11.
        
        U[1][col] += 1
        U[4][col] += 1
        V[k0][col] += 1
        W[(k0 + 1) * 2][col] += 1
        W[(k0 + 1) * 2 + 1][col] += 1
        col += 1
        
        # Term 4: A11 * (B10 + B01)
        # U: A11 -> U[4]
        # V: B10 (j=1, k=k0) -> V[9+k0], B01 (j=0, k=k0+1) -> V[k0+1]
        # W: C00 (i=0, k=k0) -> W[k0*2+0], C11 (i=1, k=k0+1) -> W[(k0+1)*2+1]
        # M4 contributes to C00 and C11.
        
        U[4][col] += 1
        V[9 + k0][col] += 1
        V[k0 + 1][col] += 1
        W[k0 * 2][col] += 1
        W[(k0 + 1) * 2 + 1][col] += 1
        col += 1
        
        # Term 5: (A00 - A11) * (B10 + B11)
        # U: A00 -> 1, A11 -> -1
        # V: B10 -> 1, B11 -> 1
        # W: C00 -> -1, C01 -> 1
        # M5 contributes to C00 (-1) and C01 (1).
        
        U[0][col] += 1
        U[4][col] += -1
        V[9 + k0][col] += 1
        V[9 + k0 + 1][col] += 1
        W[k0 * 2][col] += -1
        W[(k0 + 1) * 2][col] += 1
        col += 1
        
        # Term 6: (A01 + A10) * (B00 - B10)
        # U: A01 -> 1, A10 -> 1
        # V: B00 -> 1, B10 -> -1
        # W: C10 -> 1, C11 -> 1
        # M6 contributes to C10 (1) and C11 (1).
        
        U[1][col] += 1
        U[3][col] += 1
        V[k0][col] += 1
        V[9 + k0][col] += -1
        W[k0 * 2 + 1][col] += 1
        W[(k0 + 1) * 2 + 1][col] += 1
        col += 1
        
        # Term 7: (A01 - A10) * (B11 + B00)
        # U: A01 -> 1, A10 -> -1
        # V: B11 -> 1, B00 -> 1
        # W: C00 -> 1, C11 -> 1
        # M7 contributes to C00 (1) and C11 (1).
        
        U[1][col] += 1
        U[3][col] += -1
        V[9 + k0 + 1][col] += 1
        V[k0][col] += 1
        W[k0 * 2][col] += 1
        W[(k0 + 1) * 2 + 1][col] += 1
        col += 1
        
        # Now handle the third k index: k0+2
        # This is M(2,2,1)
        # Schoolbook for 2x2x1:
        # C_{i, k0+2} = A_{i,0} B_{0, k0+2} + A_{i,1} B_{1, k0+2}
        # Terms:
        # T1: A00 * B0_k
        # T2: A10 * B0_k
        # T3: A01 * B1_k
        # T4: A11 * B1_k
        
        # T1: A00 * B0_{k0+2}
        U[0][col] += 1
        V[k0 + 2][col] += 1
        W[(k0 + 2) * 2][col] += 1
        col += 1
        
        # T2: A10 * B0_{k0+2}
        U[3][col] += 1
        V[k0 + 2][col] += 1
        W[(k0 + 2) * 2 + 1][col] += 1
        col += 1
        
        # T3: A01 * B1_{k0+2}
        U[1][col] += 1
        V[9 + k0 + 2][col] += 1
        W[(k0 + 2) * 2][col] += 1
        col += 1
        
        # T4: A11 * B1_{k0+2}
        U[4][col] += 1
        V[9 + k0 + 2][col] += 1
        W[(k0 + 2) * 2 + 1][col] += 1
        col += 1

    # Finally, handle the M(2,1,9) part: j=2
    # Rank 1 term.
    # U: A_{i,2} -> U[i*3+2] = 1
    # V: B_{2,k} -> V[2*9+k] = 1
    # W: C_{i,k} -> W[k*2+i] = 1
    
    U[2][col] += 1  # A02
    U[5][col] += 1  # A12
    for k in range(9):
        V[18 + k][col] += 1  # B2k
        W[k * 2][col] += 1   # C0k
        W[k * 2 + 1][col] += 1 # C1k
    
    col += 1
    
    # Ensure col matches rank
    # If col > rank, we have a bug. If col < rank, we wasted space.
    # With the current logic, col should be 34.
    
    # Trim if necessary (though we sized it exactly)
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    # Convert to integers if possible? The task allows half-integers.
    # Our values are integers.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
