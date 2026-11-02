import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=2, m=4, p=8
    
    # We will use a block decomposition approach.
    # We can view the 2x4 * 4x8 multiplication as a block multiplication.
    # A is 2x4. B is 4x8.
    # We can split A into [A1 A2] where A1, A2 are 2x2.
    # We can split B into [B1; B2] where B1, B2 are 2x8.
    # Then C = A1*B1 + A2*B2.
    
    # Let's try to decompose each 2x2 * 2x8 multiplication using Strassen.
    # A1 (2x2) * B1 (2x8) -> C1 (2x8)
    # A2 (2x2) * B2 (2x8) -> C2 (2x8)
    
    # For a 2x2 * 2x8 multiplication:
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01, ..., b07], [b10, b11, ..., b17]]
    # C = [[c00, ..., c07], [c10, ..., c17]]
    
    # Strassen components for 2x2 * 2x8:
    # M1 = (a00+a11) * (b0* + b1*)  -> vector of 8
    # M2 = (a10+a11) * b0*
    # M3 = a00 * (b1* - b0*)
    # M4 = a11 * (b1* - b0*)  # Wait, standard Strassen uses b10-b00 etc.
    # Let's stick to the standard 7-term Strassen for 2x2 * 2x2 and extend to 2x8.
    
    # For 2x2 * 2x8, B is two row-vectors of length 8.
    # Let B0 = B[0, :], B1 = B[1, :].
    # M1 = (a00+a11) * (B0+B1)
    # M2 = (a10+a11) * B0
    # M3 = a00 * (B1-B0)
    # M4 = a11 * (B1-B0)  # This is not standard. Standard is a11*(b10-b00) for scalar.
    # For vectors:
    # M4 = a11 * (B1 - B0) ? No.
    
    # Let's use the explicit Strassen formulas for 2x2 * 2x2 and apply to each column pair of B.
    # But that would be 7 * 4 = 28 terms for 2x2 * 2x8.
    # Total rank for 2x4 * 4x8 would be 2 * 28 = 56.
    # This matches the current score of 1.1429 (which corresponds to rank 56, since 64/56 = 1.1428...).
    
    # Is it possible to do better?
    # The lower bound for 2x2 * 2x8 might be less than 28.
    # However, 56 is a known bound.
    
    # Let's try to optimize the constant factor or check if a different decomposition exists.
    # The task asks for "rank-minimal".
    # Current rank is 56.
    
    # Let's construct the rank-56 decomposition carefully.
    
    total_rank = 56
    
    U = np.zeros((n * m, total_rank), dtype=float) # 8 x 56
    V = np.zeros((m * p, total_rank), dtype=float) # 32 x 56
    W = np.zeros((p * n, total_rank), dtype=float) # 16 x 56
    
    # Helper to set local Strassen matrices for 2x2 * 2x2
    # Indices in U_local: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices in V_local: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices in W_local: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    U_local = [[0]*7 for _ in range(4)]
    V_local = [[0]*7 for _ in range(4)]
    W_local = [[0]*7 for _ in range(4)]
    
    # r=0 (M1)
    U_local[0][0] = 1; U_local[3][0] = 1 
    V_local[0][0] = 1; V_local[3][0] = 1 
    W_local[0][0] = 1 
    W_local[3][0] = 1 
    
    # r=1 (M2)
    U_local[2][1] = 1; U_local[3][1] = 1 
    V_local[0][1] = 1 
    W_local[2][1] = 1 
    W_local[3][1] = -1 
    
    # r=2 (M3)
    U_local[0][2] = 1 
    V_local[1][2] = 1; V_local[3][2] = -1 
    W_local[1][2] = 1 
    W_local[3][2] = 1 
    
    # r=3 (M4)
    U_local[3][3] = 1 
    V_local[2][3] = 1; V_local[0][3] = -1 
    W_local[0][3] = 1 
    W_local[2][3] = 1 
    
    # r=4 (M5)
    U_local[0][4] = 1; U_local[1][4] = 1 
    V_local[3][4] = 1 
    W_local[0][4] = -1 
    W_local[1][4] = 1 
    
    # r=5 (M6)
    U_local[2][5] = 1; U_local[0][5] = -1 
    V_local[0][5] = 1; V_local[1][5] = 1 
    W_local[3][5] = 1 
    
    # r=6 (M7)
    U_local[1][6] = 1; U_local[3][6] = -1 
    V_local[2][6] = 1; V_local[3][6] = 1 
    W_local[0][6] = 1 
    
    # Global mapping
    # We split A (2x4) into two 2x2 blocks: A1 (cols 0,1), A2 (cols 2,3)
    # We split B (4x8) into two 2x8 blocks: B1 (rows 0,1), B2 (rows 2,3)
    # C = A1*B1 + A2*B2
    
    # For A1*B1:
    # A1 uses A indices 0..3
    # B1 uses B indices 0..15 (rows 0,1 of B, which are 8 cols each)
    # Wait, B is 4x8. Flat index j*8+k.
    # B1 is rows 0,1. So flat indices 0..7 (row 0) and 8..15 (row 1).
    # C1 is 2x8.
    
    # For A2*B2:
    # A2 uses A indices 4..7 (rows 0,1 of A, cols 2,3). Flat index i*4+j.
    # A2[0,0] = A[0,2] = 2, A2[0,1] = A[0,3] = 3
    # A2[1,0] = A[1,2] = 6, A2[1,1] = A[1,3] = 7
    # B2 is rows 2,3. Flat indices 16..23 (row 2) and 24..31 (row 3).
    # C2 is 2x8.
    
    # W index convention: W[k*n+i, r]. n=2. So W[k*2+i, r].
    # C1 and C2 both contribute to the same C (2x8).
    
    # Block 1: A1*B1
    # A1 local (i_loc, j_loc) -> Global A index i_loc*4 + j_loc
    a1_map = {0: 0, 1: 1, 2: 4, 3: 5} # (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    
    # B1 local (j_loc, k_loc) -> Global B index j_loc*8 + k_loc
    # j_loc 0 -> Global row 0, j_loc 1 -> Global row 1
    # k_loc 0..7 -> Global col k_loc
    # So B1 local index lb (0..3) corresponds to (j_loc, k_loc).
    # But Strassen V_local has entries for 2x2 B.
    # We need to map the 2x2 B block to the 2x8 B block.
    # V_local[lb][l] is the coefficient for B[j_loc, k_loc] in term l.
    # For 2x8, B has 2 rows and 8 columns.
    # The Strassen terms M1..M7 are linear combinations of the entries of B.
    # Since the multiplication is C[i,k] = sum_j A[i,j]B[j,k], and Strassen works for 2x2,
    # we can apply it to each pair of columns of B.
    # B1 is 2x8. Let's split B1 into 4 blocks of 2x2: B1_0, B1_1, B1_2, B1_3.
    # B1_0: cols 0,1. B1_1: cols 2,3. B1_2: cols 4,5. B1_3: cols 6,7.
    
    # This means for each 2x2 block of B1, we have 7 terms.
    # Total terms for A1*B1 = 4 * 7 = 28.
    # Similarly for A2*B2 = 28.
    # Total rank = 56.
    
    # Let's iterate over the 4 column pairs of B1.
    for c_pair in range(4):
        base_r = c_pair * 7
        # B1 columns 2*c_pair, 2*c_pair+1
        # B1 is rows 0,1 of global B.
        # Global B indices:
        # Row 0, Col 2*c_pair -> 0*8 + 2*c_pair
        # Row 0, Col 2*c_pair+1 -> 0*8 + 2*c_pair+1
        # Row 1, Col 2*c_pair -> 1*8 + 2*c_pair
        # Row 1, Col 2*c_pair+1 -> 1*8 + 2*c_pair+1
        
        b1_map = {
            0: 0*8 + 2*c_pair,
            1: 0*8 + 2*c_pair+1,
            2: 1*8 + 2*c_pair,
            3: 1*8 + 2*c_pair+1
        }
        
        # C1 output columns 2*c_pair, 2*c_pair+1
        # C1 is rows 0,1 of global C.
        # Global C indices (W convention k*n+i):
        # Row i=0, Col k=2*c_pair -> W[2*c_pair*2 + 0]
        # Row i=0, Col k=2*c_pair+1 -> W[(2*c_pair+1)*2 + 0]
        # Row i=1, Col k=2*c_pair -> W[2*c_pair*2 + 1]
        # Row i=1, Col k=2*c_pair+1 -> W[(2*c_pair+1)*2 + 1]
        
        c1_map = {
            0: 2*c_pair*2 + 0,
            1: (2*c_pair+1)*2 + 0,
            2: 2*c_pair*2 + 1,
            3: (2*c_pair+1)*2 + 1
        }
        
        for l in range(7):
            r = base_r + l
            for la in range(4):
                val = U_local[la][l]
                if val != 0:
                    U[a1_map[la]][r] += val
            for lb in range(4):
                val = V_local[lb][l]
                if val != 0:
                    V[b1_map[lb]][r] += val
            for lc in range(4):
                val = W_local[lc][l]
                if val != 0:
                    W[c1_map[lc]][r] += val

    # Block 2: A2*B2
    # A2 local (i_loc, j_loc) -> Global A index i_loc*4 + 2 + j_loc
    a2_map = {0: 2, 1: 3, 2: 6, 3: 7}
    
    for c_pair in range(4):
        base_r = 28 + c_pair * 7
        # B2 columns 2*c_pair, 2*c_pair+1
        # B2 is rows 2,3 of global B.
        b2_map = {
            0: 2*8 + 2*c_pair,
            1: 2*8 + 2*c_pair+1,
            2: 3*8 + 2*c_pair,
            3: 3*8 + 2*c_pair+1
        }
        
        # C2 output columns 2*c_pair, 2*c_pair+1
        # C2 is rows 0,1 of global C (same as C1, they add up)
        c2_map = {
            0: 2*c_pair*2 + 0,
            1: (2*c_pair+1)*2 + 0,
            2: 2*c_pair*2 + 1,
            3: (2*c_pair+1)*2 + 1
        }
        
        for l in range(7):
            r = base_r + l
            for la in range(4):
                val = U_local[la][l]
                if val != 0:
                    U[a2_map[la]][r] += val
            for lb in range(4):
                val = V_local[lb][l]
                if val != 0:
                    V[b2_map[lb]][r] += val
            for lc in range(4):
                val = W_local[lc][l]
                if val != 0:
                    W[c2_map[lc]][r] += val

    # Convert to lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if time.time() - start_time > 140:
        pass
        
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
