"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Constructs a rank-minimal (or near-minimal) tensor decomposition for multiplying
    a 3x4 matrix by a 4x8 matrix.
    
    Strategy:
    1. Split the 3x4x8 problem into four 3x2x4 sub-problems based on the middle dimension (j=0..3).
       - j=0,1 uses A[:, 0:2] and B[0:2, :]
       - j=2,3 uses A[:, 2:4] and B[2:4, :]
    2. For each 3x2x4 sub-problem, split the 4 columns of B into two 2-column blocks.
       This results in eight 3x2x2 sub-problems.
    3. Each 3x2x2 sub-problem is solved using Strassen's algorithm for the 2x2x2 part,
       combined with the 3-row structure.
       Specifically, for a 3x2 A and 2x2 B:
       We can view this as two independent 3x1x2 multiplications? No.
       C[i, k] = sum_j A[i, j] B[j, k].
       Using Strassen on the 2x2 B and the 2 columns of A?
       
       Let's derive the rank for 3x2x2.
       Schoolbook: 3*2*2 = 12.
       Strassen on 2x2x2 has rank 7.
       We have 3 rows in A.
       We can group the 3 rows of A into (2, 1).
       Let A = [A_top (2x2); A_bot (1x2)].
       Let B = B (2x2).
       C_top (2x2) = A_top * B. Rank 7 (Strassen).
       C_bot (1x2) = A_bot * B. 
       A_bot is 1x2. B is 2x2.
       C_bot[0, 0] = A_bot[0,0]B[0,0] + A_bot[0,1]B[1,0]
       C_bot[0, 1] = A_bot[0,0]B[0,1] + A_bot[0,1]B[1,1]
       This is a 1x2x2 multiplication.
       Schoolbook rank for 1x2x2 is 4.
       Can we do better?
       Let x = A_bot[0,0], y = A_bot[0,1].
       Let B = [[b00, b01], [b10, b11]].
       c0 = x*b00 + y*b10
       c1 = x*b01 + y*b11
       
       This is just 4 multiplications.
       Is there a rank 3 decomposition?
       The tensor is T_{i,j,k} = delta_{i,0} delta_{j,k}? No.
       Indices: A index (0, j), B index (j, k), C index (0, k).
       T_{(0,j), (j,k), (0,k)} = 1.
       This is equivalent to 2x2x2 tensor?
       Let's check rank of 1x2x2.
       It is the same as 2x2x2? No.
       The 2x2x2 tensor has rank 7.
       The 1x2x2 tensor is a "slice" or related?
       Actually, 1x2x2 multiplication is:
       c0 = a0 b00 + a1 b10
       c1 = a0 b01 + a1 b11
       
       Consider the 2x2x2 Strassen decomposition.
       M1 = (a00+a11)(b00+b11)
       M2 = (a10+a11)b00
       M3 = a00(b10-b11)
       M4 = a11(b01-b00)
       M5 = (a00+a01)b11
       M6 = (a10-a00)(b00+b01)
       M7 = (a01-a11)(b10+b11)
       
       If we set a01 = a10 = a11 = 0?
       Then A = [[a00, 0], [0, 0]] = [[x, 0], [0, 0]].
       This is not 1x2.
       
       Let's stick to the 1x2x2 schoolbook rank 4.
       Total rank per 3x2x2 = 7 (for 2x2x2) + 4 (for 1x2x2) = 11.
       
       Wait, we have 8 such sub-problems.
       Total rank = 8 * 11 = 88.
       Schoolbook is 96.
       88 < 96.
       
       Can we do better on the 1x2x2 part?
       The tensor for 1x2x2 is:
       U indices: (0,0), (0,1) -> 2 entries.
       V indices: (0,0), (0,1), (1,0), (1,1) -> 4 entries.
       W indices: (0,0), (0,1) -> 2 entries.
       
       Equations:
       sum_r U_00_r V_00_r W_00_r + U_01_r V_10_r W_00_r = 1 (for c00)
       sum_r U_00_r V_01_r W_01_r + U_01_r V_11_r W_01_r = 1 (for c01)
       All other W entries must be 0.
       
       Let's try rank 3.
       We need to satisfy:
       T_{(0,0),(0,0),(0,0)} = 1
       T_{(0,1),(1,0),(0,0)} = 1
       T_{(0,0),(0,1),(0,1)} = 1
       T_{(0,1),(1,1),(0,1)} = 1
       
       This looks like two independent 1x1x1 multiplications?
       No, they share the A and B indices partially.
       Actually, the tensor is:
       T_{a,b,c}
       a in {0,1} (col of A)
       b in {0,1,2,3} (row,col of B) -> (0,0), (0,1), (1,0), (1,1)
       c in {0,1} (col of C)
       
       Non-zeros:
       a=0, b=(0,0), c=0 -> 1
       a=1, b=(1,0), c=0 -> 1
       a=0, b=(0,1), c=1 -> 1
       a=1, b=(1,1), c=1 -> 1
       
       This is isomorphic to two 1x1x1 multiplications?
       Let's check.
       The pairs (a,b,c) are:
       (0, 0, 0)
       (1, 2, 0)  [b=(1,0) is index 2 if 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)]
       (0, 1, 1)
       (1, 3, 1)
       
       These are 4 disjoint "hyper-edges".
       The rank of a tensor with disjoint hyper-edges is the number of hyper-edges?
       Yes, if they don't share any indices in a way that allows sharing rank.
       Here, do they share indices?
       (0,0,0) uses A_0, B_00, C_0.
       (1,2,0) uses A_1, B_10, C_0.
       They share C_0.
       (0,1,1) uses A_0, B_01, C_1.
       (1,3,1) uses A_1, B_11, C_1.
       They share C_1.
       
       We can combine terms sharing C_0?
       Term 1 + Term 2: A_0 B_00 C_0 + A_1 B_10 C_0.
       This is one bilinear form? No, it's a sum of two products.
       To decompose a sum of products into a sum of rank-1 tensors, we usually need one rank-1 per product unless there is structure.
       Here, the structure is:
       C_0 = A_0 B_00 + A_1 B_10.
       This is a dot product of (A_0, A_1) and (B_00, B_10).
       This is a 1x2x1 multiplication?
       A is 1x2, B is 2x1, C is 1x1.
       Rank of 1x2x1 is 2?
       c = a0 b0 + a1 b1.
       Rank 2. (e.g., M1=a0b0, M2=a1b1).
       Can it be rank 1?
       c = (a0+a1)(b0+b1) - a0b1 - a1b0 ... no, that's rank 3.
       c = (a0+a1)(b0+b1) - a0b0 - a1b1 ... no.
       The tensor T_{a,b,c} = delta_{a,b}.
       Rank of identity tensor of size 2 is 2.
       So 1x2x1 has rank 2.
       
       So C_0 part has rank 2.
       C_1 part has rank 2.
       Total rank for 1x2x2 is 2 + 2 = 4.
       So schoolbook is optimal for 1x2x2?
       Yes, rank 4.
       
       So 3x2x2 rank is 7 + 4 = 11.
       Total 8 * 11 = 88.
       
       Is there a better way for 3x2x2?
       Maybe treat 3x2 as 3x1 + 3x1?
       No, 3x2x2.
       
       Let's check if we can combine the 2x2x2 and 1x2x2 parts.
       A = [[a00, a01], [a10, a11], [a20, a21]].
       B = [[b00, b01], [b10, b11]].
       
       C = A B.
       
       We used Strassen for top 2 rows.
       M1..M7 compute C_top.
       Then we compute C_bot = [a20, a21] B.
       
       Can we reuse any M?
       M2 = (a10+a11)b00.
       M3 = a00(b10-b11).
       
       C_bot[0] = a20 b00 + a21 b10.
       C_bot[1] = a20 b01 + a21 b11.
       
       It doesn't seem like we can reuse Strassen terms directly for the 3rd row without adding more terms.
       
       So 88 is a safe bet.
       
       Let's implement the 88 rank decomposition.
    """
    n, m, p = DIMENSIONS
    # We will construct U, V, W matrices.
    # Total rank R = 8 * 11 = 88.
    
    R = 88
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4 + M5
    # C10 = M1 - M2 + M6 + M7
    # C11 = M2 - M3 + M6
    
    # Indices for 2x2x2 block
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B indices: (0,0), (0,1), (1,0), (1,1)
    # C indices: (0,0), (0,1), (1,0), (1,1)
    
    # We have 8 blocks of 3x2x2.
    # Block (j_pair, k_pair)
    # j_pair in {0, 1} -> j in {0,1} or {2,3}
    # k_pair in {0, 1, 2, 3} -> k in {0,1}, {2,3}, {4,5}, {6,7}
    
    rank_offset = 0
    for jp in range(2):
        j_base = jp * 2
        for kp in range(4):
            k_base = kp * 2
            
            # --- 2x2x2 part (Top 2 rows of A) ---
            # A rows: 0, 1 (global rows 0, 1)
            # A cols: j_base, j_base+1
            # B rows: j_base, j_base+1
            # B cols: k_base, k_base+1
            # C rows: 0, 1
            # C cols: k_base, k_base+1
            
            # M1: (A00+A11)(B00+B11)
            U[0*m + j_base, rank_offset] = 1
            U[1*m + j_base+1, rank_offset] = 1
            V[j_base*p + k_base, rank_offset] = 1
            V[(j_base+1)*p + (k_base+1), rank_offset] = 1
            W[k_base*n + 0, rank_offset] = 1
            W[(k_base+1)*n + 1, rank_offset] = 1
            rank_offset += 1
            
            # M2: (A10+A11)B00
            U[1*m + j_base, rank_offset] = 1
            U[1*m + j_base+1, rank_offset] = 1
            V[j_base*p + k_base, rank_offset] = 1
            W[k_base*n + 1, rank_offset] = 1
            rank_offset += 1
            
            # M3: A00(B10-B11)
            U[0*m + j_base, rank_offset] = 1
            V[(j_base+1)*p + k_base, rank_offset] = 1
            V[(j_base+1)*p + (k_base+1), rank_offset] = -1
            W[k_base*n + 0, rank_offset] = 1
            W[(k_base+1)*n + 0, rank_offset] = 1
            rank_offset += 1
            
            # M4: A11(B01-B00)
            U[1*m + j_base+1, rank_offset] = 1
            V[j_base*p + (k_base+1), rank_offset] = 1
            V[j_base*p + k_base, rank_offset] = -1
            W[k_base*n + 0, rank_offset] = -1
            W[(k_base+1)*n + 0, rank_offset] = 1
            rank_offset += 1
            
            # M5: (A00+A01)B11
            U[0*m + j_base, rank_offset] = 1
            U[0*m + j_base+1, rank_offset] = 1
            V[(j_base+1)*p + (k_base+1), rank_offset] = 1
            W[(k_base+1)*n + 0, rank_offset] = 1
            W[(k_base+1)*n + 1, rank_offset] = 1
            rank_offset += 1
            
            # M6: (A10-A00)(B00+B01)
            U[1*m + j_base, rank_offset] = 1
            U[0*m + j_base, rank_offset] = -1
            V[j_base*p + k_base, rank_offset] = 1
            V[j_base*p + (k_base+1), rank_offset] = 1
            W[k_base*n + 1, rank_offset] = 1
            W[(k_base+1)*n + 1, rank_offset] = 1
            rank_offset += 1
            
            # M7: (A01-A11)(B10+B11)
            U[0*m + j_base+1, rank_offset] = 1
            U[1*m + j_base+1, rank_offset] = -1
            V[(j_base+1)*p + k_base, rank_offset] = 1
            V[(j_base+1)*p + (k_base+1), rank_offset] = 1
            W[k_base*n + 1, rank_offset] = 1
            W[(k_base+1)*n + 1, rank_offset] = 1
            rank_offset += 1
            
            # --- 1x2x2 part (Bottom 1 row of A) ---
            # A row: 2
            # A cols: j_base, j_base+1
            # B rows: j_base, j_base+1
            # B cols: k_base, k_base+1
            # C row: 2
            # C cols: k_base, k_base+1
            
            # C20 = A20 B00 + A21 B10
            # C21 = A20 B01 + A21 B11
            
            # Term 1: A20 B00 -> C20
            U[2*m + j_base, rank_offset] = 1
            V[j_base*p + k_base, rank_offset] = 1
            W[k_base*n + 2, rank_offset] = 1
            rank_offset += 1
            
            # Term 2: A21 B10 -> C20
            U[2*m + j_base+1, rank_offset] = 1
            V[(j_base+1)*p + k_base, rank_offset] = 1
            W[k_base*n + 2, rank_offset] = 1
            rank_offset += 1
            
            # Term 3: A20 B01 -> C21
            U[2*m + j_base, rank_offset] = 1
            V[j_base*p + (k_base+1), rank_offset] = 1
            W[(k_base+1)*n + 2, rank_offset] = 1
            rank_offset += 1
            
            # Term 4: A21 B11 -> C21
            U[2*m + j_base+1, rank_offset] = 1
            V[(j_base+1)*p + (k_base+1), rank_offset] = 1
            W[(k_base+1)*n + 2, rank_offset] = 1
            rank_offset += 1
            
    # Convert to lists of lists as required by interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
