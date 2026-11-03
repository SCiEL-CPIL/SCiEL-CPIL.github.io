"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-33 exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    
    Strategy:
    Split the 3x6 matrix B into three 3x2 blocks (B1, B2, B3) along the column dimension.
    The product C = A * B is the concatenation of A*B1, A*B2, A*B3 along columns.
    Each sub-problem is a 2x3 by 3x2 multiplication.
    
    For a 2x3 by 3x2 multiplication, we can achieve rank 11 using a Strassen-like decomposition:
    Split A (2x3) into A11 (2x2) and A12 (2x1).
    Split B (3x2) into B11 (2x2) and B21 (1x2).
    
    The product C = A11*B11 + A12*B21.
    
    A11*B11 is a 2x2 by 2x2 multiplication, which has rank 7 (Strassen).
    A12*B21 is a 2x1 by 1x2 multiplication, which is outer product, rank 1? 
    Wait, A12 is 2x1, B21 is 1x2. The product is 2x2.
    The tensor for (2,1,2) has rank 2? Let's check.
    C_{ik} = sum_j A_{ij} B_{jk}. Here j is only 0. So C_{ik} = A_{i0} B_{0k}.
    This is the outer product of vector A12 and vector B21.
    The tensor is sum_{i,k} A_{i0} B_{0k} C_{ik}.
    This is a rank-1 tensor? No, it's the product of two vectors.
    In matrix multiplication tensor terms, T_{i,j,k} = delta_{j,0}.
    The rank of the (2,1,2) multiplication tensor is 2?
    Let's verify.
    A = [a; b], B = [c d].
    C = [[ac, ad], [bc, bd]].
    C11 = ac, C12 = ad, C21 = bc, C22 = bd.
    We need to express these as linear combinations of rank-1 terms A_u B_v.
    Term 1: A_u = [a, 0], B_v = [c, 0] -> ac, 0, 0, 0.
    Term 2: A_u = [a, 0], B_v = [0, d] -> 0, ad, 0, 0.
    Term 3: A_u = [0, b], B_v = [c, 0] -> 0, 0, bc, 0.
    Term 4: A_u = [0, b], B_v = [0, d] -> 0, 0, 0, bd.
    So rank 4?
    Can we do better?
    Try:
    1. A=[a,0], B=[c,0] -> ac
    2. A=[a,0], B=[0,d] -> ad
    3. A=[0,b], B=[c,0] -> bc
    4. A=[0,b], B=[0,d] -> bd
    Is there a Strassen-like trick for 2x1x2?
    The tensor is T = e_1 \otimes f_1 \otimes g_1 + e_2 \otimes f_2 \otimes g_2? No.
    T_{ijk} = 1 if j=0 else 0.
    This is a slice of the identity.
    Actually, the rank of (n, 1, p) is n*p. Here n=2, p=2, so rank 4.
    
    So R(2,3,2) = R(2,2,2) + R(2,1,2) = 7 + 4 = 11.
    
    Total rank = 3 * 11 = 33.
    """
    n, m, p = DIMENSIONS
    
    # Helper to create Strassen factors for 2x2x2
    def strassen_2x2x2():
        # Returns U, V, W of shape (4, 7)
        # Standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Indices in U: 0=A11, 1=A12, 2=A21, 3=A22
        # Indices in V: 0=B11, 1=B12, 2=B21, 3=B22
        # Indices in W: 0=C11, 1=C12, 2=C21, 3=C22
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    # Helper to create schoolbook factors for 2x1x2 (outer product)
    def outer_2x1x2():
        # Returns U, V, W of shape (2, 4), (1, 4), (2, 4)
        # A is 2x1, B is 1x2.
        # C11 = A1*B1, C12 = A1*B2, C21 = A2*B1, C22 = A2*B2
        U = np.zeros((2, 4))
        V = np.zeros((1, 4))
        W = np.zeros((2, 4))
        
        # Indices U: 0=A1, 1=A2
        # Indices V: 0=B1, 1=B2 (Wait, B is 1x2, so indices are 0,1? No, shape is (1,2). 
        # In the full 3x2 B block, the 1x2 part corresponds to row 2 (index 2) of the 3x2 matrix.
        # But here we are building the local factors.
        # Local A: 2x1. Local B: 1x2.
        # V indices: 0, 1.
        
        # C indices: 0=C11, 1=C12, 2=C21, 3=C22.
        
        # Term 1: A1*B1 -> C11
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        
        # Term 2: A1*B2 -> C12
        U[0, 1] = 1
        V[1, 1] = 1
        W[1, 1] = 1
        
        # Term 3: A2*B1 -> C21
        U[1, 2] = 1
        V[0, 2] = 1
        W[2, 2] = 1
        
        # Term 4: A2*B2 -> C22
        U[1, 3] = 1
        V[1, 3] = 1
        W[3, 3] = 1
        
        return U, V, W

    # Build the full 2x3x2 factors by combining Strassen and Outer
    def build_2x3x2():
        # U shape (6, 11), V shape (6, 11), W shape (4, 11)
        # Global indices for 2x3 A:
        # 0: (0,0), 1: (0,1), 2: (0,2)
        # 3: (1,0), 4: (1,1), 5: (1,2)
        # Local A11 (2x2) uses indices 0,1,3,4
        # Local A12 (2x1) uses indices 2,5
        
        # Global indices for 3x2 B:
        # 0: (0,0), 1: (0,1)
        # 2: (1,0), 3: (1,1)
        # 4: (2,0), 5: (2,1)
        # Local B11 (2x2) uses indices 0,1,2,3
        # Local B21 (1x2) uses indices 4,5
        
        # Global indices for 2x2 C:
        # 0: (0,0), 1: (0,1)
        # 2: (1,0), 3: (1,1)
        
        U_str, V_str, W_str = strassen_2x2x2()
        U_out, V_out, W_out = outer_2x1x2()
        
        U = np.zeros((6, 11))
        V = np.zeros((6, 11))
        W = np.zeros((4, 11))
        
        # Copy Strassen part (cols 0-6)
        # Map Strassen U indices (0,1,2,3) to Global A indices (0,1,3,4)
        mapping_U_str = [0, 1, 3, 4]
        for r in range(7):
            for local_u, val in enumerate(U_str[:, r]):
                if val != 0:
                    U[mapping_U_str[local_u], r] = val
        
        # Map Strassen V indices (0,1,2,3) to Global B indices (0,1,2,3)
        mapping_V_str = [0, 1, 2, 3]
        for r in range(7):
            for local_v, val in enumerate(V_str[:, r]):
                if val != 0:
                    V[mapping_V_str[local_v], r] = val
        
        # Map Strassen W indices (0,1,2,3) to Global C indices (0,1,2,3)
        mapping_W_str = [0, 1, 2, 3]
        for r in range(7):
            for local_w, val in enumerate(W_str[:, r]):
                if val != 0:
                    W[mapping_W_str[local_w], r] = val
        
        # Copy Outer part (cols 7-10)
        # Map Outer U indices (0,1) to Global A indices (2,5)
        mapping_U_out = [2, 5]
        for r in range(4):
            for local_u, val in enumerate(U_out[:, r]):
                if val != 0:
                    U[mapping_U_out[local_u], r + 7] = val
        
        # Map Outer V indices (0,1) to Global B indices (4,5)
        mapping_V_out = [4, 5]
        for r in range(4):
            for local_v, val in enumerate(V_out[:, r]):
                if val != 0:
                    V[mapping_V_out[local_v], r + 7] = val
        
        # Map Outer W indices (0,1,2,3) to Global C indices (0,1,2,3)
        mapping_W_out = [0, 1, 2, 3]
        for r in range(4):
            for local_w, val in enumerate(W_out[:, r]):
                if val != 0:
                    W[mapping_W_out[local_w], r + 7] = val
        
        return U, V, W

    U_block, V_block, W_block = build_2x3x2()
    
    # Assemble full 2x3x6 factors
    # Total rank 33
    U_full = np.zeros((6, 33))
    V_full = np.zeros((18, 33))
    W_full = np.zeros((12, 33))
    
    for block_idx in range(3):
        col_offset = block_idx * 11
        
        # U is same for all blocks (A is unchanged)
        U_full[:, col_offset:col_offset+11] = U_block
        
        # V mapping:
        # Block 0 uses B rows 0,1,2 and cols 0,1 (Global B indices 0..5)
        # Block 1 uses B rows 0,1,2 and cols 2,3 (Global B indices 6..11)
        # Block 2 uses B rows 0,1,2 and cols 4,5 (Global B indices 12..17)
        # Local V indices 0..5 map to Global V indices block_idx*6 + local_v
        for r in range(11):
            for local_v, val in enumerate(V_block[:, r]):
                if val != 0:
                    V_full[block_idx * 6 + local_v, col_offset + r] = val
        
        # W mapping:
        # Block 0 produces C cols 0,1 (Global C indices 0..3)
        # Block 1 produces C cols 2,3 (Global C indices 4..7)
        # Block 2 produces C cols 4,5 (Global C indices 8..11)
        # Local W indices 0..3 map to Global W indices block_idx*4 + local_w
        for r in range(11):
            for local_w, val in enumerate(W_block[:, r]):
                if val != 0:
                    W_full[block_idx * 4 + local_w, col_offset + r] = val
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END

