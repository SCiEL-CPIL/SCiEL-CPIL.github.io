"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _get_strassen_2x2():
    """
    Returns U, V, W matrices for 2x2x2 Strassen decomposition.
    Shapes: (4, 7), (4, 7), (4, 7)
    Indices for U/V/W: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1 # C00, C11
    
    # M2 = (A10+A11)B00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1; W[3,1] = -1 # C10, C11
    
    # M3 = A00(B10-B00)
    U[0,2] = 1
    V[2,2] = 1; V[0,2] = -1
    W[1,2] = 1; W[3,2] = 1 # C01, C11
    
    # M4 = A11(B11-B01)
    U[3,3] = 1
    V[3,3] = 1; V[1,3] = -1
    W[0,3] = 1; W[2,3] = 1 # C00, C10
    
    # M5 = (A00+A01)B11
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[1,4] = 1 # C00, C01
    
    # M6 = (A10-A00)(B00+B01)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1 # C11
    
    # M7 = (A01-A11)(B10+B11)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1; W[1,6] = -1 # C00, C01
    
    return U, V, W

def _construct_4x4x4_strassen():
    """
    Construct U, V, W for 4x4x4 matrix multiplication using Strassen.
    Returns U (16, 49), V (16, 49), W (16, 49)
    """
    U2, V2, W2 = _get_strassen_2x2()
    
    # Outer Strassen coefficients for 4x4
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x2)
    # C blocks: C00, C01, C10, C11 (each 2x2)
    
    # Outer terms L_A (2x2 matrix of linear forms in A blocks)
    # and L_B (2x2 matrix of linear forms in B blocks)
    
    # We define the 7 outer combinations
    # Each combination is a pair of 2x2 matrices (LA, LB)
    # LA[i,j] is a linear combination of A blocks
    # LB[i,j] is a linear combination of B blocks
    
    # Let's represent LA and LB as 4x4 matrices of coefficients for A and B blocks?
    # A has 4 blocks (0,1,2,3). B has 4 blocks (0,1,2,3).
    
    outer_U = np.zeros((4, 7)) # Coeffs for A blocks in LA
    outer_V = np.zeros((4, 7)) # Coeffs for B blocks in LB
    
    # M1 = (A00+A11)(B00+B11)
    outer_U[0,0] = 1; outer_U[3,0] = 1
    outer_V[0,0] = 1; outer_V[3,0] = 1
    
    # M2 = (A10+A11)B00
    outer_U[2,1] = 1; outer_U[3,1] = 1
    outer_V[0,1] = 1
    
    # M3 = A00(B10-B00)
    outer_U[0,2] = 1
    outer_V[2,2] = 1; outer_V[0,2] = -1
    
    # M4 = A11(B11-B01)
    outer_U[3,3] = 1
    outer_V[3,3] = 1; outer_V[1,3] = -1
    
    # M5 = (A00+A01)B11
    outer_U[0,4] = 1; outer_U[1,4] = 1
    outer_V[3,4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    outer_U[2,5] = 1; outer_U[0,5] = -1
    outer_V[0,5] = 1; outer_V[1,5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    outer_U[1,6] = 1; outer_U[3,6] = -1
    outer_V[2,6] = 1; outer_V[3,6] = 1
    
    # Now, for each outer term l, we have LA and LB.
    # LA is a 2x2 matrix where each entry is a linear combination of A blocks.
    # Actually, the Strassen formula for 2x2 multiplication of matrices X, Y:
    # X = [[X00, X01], [X10, X11]]
    # Y = [[Y00, Y01], [Y10, Y11]]
    # M1 = (X00+X11)(Y00+Y11)
    # ...
    # Here X is a 2x2 matrix of A-blocks?
    # No, in the 4x4 case, the "scalars" of the 2x2 Strassen are the 2x2 blocks of A.
    # So X00 = A00, X01 = A01, X10 = A10, X11 = A11.
    # Y00 = B00, Y01 = B01, Y10 = B10, Y11 = B11.
    
    # The outer Strassen computes 7 "products" of 2x2 matrices.
    # Product l is X_l * Y_l.
    # X_l is a 2x2 matrix formed by linear combinations of A blocks.
    # Y_l is a 2x2 matrix formed by linear combinations of B blocks.
    
    # We need to decompose X_l * Y_l (a 2x2 matrix product) into scalar terms.
    # We use the inner Strassen (2x2x2) for this.
    
    U_4 = np.zeros((16, 49))
    V_4 = np.zeros((16, 49))
    W_4 = np.zeros((16, 49))
    
    col = 0
    for l in range(7):
        # Construct X_l and Y_l as 2x2 matrices of coefficients for A and B blocks
        # X_l[i,j] is a vector of length 4 (coeffs for A00..A11)
        # Y_l[i,j] is a vector of length 4 (coeffs for B00..B11)
        
        # The outer coefficients outer_U[:, l] give the linear combo for the "scalar" positions in X.
        # Wait, the outer Strassen defines X_l as a matrix where the entries are linear combos.
        # For M1: X1 = [[A00+A11, 0], [0, 0]]? No.
        # Standard Strassen:
        # M1 = (X00+X11)(Y00+Y11). This is a product of two scalars (which are 2x2 blocks).
        # So X1 is not a 2x2 matrix of blocks, but the term is (X00+X11) * (Y00+Y11).
        # This is a product of two 2x2 matrices.
        # Let P_l = X_l * Y_l.
        # X_l is a 2x2 matrix? No, X_l is a linear combination of X blocks.
        # In Strassen, M_k are scalars. Here they are 2x2 matrices.
        # M1 = (A00+A11)(B00+B11).
        # This is the product of matrix (A00+A11) and matrix (B00+B11).
        # Let X1 = A00+A11, Y1 = B00+B11.
        # P1 = X1 Y1.
        
        # So for each l, we have a 2x2 matrix X_l and a 2x2 matrix Y_l.
        # X_l = sum_{block} c_{block} A_{block}
        # Y_l = sum_{block} d_{block} B_{block}
        
        # We need to decompose P_l = X_l Y_l.
        # X_l and Y_l are 2x2 matrices.
        # We can use the inner Strassen decomposition of 2x2x2.
        # The inner Strassen decomposes the product of two 2x2 matrices into 7 scalar terms.
        
        # Let's get the coefficients for X_l and Y_l.
        # X_l is a 2x2 matrix. Its entries are linear combos of A blocks.
        # Actually, X_l is just a 2x2 matrix of "scalars" (which are 2x2 blocks).
        # So X_l[0,0] is a linear combo of A blocks.
        
        # Let's define X_l and Y_l as 4x4 matrices? No, 2x2 matrices of vectors.
        
        # For l=0 (M1):
        # X1 = A00 + A11. This is a 2x2 matrix where every entry is A00+A11?
        # No, A00 and A11 are 2x2 matrices.
        # X1 is a 2x2 matrix.
        # X1 = [[A00+A11, 0], [0, 0]]? No.
        # The term is (A00+A11)(B00+B11).
        # This is the product of the matrix (A00+A11) and (B00+B11).
        # So X1 = A00+A11 (a 2x2 matrix).
        # Y1 = B00+B11 (a 2x2 matrix).
        
        # So X_l is a 2x2 matrix.
        # We can represent X_l by its 4 entries, each being a vector of coeffs for A blocks.
        
        X_l = np.zeros((2, 2, 4)) # [row, col, block_index]
        Y_l = np.zeros((2, 2, 4))
        
        # Map outer_U coefficients to X_l
        # outer_U[0] is coeff for A00, outer_U[1] for A01, etc.
        # But Strassen M1 uses A00 and A11.
        # X1 = A00 + A11.
        # So X1[0,0] = A00+A11? No, X1 IS A00+A11.
        # So X1 is a 2x2 matrix where all entries are the same?
        # No, A00 and A11 are 2x2 matrices.
        # X1 = A00 + A11 is a 2x2 matrix.
        # So X1[0,0] = (A00+A11)[0,0], etc.
        # This means X1 is a linear combination of A blocks.
        # X1 = c0*A00 + c1*A01 + c2*A10 + c3*A11.
        
        # For M1: X1 = 1*A00 + 0*A01 + 0*A10 + 1*A11.
        # So X_l is a 2x2 matrix formed by this linear combo.
        # Wait, if X1 = A00+A11, then X1 is a 2x2 matrix.
        # We can decompose X1 * Y1 using inner Strassen.
        
        # Coeffs for X1: [1, 0, 0, 1]
        # Coeffs for Y1: [1, 0, 0, 1]
        
        # We need to extract the 4 entries of the 2x2 matrix X1.
        # X1[0,0] = A00[0,0] + A11[0,0]
        # X1[0,1] = A00[0,1] + A11[0,1]
        # etc.
        
        # So for the inner Strassen, the "input" A is X1.
        # The "input" B is Y1.
        
        # We can just use the inner Strassen U, V, W.
        # The inner U acts on the 4 entries of X1.
        # The entries of X1 are linear combos of A blocks.
        # So the total U for this term is U_inner * Coeffs_X.
        
        # Let's do this for all 7 outer terms.
        
        coeffs_X = outer_U[:, l]
        coeffs_Y = outer_V[:, l]
        
        # coeffs_X is a vector of length 4.
        # X1 = sum coeffs_X[i] * A_i.
        # The 4 entries of X1 (flattened) are linear combos of the 16 entries of A.
        # We can construct a 4x16 matrix M_X such that X1_flat = M_X @ A_flat.
        
        M_X = np.zeros((4, 16))
        for block in range(4):
            if coeffs_X[block] != 0:
                # A_block is 2x2. Flatten to 4.
                # Place at correct position in A_flat (16)
                # A_flat index: block*4 + i
                # X1_flat index: i
                for i in range(4):
                    M_X[i, block*4 + i] = coeffs_X[block]
        
        M_Y = np.zeros((4, 16))
        for block in range(4):
            if coeffs_Y[block] != 0:
                for i in range(4):
                    M_Y[i, block*4 + i] = coeffs_Y[block]
        
        # Now, decompose X1 * Y1 using inner Strassen.
        # Inner Strassen gives 7 terms.
        # For each inner term k:
        # u_k (4,1) acts on X1_flat.
        # v_k (4,1) acts on Y1_flat.
        # w_k (4,1) acts on C_block_flat.
        
        # Total u = M_X @ u_k (16,1)
        # Total v = M_Y @ v_k (16,1)
        # Total w: w_k acts on C_block_flat (4,1).
        # C_block is a 2x2 block of C (4x4).
        # Which block?
        # The outer Strassen places P_l into C.
        # C = [[P1+P4-P5+P7, P3+P5], [P2+P4, P1-P2+P3+P6]]
        # So P_l contributes to specific C blocks with specific signs.
        
        # We need to map w_k (which is for a 2x2 output) to the 4x4 C.
        
        # Let's define the placement of P_l in C.
        # C_block_00 (top-left): P1+P4-P5+P7
        # C_block_01 (top-right): P3+P5
        # C_block_10 (bottom-left): P2+P4
        # C_block_11 (bottom-right): P1-P2+P3+P6
        
        # For outer term l, we find the C blocks it contributes to.
        
        # We'll create a 4x16 matrix M_W_l such that C_flat = M_W_l @ C_block_flat?
        # No, C_block_flat is 4 entries. C_flat is 16 entries.
        # M_W_l is 16x4.
        
        M_W_l = np.zeros((16, 4))
        
        # Determine which C blocks P_l affects
        # l=0 (P1): C00 (+1), C11 (+1)
        # l=1 (P2): C10 (+1), C11 (-1)
        # l=2 (P3): C01 (+1), C11 (+1)
        # l=3 (P4): C00 (+1), C10 (+1)
        # l=4 (P5): C00 (-1), C01 (+1)
        # l=5 (P6): C11 (+1)
        # l=6 (P7): C00 (+1), C01 (-1)
        
        if l == 0:
            M_W_l[0:4, :] = 1.0 # C00
            M_W_l[12:16, :] = 1.0 # C11
        elif l == 1:
            M_W_l[8:12, :] = 1.0 # C10
            M_W_l[12:16, :] = -1.0 # C11
        elif l == 2:
            M_W_l[4:8, :] = 1.0 # C01
            M_W_l[12:16, :] = 1.0 # C11
        elif l == 3:
            M_W_l[0:4, :] = 1.0 # C00
            M_W_l[8:12, :] = 1.0 # C10
        elif l == 4:
            M_W_l[0:4, :] = -1.0 # C00
            M_W_l[4:8, :] = 1.0 # C01
        elif l == 5:
            M_W_l[12:16, :] = 1.0 # C11
        elif l == 6:
            M_W_l[0:4, :] = 1.0 # C00
            M_W_l[4:8, :] = -1.0 # C01
            
        # Now combine with inner Strassen
        for k in range(7):
            u_inner = U2[:, k]
            v_inner = V2[:, k]
            w_inner = W2[:, k]
            
            U_4[:, col] = M_X @ u_inner
            V_4[:, col] = M_Y @ v_inner
            W_4[:, col] = M_W_l @ w_inner
            
            col += 1
            
    return U_4, V_4, W_4

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Get 4x4x4 Strassen decomposition
    U_4, V_4, W_4 = _construct_4x4x4_strassen()
    
    # We need 4x4x8.
    # This is two 4x4x4 multiplications.
    # B is 4x8. Split into B1 (4x4) and B2 (4x4).
    # C is 4x8. Split into C1 (4x4) and C2 (4x4).
    
    # U is shared.
    # V is concatenated.
    # W is concatenated.
    
    # U shape: (16, 49)
    # V shape: (16, 49) -> (32, 98)
    # W shape: (16, 49) -> (32, 98)
    
    U = np.zeros((16, 98))
    V = np.zeros((32, 98))
    W = np.zeros((32, 98))
    
    # First 49 terms: for B1, C1
    U[:, 0:49] = U_4
    V[0:16, 0:49] = V_4 # B1 is first 16 entries of B
    W[0:16, 0:49] = W_4 # C1 is first 16 entries of C
    
    # Next 49 terms: for B2, C2
    U[:, 49:98] = U_4
    V[16:32, 49:98] = V_4 # B2 is second 16 entries of B
    W[16:32, 49:98] = W_4 # C2 is second 16 entries of C
    
    # Verify dimensions
    # U: (16, 98)
    # V: (32, 98)
    # W: (32, 98)
    
    # Convert to lists
    U_list = [[float(U[i, j]) for j in range(98)] for i in range(16)]
    V_list = [[float(V[i, j]) for j in range(98)] for i in range(32)]
    W_list = [[float(W[i, j]) for j in range(98)] for i in range(32)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
