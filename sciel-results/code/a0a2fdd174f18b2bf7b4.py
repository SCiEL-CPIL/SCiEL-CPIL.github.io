"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    """
    n, m, p = 3, 3, 7
    
    # We use a Strassen-like decomposition for 4x4 matrices with schoolbook base (2x2).
    # Rank for 4x4x4 is 7 (Strassen steps) * 4 (Schoolbook base) = 28.
    # We split the 3x7 output into two 3x4 and 3x3 parts? 
    # Actually, we pad B to 4x8.
    # B_pad is 4x8. Split into B_L (4x4) and B_R (4x4).
    # A_pad is 4x4.
    # Compute C_L = A_pad * B_L (4x4) using rank 28 decomposition.
    # Compute C_R = A_pad * B_R (4x4) using rank 28 decomposition.
    # Total rank 56.
    
    rank_total = 56
    U = [[0] * rank_total for _ in range(n * m)]  # 9 x 56
    V = [[0] * rank_total for _ in range(m * p)]  # 21 x 56
    W = [[0] * rank_total for _ in range(p * n)]  # 21 x 56
    
    # Helper to add a rank-1 component
    # u_idx: index in A_pad (0..15)
    # v_idx: index in B_pad (0..31)
    # w_idx: index in C_pad (0..15)
    # coeff: coefficient
    def add_component(u_idx, v_idx, w_idx, coeff, rank_offset):
        if coeff == 0:
            return
        # Map A_pad index (4x4) to U index (3x3 part)
        # A_pad is 4x4. U corresponds to A (3x3).
        # A_pad[i,j] where i,j in 0..3.
        # If i<3 and j<3, U[i*3+j] gets coeff.
        # Else, it's 0 in U (since A is 3x3, padded with 0).
        # Wait, the linear forms in Strassen involve sums of A entries.
        # So we need to distribute the coeff to the constituent A entries.
        
        # This helper is too simple for Strassen linear forms.
        # We need to generate the linear forms first.
        pass

    # Let's build the decomposition explicitly.
    
    # Strassen formulas for 2x2 blocks:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each Mi is a 2x2 matrix.
    # We use schoolbook for each 2x2 multiplication.
    # Mi = Ai * Bi.
    # Mi[0,0] = Ai[0,0]*Bi[0,0] + Ai[0,1]*Bi[1,0]
    # Mi[0,1] = Ai[0,0]*Bi[0,1] + Ai[0,1]*Bi[1,1]
    # Mi[1,0] = Ai[1,0]*Bi[0,0] + Ai[1,1]*Bi[1,0]
    # Mi[1,1] = Ai[1,0]*Bi[0,1] + Ai[1,1]*Bi[1,1]
    
    # We need to express Ai and Bi in terms of the 4x4 A_pad and B_pad.
    
    # A_pad is 4x4.
    # A11 = A_pad[0:2, 0:2]
    # A12 = A_pad[0:2, 2:4]
    # A21 = A_pad[2:4, 0:2]
    # A22 = A_pad[2:4, 2:4]
    
    # B_pad is 4x4.
    # B11 = B_pad[0:2, 0:2]
    # B12 = B_pad[0:2, 2:4]
    # B21 = B_pad[2:4, 0:2]
    # B22 = B_pad[2:4, 2:4]
    
    # We will generate the 28 components for one 4x4 multiplication.
    # Then we will call this logic twice for B_L and B_R.
    
    def generate_4x4_components(B_offset):
        """
        Generates the 28 rank-1 components for A_pad * B_part.
        B_offset is 0 for B_L (cols 0-3 of B_pad) and 4 for B_R (cols 4-7 of B_pad).
        Returns list of (u_linear_form, v_linear_form, w_linear_form)
        """
        components = []
        
        # Define the 7 Strassen terms
        # Each term is defined by:
        # A_form: list of (row, col, coeff) for A_pad
        # B_form: list of (row, col, coeff) for B_pad (relative to B_part)
        # C_form: list of (row, col, coeff) for C_pad (relative to C_part)
        
        # M1 = (A11+A22)(B11+B22)
        A1 = [
            (0,0,1), (0,1,1),
            (1,0,1), (1,1,1),
            (2,2,1), (2,3,1),
            (3,2,1), (3,3,1)
        ]
        B1 = [
            (0,0,1), (0,1,1),
            (1,0,1), (1,1,1),
            (2,2,1), (2,3,1),
            (3,2,1), (3,3,1)
        ]
        C1 = [
            (0,0,1), (1,1,1),
            (2,2,-1), (3,3,1)
        ]
        # Wait, C11 = M1 + M4 - M5 + M7.
        # So M1 contributes +1 to C00, +1 to C11, -1 to C22, +1 to C33?
        # No, the Strassen formulas are:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # Where C11 is top-left 2x2 block of C.
        # So M1 contributes to C[0,0], C[1,1], C[2,2], C[3,3]?
        # No, M1 is a 2x2 matrix.
        # M1[0,0] contributes to C[0,0].
        # M1[0,1] contributes to C[0,1].
        # M1[1,0] contributes to C[1,0].
        # M1[1,1] contributes to C[1,1].
        # And similarly for other blocks.
        
        # Let's be precise.
        # C = [C11 C12; C21 C22] where Cij are 2x2 blocks.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So for M1:
        # M1[0,0] -> C[0,0] coeff 1
        # M1[0,1] -> C[0,1] coeff 1
        # M1[1,0] -> C[1,0] coeff 1
        # M1[1,1] -> C[1,1] coeff 1
        # M1[0,0] -> C[2,2] coeff -1 (from -M5? No, M1 is in C22 too)
        # C22 = M1 - M2 + M3 + M6
        # So M1[0,0] -> C[2,2] coeff 1
        # M1[0,1] -> C[2,3] coeff 1
        # M1[1,0] -> C[3,2] coeff 1
        # M1[1,1] -> C[3,3] coeff 1
        
        # This mapping is complex. Let's just compute the linear forms for A and B,
        # and then for each of the 4 scalar products in the schoolbook expansion,
        # determine which C entries it affects and with what coefficient.
        
        def expand_term(A_form, B_form, C_map):
            """
            A_form: list of (r, c, coeff) for A_pad
            B_form: list of (r, c, coeff) for B_pad (local)
            C_map: dict mapping (c_r, c_c) to list of (global_r, global_c, coeff)
                   where c_r, c_c are indices in the 2x2 result of this term.
            """
            # A_form is a 2x2 matrix of linear forms?
            # No, A_form is the entire 4x4 matrix A_virt?
            # In Strassen, A_virt is 2x2 blocks.
            # For M1, A_virt = A11 + A22.
            # A11 is top-left 2x2 of A_pad. A22 is bottom-right 2x2.
            # So A_virt is a 2x2 matrix where each entry is a sum of A_pad entries.
            # A_virt[0,0] = A_pad[0,0] + A_pad[2,2]
            # A_virt[0,1] = A_pad[0,1] + A_pad[2,3]
            # A_virt[1,0] = A_pad[1,0] + A_pad[3,2]
            # A_virt[1,1] = A_pad[1,1] + A_pad[3,3]
            
            # B_virt similarly.
            
            # Schoolbook for A_virt * B_virt:
            # P1 = A_virt[0,0] * B_virt[0,0]
            # P2 = A_virt[0,1] * B_virt[1,0]
            # P3 = A_virt[0,0] * B_virt[0,1]
            # P4 = A_virt[0,1] * B_virt[1,1]
            # P5 = A_virt[1,0] * B_virt[0,0]
            # P6 = A_virt[1,1] * B_virt[1,0]
            # P7 = A_virt[1,0] * B_virt[0,1]
            # P8 = A_virt[1,1] * B_virt[1,1]
            
            # Wait, schoolbook is:
            # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
            # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
            # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
            # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
            
            # So we have 4 scalar multiplications? No, 8 terms, but they combine.
            # In tensor rank, each product A[i,j]B[k,l] is a rank-1 component.
            # So we have 8 rank-1 components per Strassen term?
            # No, A_virt[0,0] is a sum of A_pad entries.
            # A_virt[0,0] * B_virt[0,0] = (sum a_i) * (sum b_j) = sum sum a_i b_j.
            # This expands to multiple rank-1 components.
            # This increases the rank significantly.
            
            # My previous calculation of rank 28 was based on treating A_virt * B_virt as a single "multiplication" of rank 4.
            # But A_virt entries are linear forms.
            # The rank of the tensor for A_virt * B_virt where A_virt, B_virt are linear forms in A_pad, B_pad is indeed 4 * (number of non-zero coeffs in A_virt) * (number of non-zero coeffs in B_virt)?
            # No.
            # The rank of the product of two linear forms is 1.
            # The rank of the sum of products is the sum of ranks.
            # So if A_virt[0,0] = A00 + A22, and B_virt[0,0] = B00 + B22.
            # A_virt[0,0] * B_virt[0,0] = A00B00 + A00B22 + A22B00 + A22B22.
            # This is 4 rank-1 components.
            
            # So for M1:
            # A_virt has 4 entries, each with 2 non-zero coeffs.
            # B_virt has 4 entries, each with 2 non-zero coeffs.
            # Each of the 4 schoolbook products involves 2*2=4 scalar products.
            # Total 16 scalar products for M1.
            # 7 Strassen terms * 16 = 112 rank.
            # This is worse than 63.
            
            # So the "Strassen with Schoolbook base" idea only works if the "base" matrices are the original scalar entries, not linear forms.
            # But Strassen combines blocks, so the "base" matrices are sums of blocks.
            # Thus, the rank is NOT 28. It is much higher.
            
            # Conclusion: This approach is flawed for rank reduction unless we use a different base.
            # We should stick to the schoolbook rank 63 or find a true rank reduction.
            
            pass
        
        return components

    # Given the complexity and the risk of the rank being higher than 63,
    # and the difficulty of finding a known rank < 63 decomposition for 3x3x7,
    # I will return the schoolbook decomposition which is guaranteed to be correct and has rank 63.
    
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    return {'U': U_sb, 'V': V_sb, 'W': W_sb}

# EVOLVE-BLOCK-END
