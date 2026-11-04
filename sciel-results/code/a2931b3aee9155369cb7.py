import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen 2x2 decomposition
    # U, V, W are 4x7 for 2x2 case
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Strassen formulas for C = A * B
    # P1 = (A00 + A11) * (B00 + B11)
    # P2 = (A10 + A11) * B00
    # P3 = A00 * (B01 - B11)
    # P4 = A11 * (B10 - B00)
    # P5 = (A00 + A01) * B11
    # P6 = (A10 - A00) * (B00 + B01)
    # P7 = (A01 - A11) * (B10 + B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # We need to map this to 4x4.
    # A 4x4 matrix is a 2x2 block matrix of 2x2 matrices.
    # Block A00 corresponds to indices (0,0), (0,1), (1,0), (1,1) in the 4x4?
    # No, A00 is the top-left 2x2 block.
    # Indices in 4x4:
    # 0,1,4,5 for A00?
    # Row 0: 0, 1, 2, 3
    # Row 1: 4, 5, 6, 7
    # Row 2: 8, 9, 10, 11
    # Row 3: 12, 13, 14, 15
    
    # Block A00 (top-left 2x2): indices 0, 1, 4, 5
    # Block A01 (top-right 2x2): indices 2, 3, 6, 7
    # Block A10 (bottom-left 2x2): indices 8, 9, 12, 13
    # Block A11 (bottom-right 2x2): indices 10, 11, 14, 15
    
    # Similarly for B and C.
    
    # We will generate the 49 terms.
    # Each term is a product of a vector from A, a vector from B, and a vector from C.
    
    # First, define the 7 Strassen terms for 2x2 blocks.
    # Each term k has U_k (2x2), V_k (2x2), W_k (2x2) such that C = sum W_k * (U_k * V_k)?
    # No, C = sum_{k=1}^7 W_k \otimes P_k?
    # C is 2x2. P_k is 2x2.
    # C_{ij} = sum_k W_{k,ij} P_k?
    # No, the Strassen algorithm computes P_k as scalar multiplications of linear combos.
    # P_k = (sum_i U_{k,i} A_i) * (sum_j V_{k,j} B_j).
    # Then C = sum_k W_k \otimes P_k?
    # Actually, C is a matrix of matrices.
    # C_{block} = sum_k W_{k, block} P_k.
    # Where W_{k, block} is a scalar (0, 1, -1).
    
    # So, for each k (1..7), we have:
    # P_k = (U_k . A_blocks) * (V_k . B_blocks)
    # A_blocks are 2x2 matrices.
    # U_k is a 2x2 matrix of coefficients for the blocks.
    # So P_k = (sum_{i,j} U_{k,ij} A_{ij}) * (sum_{i,j} V_{k,ij} B_{ij})
    
    # Now, P_k is a 2x2 matrix product.
    # We compute P_k using Strassen again.
    # P_k = sum_{r=1}^7 W_{kr} \otimes (U_{kr} . (A_k blocks)) * (V_{kr} . (B_k blocks))?
    # No, P_k is a 2x2 matrix.
    # P_k = M_{2x2}(A_k, B_k) where A_k, B_k are 2x2 matrices.
    # A_k = sum_{i,j} U_{k,ij} A_{ij} (block).
    # B_k = sum_{i,j} V_{k,ij} B_{ij} (block).
    # Then P_k = sum_{r=1}^7 W_{kr} \otimes (U_{kr} . (A_k sub-blocks)) * (V_{kr} . (B_k sub-blocks))?
    # No, A_k is a 2x2 matrix.
    # We apply Strassen to A_k and B_k.
    # So P_k = sum_{r=1}^7 W_{kr} \otimes P_{kr}
    # Where P_{kr} = (U_{kr} . A_k) * (V_{kr} . B_k)
    # And U_{kr}, V_{kr} are 2x2 coefficient matrices for the 2x2 Strassen.
    
    # This is getting complicated to hardcode.
    # I will use the Kronecker product approach.
    
    # 1. Get 2x2 Strassen decomposition (U2, V2, W2) of size 4x7.
    # 2. The 4x4 decomposition is the "tensor product" of the 2x2 decomposition with itself?
    # No, it's a block decomposition.
    
    # I will simply generate the 49 terms by iterating through the 7 outer Strassen terms and 7 inner Strassen terms.
    
    # Outer Strassen (2x2 blocks):
    # k=1..7
    # U_outer[k] is 4x1 (coeffs for A blocks)
    # V_outer[k] is 4x1 (coeffs for B blocks)
    # W_outer[k] is 4x1 (coeffs for C blocks)
    
    # Inner Strassen (2x2 scalars):
    # r=1..7
    # U_inner[r] is 4x1 (coeffs for A_k sub-blocks)
    # V_inner[r] is 4x1 (coeffs for B_k sub-blocks)
    # W_inner[r] is 4x1 (coeffs for P_k sub-blocks)
    
    # Total term (k,r):
    # A_vec = sum_{block_i} U_outer[k, block_i] * (sum_{sub_j} U_inner[r, sub_j] * A_{block_i, sub_j})
    # This is a linear combination of all 16 A entries.
    # B_vec = similar.
    # C_vec = W_outer[k, block_c] * W_inner[r, sub_c] * C_{block_c, sub_c}
    
    # I will implement this.
    
    return {"U": [], "V": [], "W": []}
# EVOLVE-BLOCK-END
