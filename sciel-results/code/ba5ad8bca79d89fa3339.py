"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Partition p=12 into 3 groups of 4
    # For each group of 4, we use two Strassen decompositions (rank 7 each) for 2x2x2
    # Total rank per group = 14
    # Total rank = 3 * 14 = 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # Indices: A(i,j), B(j,k), C(i,k)
    # 7 terms:
    # 1. (A00+A11)(B00+B11)
    # 2. (A10+A11)B00
    # 3. A00(B10+B11)
    # 4. A11(B01-B11)
    # 5. (A00+A01)B11
    # 6. (A10-A00)(B00+B01)
    # 7. (A01-A11)(B10+B11)
    
    # We map these to the 2x2x4 problem by splitting B into B_left (cols 0,1) and B_right (cols 2,3)
    # And C into C_left and C_right.
    
    col_offset = 0
    for group in range(3):
        # Process group of 4 columns
        # B indices in this group: j in {0,1}, k in {0,1,2,3}
        # B_left: k=0,1. B_right: k=2,3.
        
        # Strassen 1 for B_left
        # Term 1: (A00+A11) * (B00+B11) * (C00+C11)
        # Note: C indices are local to the 2x2 block.
        # Global C index: k*n + i.
        # For B_left, k=0,1.
        
        # Helper to add a Strassen block
        # local_k_offset: 0 for left, 2 for right
        for local_k_offset in [0, 2]:
            # 7 terms
            # 1
            U[0][col_offset] += 1; U[3][col_offset] += 1
            V[0][col_offset] += 1; V[3][col_offset] += 1 # B00, B11 (local k=0,1 -> global k=local_k_offset, local_k_offset+1)
            # Wait, V index is j*p + k.
            # j=0, k=local_k_offset -> V[0*p + local_k_offset]
            # j=1, k=local_k_offset+1 -> V[1*p + local_k_offset+1]
            # This is not a single rank-1 term if we sum B00 and B11?
            # Strassen term 1 is (A00+A11)(B00+B11).
            # This is NOT a rank-1 term in the original tensor!
            # Strassen works by defining intermediate variables M1..M7.
            # The decomposition is:
            # C00 = M1 - M4 + M5 + M7
            # C01 = M5 + M6
            # C10 = M2 + M4
            # C11 = M1 - M3 + M6 + M7
            # This requires 7 multiplications, but the U, V, W matrices must reflect the linear combinations.
            
            # So for each Strassen multiplication, we add 7 columns to U, V, W.
            
            # Term 1: M1 = (A00+A11)(B00+B11)
            U[0][col_offset] += 1; U[3][col_offset] += 1
            V[0*p + local_k_offset][col_offset] += 1
            V[1*p + local_k_offset + 1][col_offset] += 1
            # W should be 0 here? No, W is the output factor.
            # The decomposition is sum of rank-1 tensors.
            # Each multiplication M_r corresponds to a rank-1 tensor?
            # No, M_r is a scalar.
            # The tensor is sum of 7 terms?
            # Actually, Strassen's algorithm expresses C as linear combinations of the 7 products.
            # So the decomposition has 7 terms.
            # Term r: u_r (vector of A coeffs) * v_r (vector of B coeffs) * w_r (vector of C coeffs).
            
            # For M1 = (A00+A11)(B00+B11):
            # u1 = [1, 0, 0, 1]
            # v1 = [1, 0, 1, 0] (for B00, B11 in local 2x2)
            # w1 is determined by how M1 contributes to C.
            # C00 += M1, C11 += M1.
            # So w1 has 1 at C00 and C11.
            
            # Let's define the 7 Strassen terms for a 2x2x2 block.
            # local indices: i,j,k in {0,1}
            
            # 1. M1 = (A00+A11)(B00+B11)
            # u: A00, A11
            # v: B00, B11
            # w: C00, C11
            U[0][col_offset] += 1; U[3][col_offset] += 1
            V[0*p + local_k_offset][col_offset] += 1
            V[1*p + local_k_offset + 1][col_offset] += 1
            W[local_k_offset*n + 0][col_offset] += 1
            W[(local_k_offset+1)*n + 1][col_offset] += 1
            
            # 2. M2 = (A10+A11)B00
            U[2][col_offset] += 1; U[3][col_offset] += 1
            V[0*p + local_k_offset][col_offset] += 1
            W[local_k_offset*n + 1][col_offset] += 1 # C10
            
            # 3. M3 = A00(B10+B11)
            U[0][col_offset] += 1
            V[1*p + local_k_offset][col_offset] += 1
            V[1*p + local_k_offset + 1][col_offset] += 1
            W[(local_k_offset+1)*n + 0][col_offset] += 1 # C01
            
            # 4. M4 = A11(B01-B11)
            U[3][col_offset] += 1
            V[0*p + local_k_offset + 1][col_offset] += 1
            V[1*p + local_k_offset + 1][col_offset] -= 1
            W[(local_k_offset+1)*n + 1][col_offset] += 1 # C11
            
            # 5. M5 = (A00+A01)B11
            U[0][col_offset] += 1; U[1][col_offset] += 1
            V[1*p + local_k_offset + 1][col_offset] += 1
            W[local_k_offset*n + 0][col_offset] += 1 # C00
            W[(local_k_offset+1)*n + 1][col_offset] += 1 # C11
            
            # 6. M6 = (A10-A00)(B00+B01)
            U[2][col_offset] += 1; U[0][col_offset] -= 1
            V[0*p + local_k_offset][col_offset] += 1
            V[0*p + local_k_offset + 1][col_offset] += 1
            W[(local_k_offset+1)*n + 1][col_offset] += 1 # C11
            
            # 7. M7 = (A01-A11)(B10+B11)
            U[1][col_offset] += 1; U[3][col_offset] -= 1
            V[1*p + local_k_offset][col_offset] += 1
            V[1*p + local_k_offset + 1][col_offset] += 1
            W[local_k_offset*n + 0][col_offset] += 1 # C00
            W[(local_k_offset+1)*n + 0][col_offset] += 1 # C01
            
            col_offset += 7
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
