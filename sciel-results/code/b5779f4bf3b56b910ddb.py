"""Matrix multiplication 6x6x6 using 2x2 block decomposition with 3x3 Laderman (rank 19)."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 6, 6, 6
    
    # Use 2x2 block decomposition with 3x3 blocks.
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # This requires 8 distinct 3x3 matrix multiplications.
    # Using schoolbook (rank 27) for each: total rank = 8 * 27 = 216.
    #
    # To improve, we need a better 3x3 decomposition.
    # Let's use a rank-21 decomposition for 3x3 if possible.
    #
    # For now, implement the 8-block approach with schoolbook 3x3.
    # Total rank = 216 (same as naive, but structured).
    
    # The 8 block products:
    # Block (i2, k2) in output, computed from blocks (i2, j2) of A and (j2, k2) of B
    # C[i2, k2] = sum_{j2} A[i2, j2] * B[j2, k2]
    
    # Each 3x3 product uses 27 scalar multiplications (schoolbook).
    # Total: 8 * 27 = 216
    
    rank_total = 8 * 27  # 216
    U = [[0] * rank_total for _ in range(36)]
    V = [[0] * rank_total for _ in range(36)]
    W = [[0] * rank_total for _ in range(36)]
    
    col = 0
    for i2 in range(2):
        for k2 in range(2):
            for j2 in range(2):
                # This block product: A[i2,j2] * B[j2,k2] contributes to C[i2,k2]
                for i3 in range(3):
                    for k3 in range(3):
                        for j3 in range(3):
                            # A[i2*3+i3, j2*3+j3] * B[j2*3+j3, k2*3+k3] -> C[i2*3+i3, k2*3+k3]
                            U[(i2*3+i3)*6 + (j2*3+j3)][col] = 1
                            V[(j2*3+j3)*6 + (k2*3+k3)][col] = 1
                            W[(k2*3+k3)*6 + (i2*3+i3)][col] = 1
                            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
