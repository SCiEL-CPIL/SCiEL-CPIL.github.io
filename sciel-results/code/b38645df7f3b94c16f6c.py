"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Block-circulant embedding strategy:
    # Embed 3x5x7 into a structure using cyclic convolution with N=4.
    # Since 5 = 4+1 and 7 = 4+3, we handle the base-4 digits separately.
    # Use the 4-point DFT which has real parts in {0, ±1} (integers).
    
    # The 4-point DFT matrix F has entries:
    # F[a,b] = omega^(a*b) where omega = i = exp(2*pi*i/4)
    # F = [[1, 1, 1, 1],
    #      [1, i, -1, -i],
    #      [1, -1, 1, -1],
    #      [1, -i, -1, i]]
    # Real parts: [[1,1,1,1],[1,0,-1,0],[1,-1,1,-1],[1,0,-1,0]]
    # This has integer entries!
    
    # Strategy: Decompose the tensor using the 4-point DFT structure.
    # For the j and k dimensions, we use the fact that the multiplication
    # can be related to cyclic convolution when embedded properly.
    
    # Use N=4 for the DFT. The key identity:
    # sum_{b=0}^{3} omega^(a*b) * omega^(-c*b) = 4 * delta_{a,c}
    
    # We'll construct rank-1 terms using the DFT basis.
    # For each frequency f in [0,4), and for appropriate i, we get a term.
    
    # But we need to handle the fact that m=5 and p=7 are larger than N=4.
    # We'll use multiple DFT blocks: split j into j0 + 4*j1 and k into k0 + 4*k1.
    
    # Actually, let's use a simpler approach: use the DFT to create orthogonal
    # rank-1 terms and combine them with the i dimension.
    
    # Precompute the 4-point DFT real parts
    # omega = i, so omega^(a*b) for a,b in 0..3:
    # a=0: [1, 1, 1, 1]
    # a=1: [1, i, -1, -i] -> real: [1, 0, -1, 0]
    # a=2: [1, -1, 1, -1]
    # a=3: [1, -i, -1, i] -> real: [1, 0, -1, 0]
    
    # Use the full complex DFT but we'll work with real and imaginary parts
    # to get integer entries.
    
    # The 4-point DFT matrix (complex):
    F4 = np.array([
        [1, 1, 1, 1],
        [1, 1j, -1, -1j],
        [1, -1, 1, -1],
        [1, -1j, -1, 1j]
    ], dtype=complex)
    
    # Inverse DFT: (1/4) * F4^H
    # F4 is unitary up to scaling: F4^H @ F4 = 4*I
    
    # For the tensor T[i,j,k] = 1 for i<3, j<5, k<7:
    # We can write j = 4*j1 + j0 where j0 in [0,4), j1 in [0,1) (since 5 = 4+1)
    # k = 4*k1 + k0 where k0 in [0,4), k1 in [0,2) (since 7 = 4+3)
    
    # The multiplication sum over j: sum_{j=0}^{4} A[i,j] * B[j,k]
    # = sum_{j0=0}^{3} sum_{j1=0}^{1} A[i, 4*j1+j0] * B[4*j1+j0, k]
    
    # This doesn't directly give a convolution structure.
    
    # Alternative: Use the fact that we can embed the 5x7 multiplication
    # into an 8x8 multiplication and use the 8-point DFT, but that has √2.
    
    # Let's use a different strategy: direct construction with grouping.
    # Group the (j,k) pairs by (j mod 4, k mod 4) and use the DFT for each group.
    
    # For each (j0, k0) in [0,4) x [0,4), and for each (j1, k1) such that
    # j = 4*j1 + j0 < 5 and k = 4*k1 + k0 < 7, we have a term.
    
    # The number of valid (j1,k1) pairs for each (j0,k0):
    # j0=0: j1=0 (j=0), j1=1 (j=4) -> 2 values
    # j0=1: j1=0 (j=1) -> 1 value
    # j0=2: j1=0 (j=2) -> 1 value
    # j0=3: j1=0 (j=3) -> 1 value
    
    # k0=0: k1=0 (k=0), k1=1 (k=4) -> 2 values
    # k0=1: k1=0 (k=1), k1=1 (k=5) -> 2 values
    # k0=2: k1=0 (k=2), k1=1 (k=6) -> 2 values
    # k0=3: k1=0 (k=3) -> 1 value
    
    # Total terms: sum over j0,k0 of (num_j1(j0) * num_k1(k0))
    # = (2*2 + 1*2 + 1*2 + 1*1) for k0=0,1,2,3 respectively
    # Wait, I need to be more careful.
    
    # Actually, let's just use a verified lower-rank construction.
    # I'll use the following: decompose using the Kronecker product structure.
    
    # For 3x5x7, use the fact that 3 = 1+2, 5 = 2+3, 7 = 3+4.
    # This gives 8 sub-multiplications:
    # (1x2) x (2x3), (1x2) x (2x4), (1x3) x (3x3), (1x3) x (3x4)
    # (2x2) x (2x3), (2x2) x (2x4), (2x3) x (3x3), (2x3) x (3x4)
    
    # For each sub-multiplication, use the minimal known rank:
    # 1x2x3: rank 2 (schoolbook: 1*2*3=6, but we can do better)
    # Actually, for 1xm x pxq, the rank is at most m*p, but can be less.
    
    # This is getting too complex. Let me use a clean construction.
    
    # Use the following: the rank of 3x5x7 is at most 3*5*7 = 105.
    # I'll implement a construction that achieves rank < 105 using
    # the DFT with N=4 and careful handling of the extra dimensions.
    
    # For now, return a correct schoolbook decomposition.
    # But let me try to optimize by using the DFT structure.
    
    # Final approach: Use the 4-point DFT to create 4 orthogonal rank-1 terms
    # for each "block" of the multiplication.
    
    N = 4
    # DFT matrix real parts (using the fact that we can separate real/imag)
    # Actually, let's use the full complex DFT and take real parts where possible.
    
    # I'll use a construction that's guaranteed to work:
    # rank = n * m * p = 105, but with a structured pattern.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
