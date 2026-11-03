"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def _validate(U, V, W, n, m, p):
    """Check that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k."""
    R = len(U[0]) if U else 0
    if len(V[0]) != R or len(W[0]) != R:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def construct():
    t0 = time.time()
    DEADLINE = 140.0

    n, m, p = DIMENSIONS  # 3, 3, 4

    # Schoolbook fallback: rank 36
    def schoolbook():
        R = n * m * p
        U = [[0]*R for _ in range(n*m)]
        V = [[0]*R for _ in range(m*p)]
        W = [[0]*R for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()

    # Try WFTA-style approach: use finite field / cyclic group structure
    # For 3x3x4, we can try to embed in a larger structure.
    # 
    # Key idea: The 3x3 matmul can be done with rank 21 (Winograd 3x3x3).
    # For 3x3x4, we need rank at least 36 (trivial) but potentially less.
    #
    # Let's try a hybrid: use Winograd for the 3x3x3 sub-problem (i,j,k in {0,1,2})
    # and handle the 4th k-column separately.
    #
    # Winograd 3x3x3 uses rank 21 (vs schoolbook 27).
    # For the 4th k-value: we have 3*3 = 9 additional (i,j,k=3) entries.
    # Total rank = 21 + 9 = 30 (if we can cleanly separate).
    #
    # Actually, let's think more carefully. The tensor T[i,j,k] = 1 for all i,j,k.
    # If we decompose k into {0,1,2} and {3}, we get:
    #   For k in {0,1,2}: 3x3x3 matmul tensor (rank 21 with Winograd)
    #   For k = 3: 3x3x1 "matmul" which is just 9 scalar multiplications (rank 9)
    # Total: 21 + 9 = 30.
    #
    # Let's construct the Winograd 3x3x3 decomposition.
    # Winograd's algorithm for 3x3 uses the Fourier transform over Z/5Z or similar.
    # The standard Winograd 3x3x3 decomposition has rank 21.
    #
    # The Winograd 3x3x3 decomposition:
    # Uses a 5-point FFT-like structure. The key matrices are derived from
    # the cyclic convolution structure.
    #
    # Let me use a known Winograd 3x3x3 rank-21 decomposition.
    # The decomposition is based on the fact that 3x3 matmul can be viewed
    # as a cyclic convolution of length 5 (with appropriate zero-padding).
    #
    # Actually, let me just try to construct it numerically.
    # The Winograd 3x3x3 tensor decomposition uses 21 rank-1 terms.
    # Each term corresponds to a "multiplication" in the transformed domain.
    #
    # Let me use a different approach: try to find a rank-21 decomposition
    # for 3x3x3 using linear algebra / numerical methods.

    # For now, let's try a simpler approach: use Strassen-style decomposition
    # for parts of the problem.
    
    # Approach: Split the 3x3x4 into blocks.
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> 2x2x2 (Strassen rank 7)
    # Remaining: 36 - 8 = 28 entries, handled by schoolbook.
    # Total: 7 + 28 = 35.

    # Let's try to be smarter. Use Winograd for 3x3x3 (rank 21) + 9 for k=3.
    # Total: 30.

    # Construct Winograd 3x3x3 decomposition (rank 21).
    # The Winograd 3x3x3 uses the following structure:
    # It's based on the DFT of length 5 over the rationals (or a number field).
    # The 5 roots of unity are used to diagonalize the cyclic convolution.
    
    # Let me use a known explicit Winograd 3x3x3 decomposition.
    # From the literature, the rank-21 decomposition for 3x3x3 matmul:
    
    # The Winograd 3x3x3 decomposition can be constructed as follows:
    # We view the 3x3 matmul as computing C = A*B where A is 3x3, B is 3x3.
    # Using the Winograd scheme, we:
    # 1. Apply input transforms to A and B (linear combinations)
    # 2. Perform 21 pointwise multiplications
    # 3. Apply output transform to get C
    
    # The input transforms map 3x3 -> 5x5 (with padding), then DFT-5.
    # But for the tensor decomposition, we need the explicit rank-1 terms.
    
    # Let me construct it using the Fourier basis over a suitable field.
    # For 3x3x3, the Winograd algorithm uses the 5-point FFT.
    # The 5th roots of unity: 1, ω, ω², ω³, ω⁴ where ω = e^(2πi/5).
    # But we need rational/integer coefficients, so we work over Q(√5).
    
    # Actually, let me just try a numerical approach: 
    # Set up the tensor equation and solve for a low-rank decomposition.
    
    # The 3x3x3 matmul tensor has 27 entries, all equal to 1.
    # We want to find U (3x3 x R), V (3x3 x R), W (3x3 x R) such that
    # sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] = 1 for all i,j,k.
    
    # Let me try R=21 and solve numerically.
    
    def try_winograd_333():
        """Try to construct a rank-21 decomposition for 3x3x3."""
        n3, m3, p3 = 3, 3, 3
        R = 21
        
        # Initialize with random values and use iterative optimization
        # This is a non-convex problem, so we'll use a simple approach.
        
        # Actually, let me use the known Winograd 3x3x3 decomposition.
        # The Winograd 3x3x3 uses the following 21 multiplications:
        
        # The key insight is that 3x3 matmul can be reduced to 21 multiplications
        # using the Winograd Fourier Transform Algorithm.
        
        # Let me construct it using the cyclic group structure.
        # Z/5Z has 5 elements. The DFT-5 diagonalizes cyclic convolutions.
        
        # For 3x3 matmul, we embed into a 5x5 cyclic convolution.
        # A (3x3) is embedded into a 5x5 matrix by placing it in positions
        # (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2).
        # Similarly for B and C.
        
        # The cyclic convolution of two 5x5 matrices (with appropriate structure)
        # can be diagonalized by DFT-5 in each dimension.
        
        # This gives 5x5 = 25 potential multiplications, but due to symmetry
        # and the specific structure, only 21 are needed.
        
        # Let me use a different known decomposition. 
        # The Bini-Capovani-Lotti 3x3x3 decomposition has rank 21.
        
        # Actually, I recall that the minimal rank for 3x3x3 is 19 (over R) 
        # but 21 is the Winograd rank. Let me just use 21.
        
        # Let me try to construct it numerically using optimization.
        # Since this is complex, let me fall back to a simpler approach.
        
        return None  # Skip for now
    
    # Let me try a different approach: use the known rank-21 Winograd 3x3x3
    # by explicitly constructing the Fourier-based decomposition.
    
    # The Winograd 3x3x3 decomposition:
    # We use the 5-point DFT. The 5th roots of unity are:
    # ζ⁰ = 1
    # ζ¹ = (-1 + √5)/2 * i ... actually let me use the algebraic form.
    
    # For the tensor decomposition, we need U, V, W matrices.
    # Let me use the following approach:
    
    # The 3x3 matmul tensor T[i,j,k] = 1 for i,j,k in {0,1,2}.
    # We want T = sum_r u_r ⊗ v_r ⊗ w_r where u_r, v_r, w_r are 3x3 matrices.
    
    # Using the Winograd scheme:
    # Define the "extended" matrices of size 5x5 by padding with zeros.
    # A_ext[i,j] = A[i,j] for i,j in {0,1,2}, else 0.
    # Similarly for B_ext and C_ext.
    
    # The matmul C = A*B corresponds to a specific cyclic convolution
    # of the extended matrices.
    
    # The DFT-5 in each dimension diagonalizes this convolution.
    # The 25 DFT components give 25 potential multiplications.
    # Due to the structure, 4 of them are redundant or zero, leaving 21.
    
    # Let me construct the DFT-5 matrix.
    # DFT-5: F[t,s] = ζ^(-t*s) where ζ = e^(2πi/5)
    # ζ = cos(2π/5) + i*sin(2π/5) = (1+√5)/4 * ... 
    # Actually ζ = (-1+√5)/2 * ... let me just use the explicit values.
    
    # cos(2π/5) = (1+√5)/4 ≈ 0.809
    # sin(2π/5) = √(10-2√5)/4 ≈ 0.588
    
    # For rational coefficients, we work over Q(√5).
    # Let φ = (1+√5)/2 (golden ratio).
    # ζ = cos(2π/5) + i*sin(2π/5)
    # cos(2π/5) = (φ-1)/2 = 1/(2φ)
    # sin(2π/5) = √(4-1/φ²)/2 ... this is getting complicated.
    
    # Let me just use numerical values and round to nearest half-integer.
    
    import numpy as np
    
    # 5-point DFT matrix
    N = 5
    t = np.arange(N)
    F = np.zeros((N, N), dtype=complex)
    for a in range(N):
        for b in range(N):
            F[a, b] = np.exp(-2j * np.pi * a * b / N)
    
    # The Winograd 3x3x3 decomposition:
    # We embed 3x3 into 5x5, apply DFT-5 in both dimensions.
    # The 25 frequency components correspond to 25 multiplications.
    # But we only need 21 because of the specific structure.
    
    # Actually, I think the correct count is:
    # The 3x3 matmul tensor has rank 21 over the reals (Winograd).
    # The 25 DFT components include 4 that are identically zero 
    # due to the zero-padding structure.
    
    # Let me just construct the decomposition directly.
    
    # For each frequency component (a, b) in {0,1,2,3,4}²:
    # U[a,b] = F[a,i] * F[b,j] for the A matrix
    # V[a,b] = F[a,j] * F[b,k] for the B matrix  (note: different indexing)
    # W[a,b] = F[a,k] * F[b,i] for the C matrix
    
    # Wait, I need to be more careful with the indexing.
    
    # Let me use a simpler approach: just try to find any decomposition
    # with rank < 36 using numerical optimization.
    
    # Actually, let me just use the Strassen-based rank-35 approach
    # which I know works, and try to improve from there.
    
    # For now, return the rank-35 solution.
    total_rank = 35
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col = 0
    
    # Strassen 2x2x2 block: i in {0,1}, j in {0,1}, k in {0,1}
    strassen_U = [
        [[1,0],[0,1]],
        [[0,0],[1,1]],
        [[1,0],[0,0]],
        [[0,0],[0,1]],
        [[1,1],[0,0]],
        [[-1,0],[1,0]],
        [[0,1],[-1,0]],
    ]
    strassen_V = [
        [[1,0],[0,1]],
        [[1,0],[0,0]],
        [[0,0],[1,-1]],
        [[0,1],[-1,0]],
        [[0,0],[0,1]],
        [[1,0],[1,0]],
        [[0,1],[0,1]],
    ]
    strassen_W = [
        [[1,1],[1,1]],
        [[0,0],[1,0]],
        [[0,0],[1,0]],
        [[0,1],[0,0]],
        [[0,1],[0,0]],
        [[0,0],[1,0]],
        [[0,1],[0,0]],
    ]
    
    # Verify Strassen
    valid = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0.0
                for r in range(7):
                    s += strassen_U[r][i][j] * strassen_V[r][j][k] * strassen_W[r][i][k]
                if abs(s - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*m + j][col] = strassen_U[r][i][j]
                for k in range(2):
                    W[k*n + i][col] = strassen_W[r][i][k]
            for j in range(2):
                for k in range(2):
                    V[j*p + k][col] = strassen_V[r][j][k]
            col += 1
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i < 2 and j < 2 and k < 2:
                        continue
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        
        if _validate(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return best

# EVOLVE-BLOCK-END
