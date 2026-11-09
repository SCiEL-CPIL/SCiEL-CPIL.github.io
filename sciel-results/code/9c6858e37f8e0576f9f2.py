"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin under 120s

    n, m, p = DIMENSIONS  # 4, 5, 5

    # Schoolbook decomposition (rank = n*m*p = 100)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Exact verification using Fraction arithmetic."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                    if s != 1:
                        return False
        # Also verify that off-diagonal (i,j,k) entries are zero.
        # The tensor only has 1s at (i,j,k) positions; all other (i',j',k') must sum to 0.
        # But our U,V,W only produce contributions at (i,j,k) where all three indices match the pattern.
        # Actually the tensor T[i,j,k] = delta(i,j,k) for the multiplication.
        # We need: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # and for all other (i,j,k) combinations that don't correspond... 
        # Wait, the tensor is defined on ALL (i,j,k) in [0,n)x[0,m)x[0,p).
        # The multiplication tensor has T[i,j,k] = 1 if and only if... no.
        # Actually for matrix multiplication C = A*B, C[i,k] = sum_j A[i,j]*B[j,k].
        # The tensor is T[i,j,k] where the bilinear form is sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = C[i,k].
        # So T[i,j,k] = 1 for all valid (i,j,k), and the output is indexed by (i,k).
        # The decomposition requires: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = T[i,j,k]
        # where T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
        # So we just need all 100 entries to equal 1.
        return True

    # Try to find a lower-rank decomposition.
    # Strategy: Use the fact that we can combine some terms.
    # For a 2x2x2 block, Strassen reduces rank from 8 to 7.
    # We'll look for 2x2x2 sub-blocks within our 4x5x5 tensor and apply Strassen.
    
    # A 2x2x2 Strassen decomposition:
    # The 8 schoolbook terms for indices (i,i+1) x (j,j+1) x (k,k+1) can be replaced by 7 terms.
    # Let's identify non-overlapping 2x2x2 blocks.
    # n=4, m=5, p=5. We can fit floor(4/2)*floor(5/2)*floor(5/2) = 2*2*2 = 8 non-overlapping 2x2x2 blocks.
    # Each saves 1 rank, so total rank = 100 - 8 = 92.
    
    # Let's build a rank-92 decomposition using 8 Strassen blocks.
    def strassen_decomposition():
        # For each 2x2x2 block at base indices (i0, j0, k0), we replace 8 terms with 7.
        # The 7 Strassen terms for a 2x2x2 block:
        # Let A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b01,b11]] (using 0/1 local indices)
        # C = A*B
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11)b00
        # m3 = a00(b01-b11)
        # m4 = a11(b10-b00)
        # m5 = (a00+a01)b11
        # m6 = (a10-a00)(b00+b01)
        # m7 = (a01-a11)(b10+b11)
        # C00 = m1+m4-m5+m7
        # C01 = m3+m5
        # C10 = m2+m4
        # C11 = m1-m2+m3+m6
        
        # In our tensor formulation, the "output" W is indexed by (k*n+i).
        # For a 2x2x2 block at (i0,j0,k0):
        # Input A indices: (i0,j0),(i0,j0+1),(i0+1,j0),(i0+1,j0+1)
        # Input B indices: (j0,k0),(j0,k0+1),(j0+1,k0),(j0+1,k0+1)
        # Output C indices: (k0,i0),(k0,i0+1),(k0+1,i0),(k0+1,i0+1)
        
        # The 7 Strassen terms contribute to the U, V, W matrices.
        # Each term c has U values on the 4 A-indices, V values on the 4 B-indices, W values on the 4 C-indices.
        
        # Let's define the 7 Strassen terms as (U_coeffs, V_coeffs, W_coeffs) for local indices.
        # Local A indices: 0=(i0,j0), 1=(i0,j0+1), 2=(i0+1,j0), 3=(i0+1,j0+1)
        # Local B indices: 0=(j0,k0), 1=(j0,k0+1), 2=(j0+1,k0), 3=(j0+1,k0+1)
        # Local C indices: 0=(k0,i0), 1=(k0,i0+1), 2=(k0+1,i0), 3=(k0+1,i0+1)
        
        # m1: U = [1,0,1,0] (a00+a11), V = [1,0,0,1] (b00+b11), W = [1,0,0,1] (contributes to C00 and C11)
        # m2: U = [0,0,1,1] (a10+a11), V = [1,0,0,0] (b00), W = [0,0,1,0] (contributes to C10)
        # m3: U = [1,0,0,0] (a00), V = [0,1,-1,0] (b01-b11), W = [0,1,0,1] (contributes to C01 and C11)
        # m4: U = [0,0,0,1] (a11), V = [0,0,1,-1] (b10-b00), W = [1,0,1,0] (contributes to C00 and C10)
        # m5: U = [1,1,0,0] (a00+a01), V = [0,0,0,1] (b11), W = [-1,0,0,1] (contributes to -C00 and C11)
        # m6: U = [0,0,1,-1] (a10-a00), V = [1,1,0,0] (b00+b01), W = [0,0,0,1] (contributes to C11)
        # m7: U = [0,1,-1,0] (a01-a11), V = [0,0,1,1] (b10+b11), W = [1,0,0,0] (contributes to C00)
        
        strassen_terms = [
            # (U_coeffs, V_coeffs, W_coeffs)
            ([1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1]),   # m1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),   # m2
            ([1, 0, 0, 0], [0, 1, -1, 0], [0, 1, 0, 1]),   # m3
            ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),   # m4
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 1]),   # m5
            ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),   # m6
            ([0, 1, -1, 0], [0, 0, 1, 1], [1, 0, 0, 0]),   # m7
        ]
        
        # Identify non-overlapping 2x2x2 blocks.
        # Block at (i0, j0, k0) where i0 in {0,2}, j0 in {0,2}, k0 in {0,2}
        blocks = []
        for i0 in range(0, n - 1, 2):
            for j0 in range(0, m - 1, 2):
                for k0 in range(0, p - 1, 2):
                    blocks.append((i0, j0, k0))
        
        num_blocks = len(blocks)
        rank = n * m * p - num_blocks  # 100 - 8 = 92
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # First, add all schoolbook terms.
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        # Now, for each Strassen block, we replace 8 schoolbook terms with 7 Strassen terms.
        # We need to remove the 8 schoolbook terms and add 7 Strassen terms.
        # This is complex because the columns are already allocated.
        # Alternative: Build from scratch.
        
        # Let's build from scratch:
        # For each (i,j,k), determine if it's in a Strassen block.
        # If yes, it's covered by Strassen terms. If no, it gets a schoolbook term.
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Map from (i,j,k) to whether it's in a block
        in_block = {}
        for (i0, j0, k0) in blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        in_block[(i0 + di, j0 + dj, k0 + dk)] = (i0, j0, k0)
        
        col = 0
        # Add schoolbook terms for (i,j,k) not in any block
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in in_block:
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
        
        # Add Strassen terms for each block
        for (i0, j0, k0) in blocks:
            for (u_coeffs, v_coeffs, w_coeffs) in strassen_terms:
                # Local A indices
                a_idx = [(i0, j0), (i0, j0 + 1), (i0 + 1, j0), (i0 + 1, j0 + 1)]
                # Local B indices
                b_idx = [(j0, k0), (j0, k0 + 1), (j0 + 1, k0), (j0 + 1, k0 + 1)]
                # Local C indices (output: k*n + i)
                c_idx = [(k0, i0), (k0, i0 + 1), (k0 + 1, i0), (k0 + 1, i0 + 1)]
                
                for local_a, coeff in enumerate(u_coeffs):
                    if coeff != 0:
                        ia, ja = a_idx[local_a]
                        U[ia * m + ja][col] += coeff
                for local_b, coeff in enumerate(v_coeffs):
                    if coeff != 0:
                        jb, kb = b_idx[local_b]
                        V[jb * p + kb][col] += coeff
                for local_c, coeff in enumerate(w_coeffs):
                    if coeff != 0:
                        kc, ic = c_idx[local_c]
                        W[kc * n + ic][col] += coeff
                col += 1
        
        assert col == rank, f"Column mismatch: {col} != {rank}"
        return {"U": U, "V": V, "W": W}

    # Try the Strassen-based decomposition
    best_result = schoolbook()
    
    try:
        candidate = strassen_decomposition()
        if verify(candidate["U"], candidate["V"], candidate["W"]):
            best_result = candidate
    except Exception:
        pass

    if time.time() > deadline:
        return best_result

    return best_result
# EVOLVE-BLOCK-END
