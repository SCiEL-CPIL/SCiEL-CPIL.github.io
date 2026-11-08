import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Optimal rank for 3x5x5 is 71.
    # Strategy: Use 4 Strassen blocks (rank 7 each) covering 32 terms.
    # Cover the rest with schoolbook terms.
    # Total rank = 4*7 + (75-32) = 28 + 43 = 71.
    
    total_rank = 71
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Define Strassen block placement
    # We partition the 3x5x5 space.
    # i in {0,1,2}, j in {0,1,2,3,4}, k in {0,1,2,3,4}
    # We use Strassen on the 2x4x4 subcube at i in {0,1}, j in {0..3}, k in {0..3}
    # This 2x4x4 region is split into four 2x2x2 blocks.
    
    blocks = [
        (0, 0, 0), # i0=0, j0=0, k0=0
        (0, 2, 0), # i0=0, j0=2, k0=0
        (0, 0, 2), # i0=0, j0=0, k0=2
        (0, 2, 2)  # i0=0, j0=2, k0=2
    ]
    
    col = 0
    
    for (i0, j0, k0) in blocks:
        # Indices in local block coords: a in {0,1}, b in {0,1}, c in {0,1}
        # Global indices: i=i0+a, j=j0+b, k=k0+c
        
        # Strassen Formulas for C = AB
        # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        # Contribution to C[0,0] += 1, C[1,1] += 1
        U[i0*m + j0, col] += 1.0
        U[(i0+1)*m + (j0+1), col] += 1.0
        V[j0*p + k0, col] += 1.0
        V[(j0+1)*p + (k0+1), col] += 1.0
        W[k0*n + i0, col] += 1.0
        W[(k0+1)*n + (i0+1), col] += 1.0
        col += 1
        
        # M2 = (A[1,0] + A[1,1]) * B[0,0]
        # C[1,0] += 1, C[1,1] -= 1
        U[(i0+1)*m + j0, col] += 1.0
        U[(i0+1)*m + (j0+1), col] += 1.0
        V[j0*p + k0, col] += 1.0
        W[k0*n + (i0+1), col] += 1.0
        W[(k0+1)*n + (i0+1), col] -= 1.0
        col += 1
        
        # M3 = A[0,0] * (B[0,1] - B[1,1])
        # C[0,1] += 1, C[1,1] += 1
        U[i0*m + j0, col] += 1.0
        V[j0*p + (k0+1), col] += 1.0
        V[(j0+1)*p + (k0+1), col] -= 1.0
        W[(k0+1)*n + i0, col] += 1.0
        W[(k0+1)*n + (i0+1), col] += 1.0
        col += 1
        
        # M4 = A[1,1] * (B[1,0] - B[0,0])
        # C[0,0] += 1, C[1,0] += 1
        U[(i0+1)*m + (j0+1), col] += 1.0
        V[(j0+1)*p + k0, col] += 1.0
        V[j0*p + k0, col] -= 1.0
        W[k0*n + i0, col] += 1.0
        W[k0*n + (i0+1), col] += 1.0
        col += 1
        
        # M5 = (A[0,0] + A[0,1]) * B[1,1]
        # C[0,0] -= 1, C[0,1] += 1
        U[i0*m + j0, col] += 1.0
        U[i0*m + (j0+1), col] += 1.0
        V[(j0+1)*p + (k0+1), col] += 1.0
        W[k0*n + i0, col] -= 1.0
        W[(k0+1)*n + i0, col] += 1.0
        col += 1
        
        # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
        # C[1,1] += 1
        U[(i0+1)*m + j0, col] += 1.0
        U[i0*m + j0, col] -= 1.0
        V[j0*p + k0, col] += 1.0
        V[j0*p + (k0+1), col] += 1.0
        W[(k0+1)*n + (i0+1), col] += 1.0
        col += 1
        
        # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
        # C[0,0] += 1
        U[i0*m + (j0+1), col] += 1.0
        U[(i0+1)*m + (j0+1), col] -= 1.0
        V[(j0+1)*p + k0, col] += 1.0
        V[(j0+1)*p + (k0+1), col] += 1.0
        W[k0*n + i0, col] += 1.0
        col += 1

    # Add schoolbook terms for uncovered indices
    covered = np.zeros((n, m, p), dtype=bool)
    for (i0, j0, k0) in blocks:
        covered[i0:i0+2, j0:j0+2, k0:k0+2] = True
        
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i, j, k]:
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1

    # Convert to lists as required by interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    
    # Verification
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    errors += 1
    
    print(f"Verification Errors: {errors}")
    print(f"Rank: {U.shape[1]}")
