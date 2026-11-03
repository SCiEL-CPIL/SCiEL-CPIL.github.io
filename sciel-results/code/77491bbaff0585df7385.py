"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe
    
    n, m, p = DIMENSIONS
    # Hybrid: 6 Strassen 2x2x2 blocks + schoolbook remainder = rank 99
    
    blocks = [
        (0, 0, 0), (0, 2, 0), (0, 0, 2),
        (0, 2, 2), (0, 0, 4), (0, 2, 4)
    ]
    
    # Determine which (i,j,k) terms are covered by Strassen
    covered = set()
    for (i0, j0, k0) in blocks:
        for i in (i0, i0+1):
            for j in (j0, j0+1):
                for k in (k0, k0+1):
                    covered.add((i, j, k))
    
    remaining_terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)
                       if (i, j, k) not in covered]
    
    rank = len(blocks) * 7 + len(remaining_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for (i0, j0, k0) in blocks:
        # M1 = (A00+A11)(B00+B11)
        U[i0*m+j0][col] += 1; U[(i0+1)*m+(j0+1)][col] += 1
        V[j0*p+k0][col] += 1; V[(j0+1)*p+(k0+1)][col] += 1
        W[k0*n+i0][col] += 1; W[(k0+1)*n+(i0+1)][col] += 1
        col += 1
        # M2 = (A10+A11)*B00
        U[(i0+1)*m+j0][col] += 1; U[(i0+1)*m+(j0+1)][col] += 1
        V[j0*p+k0][col] += 1
        W[k0*n+(i0+1)][col] += 1; W[(k0+1)*n+(i0+1)][col] -= 1
        col += 1
        # M3 = A00*(B10-B11)
        U[i0*m+j0][col] += 1
        V[j0*p+(k0+1)][col] += 1; V[(j0+1)*p+(k0+1)][col] -= 1
        W[(k0+1)*n+i0][col] += 1; W[(k0+1)*n+(i0+1)][col] += 1
        col += 1
        # M4 = A11*(B01-B00)
        U[(i0+1)*m+(j0+1)][col] += 1
        V[(j0+1)*p+k0][col] += 1; V[j0*p+k0][col] -= 1
        W[k0*n+i0][col] += 1; W[k0*n+(i0+1)][col] += 1
        col += 1
        # M5 = (A00+A01)*B11
        U[i0*m+j0][col] += 1; U[i0*m+(j0+1)][col] += 1
        V[(j0+1)*p+(k0+1)][col] += 1
        W[(k0+1)*n+i0][col] += 1; W[k0*n+i0][col] -= 1
        col += 1
        # M6 = (A10-A00)*(B00+B10)
        U[(i0+1)*m+j0][col] += 1; U[i0*m+j0][col] -= 1
        V[j0*p+k0][col] += 1; V[j0*p+(k0+1)][col] += 1
        W[(k0+1)*n+(i0+1)][col] += 1
        col += 1
        # M7 = (A01-A11)*(B01+B11)
        U[i0*m+(j0+1)][col] += 1; U[(i0+1)*m+(j0+1)][col] -= 1
        V[(j0+1)*p+k0][col] += 1; V[(j0+1)*p+(k0+1)][col] += 1
        W[k0*n+i0][col] += 1
        col += 1
    
    for (i, j, k) in remaining_terms:
        U[i*m+j][col] = 1
        V[j*p+k][col] = 1
        W[k*n+i][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
