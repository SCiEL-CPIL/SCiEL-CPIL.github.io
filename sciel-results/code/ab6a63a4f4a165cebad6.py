"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    
    # Identify 6 disjoint 2x2x2 subcubes for Strassen optimization
    # Each subcube: (i_offset, j_offset, k_offset) for 2x2x2 block
    subcubes = [
        (0, 0, 0),  # i={0,1}, j={0,1}, k={0,1}
        (0, 2, 0),  # i={0,1}, j={2,3}, k={0,1}
        (0, 0, 2),  # i={0,1}, j={0,1}, k={2,3}
        (0, 2, 2),  # i={0,1}, j={2,3}, k={2,3}
        (1, 0, 4),  # i={1,2}, j={0,1}, k={4,5}
        (1, 2, 4),  # i={1,2}, j={2,3}, k={4,5}
    ]
    
    # Collect all terms in these subcubes
    subcube_terms = set()
    for (i0, j0, k0) in subcubes:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    subcube_terms.add((i0+di, j0+dj, k0+dk))
    
    # Remaining schoolbook terms
    all_terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    remaining_terms = [t for t in all_terms if t not in subcube_terms]
    
    # Total rank = 7*6 (Strassen) + len(remaining_terms) (schoolbook)
    rank = 7 * len(subcubes) + len(remaining_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for (i0, j0, k0) in subcubes:
        # Apply Strassen's 7-term formula to this 2x2x2 subcube
        # Local indices: i∈{0,1}, j∈{0,1}, k∈{0,1} mapped to global (i0+i, j0+j, k0+k)
        
        # M1 = (A[i0,j0]+A[i0+1,j0+1])(B[j0,k0]+B[j0+1,k0+1])
        u1 = [0]*(n*m); v1 = [0]*(m*p); w1 = [0]*(p*n)
        u1[(i0)*m+(j0)] = 1; u1[(i0+1)*m+(j0+1)] = 1
        v1[(j0)*p+(k0)] = 1; v1[(j0+1)*p+(k0+1)] = 1
        w1[(k0)*n+(i0)] = 1; w1[(k0+1)*n+(i0+1)] = 1
        for r in range(n*m): U[r][col] = u1[r]
        for r in range(m*p): V[r][col] = v1[r]
        for r in range(p*n): W[r][col] = w1[r]
        col += 1
        
        # M2 = (A[i0+1,j0]+A[i0+1,j0+1])B[j0,k0]
        u2 = [0]*(n*m); v2 = [0]*(m*p); w2 = [0]*(p*n)
        u2[(i0+1)*m+(j0)] = 1; u2[(i0+1)*m+(j0+1)] = 1
        v2[(j0)*p+(k0)] = 1
        w2[(k0)*n+(i0+1)] = 1
        for r in range(n*m): U[r][col] = u2[r]
        for r in range(m*p): V[r][col] = v2[r]
        for r in range(p*n): W[r][col] = w2[r]
        col += 1
        
        # M3 = A[i0,j0](B[j0+1,k0]-B[j0+1,k0+1])
        u3 = [0]*(n*m); v3 = [0]*(m*p); w3 = [0]*(p*n)
        u3[(i0)*m+(j0)] = 1
        v3[(j0+1)*p+(k0)] = 1; v3[(j0+1)*p+(k0+1)] = -1
        w3[(k0)*n+(i0)] = 1
        for r in range(n*m): U[r][col] = u3[r]
        for r in range(m*p): V[r][col] = v3[r]
        for r in range(p*n): W[r][col] = w3[r]
        col += 1
        
        # M4 = A[i0+1,j0+1](B[j0,k0+1]-B[j0,k0])
        u4 = [0]*(n*m); v4 = [0]*(m*p); w4 = [0]*(p*n)
        u4[(i0+1)*m+(j0+1)] = 1
        v4[(j0)*p+(k0+1)] = 1; v4[(j0)*p+(k0)] = -1
        w4[(k0+1)*n+(i0+1)] = 1
        for r in range(n*m): U[r][col] = u4[r]
        for r in range(m*p): V[r][col] = v4[r]
        for r in range(p*n): W[r][col] = w4[r]
        col += 1
        
        # M5 = (A[i0,j0]+A[i0,j0+1])B[j0+1,k0+1]
        u5 = [0]*(n*m); v5 = [0]*(m*p); w5 = [0]*(p*n)
        u5[(i0)*m+(j0)] = 1; u5[(i0)*m+(j0+1)] = 1
        v5[(j0+1)*p+(k0+1)] = 1
        w5[(k0+1)*n+(i0)] = 1
        for r in range(n*m): U[r][col] = u5[r]
        for r in range(m*p): V[r][col] = v5[r]
        for r in range(p*n): W[r][col] = w5[r]
        col += 1
        
        # M6 = (A[i0+1,j0]-A[i0,j0])(B[j0,k0]+B[j0,k0+1])
        u6 = [0]*(n*m); v6 = [0]*(m*p); w6 = [0]*(p*n)
        u6[(i0+1)*m+(j0)] = 1; u6[(i0)*m+(j0)] = -1
        v6[(j0)*p+(k0)] = 1; v6[(j0)*p+(k0+1)] = 1
        w6[(k0)*n+(i0+1)] = 1
        for r in range(n*m): U[r][col] = u6[r]
        for r in range(m*p): V[r][col] = v6[r]
        for r in range(p*n): W[r][col] = w6[r]
        col += 1
        
        # M7 = (A[i0,j0+1]-A[i0+1,j0+1])(B[j0+1,k0]+B[j0+1,k0+1])
        u7 = [0]*(n*m); v7 = [0]*(m*p); w7 = [0]*(p*n)
        u7[(i0)*m+(j0+1)] = 1; u7[(i0+1)*m+(j0+1)] = -1
        v7[(j0+1)*p+(k0)] = 1; v7[(j0+1)*p+(k0+1)] = 1
        w7[(k0)*n+(i0)] = 1
        for r in range(n*m): U[r][col] = u7[r]
        for r in range(m*p): V[r][col] = v7[r]
        for r in range(p*n): W[r][col] = w7[r]
        col += 1
    
    # Add remaining schoolbook terms
    for (i, j, k) in remaining_terms:
        U[i*m+j][col] = 1
        V[j*p+k][col] = 1
        W[k*n+i][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
