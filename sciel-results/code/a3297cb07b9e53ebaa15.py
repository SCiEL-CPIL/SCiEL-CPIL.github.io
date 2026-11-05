import numpy as np

def construct():
    n, m, p = 2, 3, 10
    rank = 55
    
    # U: (n*m, rank) -> (6, 55)
    # V: (m*p, rank) -> (30, 55)
    # W: (p*n, rank) -> (20, 55)
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    r = 0
    
    # Helper to add a Strassen block
    # i0: start row of A (0)
    # j0: start col of A / row of B (0)
    # k0: start col of B
    def add_strassen(i0, j0, k0):
        nonlocal r
        # A indices: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
        # B indices: (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
        # C indices: (k0, i0), (k0, i0+1), (k0+1, i0), (k0+1, i0+1)
        # Note: W index is (k, i)
        
        # P1: (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1])
        # U
        U[i0*m + j0, r] += 1
        U[(i0+1)*m + (j0+1), r] += 1
        # V
        V[j0*p + k0, r] += 1
        V[(j0+1)*p + (k0+1), r] += 1
        # W
        # C[i0, k0] gets +1? 
        # C[i0, k0] = P1 + P4 - P5 + P7
        # C[i0, k0+1] = P3 + P5
        # C[i0+1, k0] = P2 + P4
        # C[i0+1, k0+1] = P1 - P2 + P3 + P6
        
        W[k0*n + i0, r] += 1
        W[(k0+1)*n + (i0+1), r] += 1
        r += 1
        
        # P2: (A[i0+1,j0] + A[i0+1,j0+1]) * B[j0,k0]
        U[i0+1, j0] = 0 # reset? No, we are using +=
        # Wait, I need to be careful with nonlocal r and zeroing.
        # The arrays are pre-allocated zeros. I am adding to them.
        # So I just need to set the non-zero entries for this rank r.
        
        # P2
        U[(i0+1)*m + j0, r] += 1
        U[(i0+1)*m + (j0+1), r] += 1
        V[j0*p + k0, r] += 1
        # W
        # C[i0+1, k0] gets +1
        W[k0*n + (i0+1), r] += 1
        r += 1
        
        # P3: A[i0,j0] * (B[j0,k0+1] - B[j0+1,k0+1])
        U[i0*m + j0, r] += 1
        V[j0*p + (k0+1), r] += 1
        V[(j0+1)*p + (k0+1), r] += -1
        # W
        # C[i0, k0+1] gets +1
        W[(k0+1)*n + i0, r] += 1
        r += 1
        
        # P4: A[i0+1,j0+1] * (B[j0+1,k0] - B[j0,k0])
        U[(i0+1)*m + (j0+1), r] += 1
        V[(j0+1)*p + k0, r] += 1
        V[j0*p + k0, r] += -1
        # W
        # C[i0, k0] gets +1
        W[k0*n + i0, r] += 1
        # C[i0+1, k0] gets +1
        W[k0*n + (i0+1), r] += 1
        r += 1
        
        # P5: (A[i0,j0] + A[i0,j0+1]) * B[j0+1,k0+1]
        U[i0*m + j0, r] += 1
        U[i0*m + (j0+1), r] += 1
        V[(j0+1)*p + (k0+1), r] += 1
        # W
        # C[i0, k0] gets -1
        W[k0*n + i0, r] += -1
        # C[i0, k0+1] gets +1
        W[(k0+1)*n + i0, r] += 1
        r += 1
        
        # P6: (A[i0+1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k0+1])
        U[(i0+1)*m + j0, r] += 1
        U[i0*m + j0, r] += -1
        V[j0*p + k0, r] += 1
        V[j0*p + (k0+1), r] += 1
        # W
        # C[i0+1, k0+1] gets +1
        W[(k0+1)*n + (i0+1), r] += 1
        r += 1
        
        # P7: (A[i0,j0+1] - A[i0+1,j0+1]) * (B[j0+1,k0] + B[j0+1,k0+1])
        U[i0*m + (j0+1), r] += 1
        U[(i0+1)*m + (j0+1), r] += -1
        V[(j0+1)*p + k0, r] += 1
        V[(j0+1)*p + (k0+1), r] += 1
        # W
        # C[i0, k0] gets +1
        W[k0*n + i0, r] += 1
        r += 1
        
        return r

    # Apply Strassen for j=0,1 blocks
    # A cols 0,1. B rows 0,1.
    for b in range(5):
        k0 = 2 * b
        r = add_strassen(0, 0, k0)
        
    # Apply Schoolbook for j=2
    # A col 2. B row 2.
    for i in range(n):
        for k in range(p):
            U[i*m + 2, r] = 1
            V[2*p + k, r] = 1
            W[k*n + i, r] = 1
            r += 1
            
    return {'U': U, 'V': V, 'W': W}
