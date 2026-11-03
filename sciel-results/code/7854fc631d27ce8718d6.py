"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np

def _build_2x3x4_strassen_schoo():
    """
    Build a rank-22 decomposition for 2x3x4.
    Partition j=3 into j=2 (Strassen for k=0,1 and k=2,3) + j=1 (schoolbook for k=0..3).
    """
    n, m, p = 2, 3, 4
    rank = 22
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Strassen for 2x2x2 at j_off=0, k_off=0
    # m1: (a+d)(e+h)
    U[0, col] = 1; U[3, col] = 1  # (0,0) and (1,1) in j-block 0
    V[0, col] = 1; V[3, col] = 1  # (0,0) and (1,1) in k-block 0
    W[0, col] = 1; W[2, col] = 1  # (0,0) and (1,1) in output
    col += 1
    # m2: (c+d)e
    U[2, col] = 1; U[3, col] = 1
    V[0, col] = 1
    W[1, col] = 1; W[3, col] = -1
    col += 1
    # m3: a(f-h)
    U[0, col] = 1
    V[1, col] = 1; V[3, col] = -1
    W[2, col] = 1; W[4, col] = 1  # Wait, W index for k=1, i=0 is 1*2+0=2, k=1,i=1 is 1*2+1=3
    # Let me re-check W indexing: W[k*n + i]
    # For 2x2x2: W indices: (0,0)->0, (1,0)->2, (0,1)->1, (1,1)->3
    # So W[k*n+i]: k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 2; k=1,i=1 -> 3
    # Let me redo Strassen with correct W indices
    # Actually the previous code used W[(k_off+lk)*n + (i_off+li)]
    # So for i_off=0, j_off=0, k_off=0:
    # W[0*2+0] = W[0], W[0*2+1] = W[1], W[1*2+0] = W[2], W[1*2+1] = W[3]
    # This is correct.
    
    # Let me rebuild properly using a helper
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    
    def strassen_2x2x2(i0, j0, k0):
        nonlocal col
        # i0: row offset in A (0 or 1 for the 2x2 block)
        # j0: col offset in A (0 or 1 for the 2x2 block)
        # k0: col offset in B (0 or 1 for the 2x2 block)
        
        # m1: (a+d)(e+h) -> a*e + a*h + d*e + d*h
        U[i0*m + j0, col] = 1      # a
        U[(i0+1)*m + (j0+1), col] = 1  # d
        V[j0*p + k0, col] = 1      # e
        V[(j0+1)*p + (k0+1), col] = 1 # h
        W[k0*n + i0, col] = 1      # (k0, i0)
        W[(k0+1)*n + (i0+1), col] = 1 # (k0+1, i0+1)
        col += 1
        
        # m2: (c+d)e -> c*e + d*e
        U[i0*m + j0, col] = 0
        U[(i0+1)*m + j0, col] = 1  # c
        U[(i0+1)*m + (j0+1), col] = 1  # d
        V[j0*p + k0, col] = 1      # e
        W[k0*n + (i0+1), col] = 1  # (k0, i0+1)
        W[(k0+1)*n + (i0+1), col] = -1 # (k0+1, i0+1)
        col += 1
        
        # m3: a(f-h) -> a*f - a*h
        U[i0*m + j0, col] = 1      # a
        V[(j0+1)*p + k0, col] = 1  # f
        V[(j0+1)*p + (k0+1), col] = -1 # -h
        W[k0*n + i0, col] = 1      # (k0, i0)
        W[(k0+1)*n + i0, col] = -1 # (k0+1, i0)
        col += 1
        
        # m4: d(g-e) -> d*g - d*e
        U[(i0+1)*m + (j0+1), col] = 1  # d
        V[j0*p + (k0+1), col] = 1      # g
        V[j0*p + k0, col] = -1         # -e
        W[k0*n + (i0+1), col] = 1      # (k0, i0+1)
        W[(k0+1)*n + (i0+1), col] = -1 # (k0+1, i0+1)
        col += 1
        
        # m5: (a+b)h -> a*h + b*h
        U[i0*m + j0, col] = 1      # a
        U[i0*m + (j0+1), col] = 1  # b
        V[(j0+1)*p + (k0+1), col] = 1 # h
        W[k0*n + i0, col] = -1     # (k0, i0)
        W[(k0+1)*n + i0, col] = 1  # (k0+1, i0)
        col += 1
        
        # m6: (c-a)(e+f) -> c*e + c*f - a*e - a*f
        U[i0*m + j0, col] = -1     # -a
        U[(i0+1)*m + j0, col] = 1  # c
        V[j0*p + k0, col] = 1      # e
        V[j0*p + (k0+1), col] = 1  # f
        W[(k0+1)*n + (i0+1), col] = 1 # (k0+1, i0+1)
        col += 1
        
        # m7: (b-d)(g+h) -> b*g + b*h - d*g - d*h
        U[i0*m + (j0+1), col] = 1  # b
        U[(i0+1)*m + (j0+1), col] = -1 # -d
        V[(j0+1)*p + k0, col] = 1  # g
        V[(j0+1)*p + (k0+1), col] = 1 # h
        W[k0*n + i0, col] = 1      # (k0, i0)
        col += 1
    
    # For 2x3x4:
    # j=0,1 with k=0,1: strassen
    strassen_2x2x2(0, 0, 0)
    # j=0,1 with k=2,3: strassen
    strassen_2x2x2(0, 0, 2)
    # j=2 with k=0,1,2,3: schoolbook 2x1x4
    for k in range(4):
        U[0*m + 2, col] = 1
        U[1*m + 2, col] = 1
        V[2*p + k, col] = 1
        W[k*n + 0, col] = 1
        W[k*n + 1, col] = 1
        col += 1
    
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    # p=8, split into two p=4 blocks
    # Each block is 2x3x4
    
    U4, V4, W4, rank4 = _build_2x3x4_strassen_schoo()
    
    total_rank = 2 * rank4
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Block 1: k=0..3
    U[:, 0:rank4] = U4
    V[:, 0:rank4] = V4
    W[:, 0:rank4] = W4
    
    # Block 2: k=4..7
    U[:, rank4:2*rank4] = U4
    V[:, rank4:2*rank4] = V4
    # Shift W indices by 4*n = 8
    W_shifted = np.zeros_like(W4)
    W_shifted[:, :] = W4
    # Actually W4 has indices for k=0..3, we need k=4..7
    # W[k*n+i] for k=4..7
    # W4[k*n+i] for k=0..3
    # So we need to remap
    W2 = np.zeros((p*n, rank4))
    for k in range(4):
        for i in range(n):
            W2[(k+4)*n + i, :] = W4[k*n + i, :]
    W[:, rank4:2*rank4] = W2
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
