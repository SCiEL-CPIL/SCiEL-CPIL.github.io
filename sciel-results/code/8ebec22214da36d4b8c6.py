import time
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U is n*m x rank
    # V is m*p x rank
    # W is p*n x rank
    
    # We will build the matrices dynamically.
    # First, determine the total rank.
    # Block 1: 2x1x7 -> 14
    # Block 2: 2x2x2 -> 7
    # Block 3: 2x2x2 -> 7
    # Block 4: 2x2x1 -> 4
    # Block 5: 2x2x2 -> 7
    rank = 14 + 7 + 7 + 4 + 7
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val_u, val_v, val_w):
        nonlocal col
        U[u_idx][col] += val_u
        V[v_idx][col] += val_v
        W[w_idx][col] += val_w
        col += 1

    # Helper to add Strassen 2x2x2 block
    # i0, i1: row indices in A
    # j0, j1: col indices in A (row indices in B)
    # k0, k1: col indices in B
    def add_strassen(i0, i1, j0, j1, k0, k1):
        nonlocal col
        # M1 = (A[i0,j0] + A[i1,j1])(B[j0,k0] + B[j1,k1])
        add_term(i0*m+j0, j0*p+k0, k0*n+i0, 1, 1, 1)
        add_term(i1*m+j1, j1*p+k1, k0*n+i0, 1, 1, 1)
        add_term(i0*m+j0, j0*p+k1, k1*n+i0, 1, 1, 1)
        add_term(i1*m+j1, j1*p+k0, k1*n+i0, 1, 1, 1)
        # Wait, Strassen sums are added to W.
        # The above adds 4 terms for M1? No.
        # M1 is one product. It contributes to multiple C entries.
        # My add_term adds ONE product to ONE C entry?
        # No, the tensor decomposition is C = sum U V W.
        # Each column c corresponds to one product P_c = (U_c . A) (V_c . B).
        # This product P_c is added to specific entries of C via W_c.
        # So for each Strassen product M_r, we create ONE column.
        # In that column, U has the coefficients for A, V has coefficients for B.
        # W has coefficients for C.
        
        # Let's redo add_strassen logic.
        
        # M1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        # U: A[i0,j0]=1, A[i1,j1]=1
        # V: B[j0,k0]=1, B[j1,k1]=1
        # W: C[i0,k0]=1, C[i1,k1]=1 (from M1 + M4 - M5 + M7 for C[i0,k0] and M1 - M2 + M3 + M6 for C[i1,k1])
        # Wait, M1 contributes +1 to C[i0,k0] and +1 to C[i1,k1].
        
        # Let's define the 7 products and their contributions to W.
        
        # P1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        # U: (i0,j0)=1, (i1,j1)=1
        # V: (j0,k0)=1, (j1,k1)=1
        # W: (k0,i0)=1, (k1,i1)=1
        
        # P2 = (A[i0,j0] + A[i0,j1]) * B[j1,k0]
        # U: (i0,j0)=1, (i0,j1)=1
        # V: (j1,k0)=1
        # W: (k0,i1)=1
        
        # P3 = A[i0,j0] * B[j0,k1]
        # U: (i0,j0)=1
        # V: (j0,k1)=1
        # W: (k1,i0)=1, (k1,i1)=1? No.
        # C[i0,k1] = M3 + M5. So P3 contributes to C[i0,k1].
        # C[i1,k1] = M1 - M2 + M3 + M6. So P3 contributes to C[i1,k1].
        # So W: (k1,i0)=1, (k1,i1)=1.
        
        # P4 = A[i1,j1] * B[j1,k0]
        # U: (i1,j1)=1
        # V: (j1,k0)=1
        # W: (k0,i0)=1, (k0,i1)=1.
        
        # P5 = (A[i0,j1] + A[i1,j1]) * B[j0,k0]
        # U: (i0,j1)=1, (i1,j1)=1
        # V: (j0,k0)=1
        # W: (k0,i0)=-1 (from -M5 in C[i0,k0])
        
        # P6 = (A[i0,j0] - A[i1,j1]) * (B[j1,k0] + B[j1,k1])
        # U: (i0,j0)=1, (i1,j1)=-1
        # V: (j1,k0)=1, (j1,k1)=1
        # W: (k1,i1)=1 (from +M6 in C[i1,k1])
        
        # P7 = (A[i1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k1])
        # U: (i1,j0)=1, (i0,j0)=-1
        # V: (j0,k0)=1, (j0,k1)=1
        # W: (k0,i0)=1 (from +M7 in C[i0,k0])

        # Now implement this correctly.
        
        # P1
        U[i0*m+j0][col] = 1; U[i1*m+j1][col] = 1
        V[j0*p+k0][col] = 1; V[j1*p+k1][col] = 1
        W[k0*n+i0][col] = 1; W[k1*n+i1][col] = 1
        col += 1
        
        # P2
        U[i0*m+j0][col] = 1; U[i0*m+j1][col] = 1
        V[j1*p+k0][col] = 1
        W[k0*n+i1][col] = 1
        col += 1
        
        # P3
        U[i0*m+j0][col] = 1
        V[j0*p+k1][col] = 1
        W[k1*n+i0][col] = 1; W[k1*n+i1][col] = 1
        col += 1
        
        # P4
        U[i1*m+j1][col] = 1
        V[j1*p+k0][col] = 1
        W[k0*n+i0][col] = 1; W[k0*n+i1][col] = 1
        col += 1
        
        # P5
        U[i0*m+j1][col] = 1; U[i1*m+j1][col] = 1
        V[j0*p+k0][col] = 1
        W[k0*n+i0][col] = -1
        col += 1
        
        # P6
        U[i0*m+j0][col] = 1; U[i1*m+j1][col] = -1
        V[j1*p+k0][col] = 1; V[j1*p+k1][col] = 1
        W[k1*n+i1][col] = 1
        col += 1
        
        # P7
        U[i1*m+j0][col] = 1; U[i0*m+j0][col] = -1
        V[j0*p+k0][col] = 1; V[j0*p+k1][col] = 1
        W[k0*n+i0][col] = 1
        col += 1

    # Helper to add 2x2x1 block (Rank 4)
    # i0, i1: rows of A
    # j0, j1: cols of A
    # k0: col of B
    def add_2x2x1(i0, i1, j0, j1, k0):
        nonlocal col
        # P1 = A[i0,j0] * B[j0,k0]
        U[i0*m+j0][col] = 1
        V[j0*p+k0][col] = 1
        W[k0*n+i0][col] = 1
        col += 1
        
        # P2 = A[i0,j1] * B[j1,k0]
        U[i0*m+j1][col] = 1
        V[j1*p+k0][col] = 1
        W[k0*n+i0][col] = 1
        col += 1
        
        # P3 = A[i1,j0] * B[j0,k0]
        U[i1*m+j0][col] = 1
        V[j0*p+k0][col] = 1
        W[k0*n+i1][col] = 1
        col += 1
        
        # P4 = A[i1,j1] * B[j1,k0]
        U[i1*m+j1][col] = 1
        V[j1*p+k0][col] = 1
        W[k0*n+i1][col] = 1
        col += 1

    # Block 1: 2x1x7 (A rows 0,1 col 0; B row 0 cols 0..6)
    for i in range(2):
        for k in range(7):
            U[i*m+0][col] = 1
            V[0*p+k][col] = 1
            W[k*n+i][col] = 1
            col += 1

    # Block 2: 2x2x2 (A rows 0,1 cols 1,2; B rows 1,2 cols 0,1)
    add_strassen(0, 1, 1, 2, 0, 1)

    # Block 3: 2x2x2 (A rows 0,1 cols 1,2; B rows 1,2 cols 2,3)
    add_strassen(0, 1, 1, 2, 2, 3)

    # Block 4: 2x2x1 (A rows 0,1 cols 1,2; B rows 1,2 col 4)
    add_2x2x1(0, 1, 1, 2, 4)

    # Block 5: 2x2x2 (A rows 0,1 cols 1,2; B rows 1,2 cols 5,6)
    add_strassen(0, 1, 1, 2, 5, 6)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
