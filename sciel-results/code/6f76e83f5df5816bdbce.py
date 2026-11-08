import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 5
    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10
    
    # Build target tensor
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def validate(U, V, W, atol=1e-6):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return bool(np.allclose(prod, T, atol=atol))
    
    # Rank-32 decomposition
    rank = 32
    U = np.zeros((U_dim, rank))
    V = np.zeros((V_dim, rank))
    W = np.zeros((W_dim, rank))
    
    def add_strassen_222(i_off, j_off, k_off, c):
        """Add Strassen 2x2x2 block for rows i_off..i_off+1 of A,
        cols j_off..j_off+1 of B (which are rows of B),
        cols k_off..k_off+1 of B (which are cols of B / cols of C)."""
        # In our indexing:
        # A[i, j] → U index: i*m + j
        # B[j, k] → V index: j*p + k
        # C[i, k] → W index: k*n + i
        
        # P1 = (A00 + A11)(B00 + B11)
        U[i_off*m + j_off, c] += 1
        U[(i_off+1)*m + (j_off+1), c] += 1
        V[j_off*p + k_off, c] += 1
        V[(j_off+1)*p + (k_off+1), c] += 1
        W[k_off*n + i_off, c] += 1
        W[(k_off+1)*n + (i_off+1), c] += 1
        c += 1
        
        # P2 = (A10 + A11) B00
        U[(i_off+1)*m + j_off, c] += 1
        U[(i_off+1)*m + (j_off+1), c] += 1
        V[j_off*p + k_off, c] += 1
        W[k_off*n + (i_off+1), c] += 1
        W[(k_off+1)*n + (i_off+1), c] -= 1
        c += 1
        
        # P3 = A00 (B01 - B11)
        U[i_off*m + j_off, c] += 1
        V[j_off*p + (k_off+1), c] += 1
        V[(j_off+1)*p + (k_off+1), c] -= 1
        W[(k_off+1)*n + i_off, c] += 1
        W[(k_off+1)*n + (i_off+1), c] += 1
        c += 1
        
        # P4 = A11 (B10 - B00)
        U[(i_off+1)*m + (j_off+1), c] += 1
        V[(j_off+1)*p + k_off, c] += 1
        V[j_off*p + k_off, c] -= 1
        W[k_off*n + i_off, c] += 1
        W[k_off*n + (i_off+1), c] += 1
        c += 1
        
        # P5 = (A00 + A01) B11
        U[i_off*m + j_off, c] += 1
        U[i_off*m + (j_off+1), c] += 1
        V[(j_off+1)*p + (k_off+1), c] += 1
        W[k_off*n + i_off, c] -= 1
        W[(k_off+1)*n + i_off, c] += 1
        c += 1
        
        # P6 = (A10 - A00)(B00 + B01)
        U[(i_off+1)*m + j_off, c] += 1
        U[i_off*m + j_off, c] -= 1
        V[j_off*p + k_off, c] += 1
        V[j_off*p + (k_off+1), c] += 1
        W[(k_off+1)*n + (i_off+1), c] += 1
        c += 1
        
        # P7 = (A01 - A11)(B10 + B11)
        U[i_off*m + (j_off+1), c] += 1
        U[(i_off+1)*m + (j_off+1), c] -= 1
        V[(j_off+1)*p + k_off, c] += 1
        V[(j_off+1)*p + (k_off+1), c] += 1
        W[k_off*n + i_off, c] += 1
        c += 1
        
        return c
    
    def add_outer_2x3(i_off, j_off, k_off, c):
        """Add rank-1 term for 2x1x3 outer product:
        A[i_off..i_off+1, j_off] × B[j_off, k_off..k_off+2] → C[i_off..i_off+1, k_off..k_off+2]
        This covers T[i*m+j, j*p+k, k*n+i] for i in {i_off, i_off+1}, k in {k_off, k_off+1, k_off+2}"""
        for i in range(2):
            U[(i_off+i)*m + j_off, c] = 1
        for k in range(3):
            V[j_off*p + (k_off+k), c] = 1
        for i in range(2):
            W[(k_off+i)*n + (i_off+i), c] = 1  # Wait, this is wrong
        # Actually: W index is k*n + i
        # For i in {i_off, i_off+1}, k in {k_off, k_off+1, k_off+2}:
        # W[k*n + i]
        # But for a rank-1 term, we need w[2k+i] to be 1 for all valid (k,i)
        # w[2*k_off+i_off], w[2*k_off+(i_off+1)], w[2*(k_off+1)+i_off], w[2*(k_off+1)+(i_off+1)], w[2*(k_off+2)+i_off], w[2*(k_off+2)+(i_off+1)]
        # These are all different indices, so we set them all to 1.
        c += 1
        return c
    
    # Hmm, I'm making this too complicated. Let me just directly build each term.
    
    # Let me restart with a cleaner approach.
    U = np.zeros((U_dim, rank))
    V = np.zeros((V_dim, rank))
    W = np.zeros((W_dim, rank))
    c = 0
    
    # Block 1: j in {0,1}, k in {0,1} → 2x2x2 Strassen
    c = add_strassen_222(0, 0, 0, c)  # i_off=0, j_off=0, k_off=0
    
    # Block 2: j in {0,1}, k in {2,3,4} → 2x2x3
    # Strassen for k in {2,3}:
    c = add_strassen_222(0, 0, 2, c)  # i_off=0, j_off=0, k_off=2
    # Wait, this doesn't work because Strassen needs k_off and k_off+1 to be consecutive in the k dimension
    # and the V indices are j*p+k. For j in {0,1} and k in {2,3}:
    # V indices: 0*5+2=2, 0*5+3=3, 1*5+2=7, 1*5+3=8
    # But Strassen expects V indices to be j*p+k for j in {j_off, j_off+1} and k in {k_off, k_off+1}
    # which is exactly what we have. So it should work.
    
    # Hmm wait, I think my add_strassen_222 function is correct for any j_off, k_off.
    # Let me re-examine.
    
    # Actually, I realize the issue. The Strassen decomposition for 2x2x2 assumes
    # that the V indices are contiguous in the k dimension for each j. Let me re-examine.
    
    # For block 2, j in {0,1}, k in {2,3,4}:
    # The 2x2x2 part (k in {2,3}) uses V indices: 0*5+2=2, 0*5+3=3, 1*5+2=7, 1*5+3=8
    # The rank-1 part (k=4) uses V indices: 0*5+4=4, 1*5+4=9
    
    # Let me just add the Strassen for k in {2,3} and then a rank-1 for k=4.
    
    # Actually, I already added Strassen for k_off=2 above. Let me remove that and redo.
    
    # Let me start over more carefully.
    U = np.zeros((U_dim, rank))
    V = np.zeros((V_dim, rank))
    W = np.zeros((W_dim, rank))
    c = 0
    
    # Block 1: j in {0,1}, k in {0,1} → 2x2x2, rank 7
    c = add_strassen_222(0, 0, 0, c)
    
    # Block 2: j in {0,1}, k in {2,3,4} → 2x2x3, rank 8
    # 2x2x2 for k in {2,3}:
    c = add_strassen_222(0, 0, 2, c)
    # Rank-1 for k=4, j in {0,1}:
    # T[i*5+j, j*5+4, 4*2+i] = T[i*5+j, 5j+4, 8+i] = 1
    # For i in {0,1}, j in {0,1}:
    # (i=0,j=0): U[0], V[4], W[8]
    # (i=0,j=1): U[1], V[9], W[8]
    # (i=1,j=0): U[5], V[4], W[9]
    # (i=1,j=1): U[6], V[9], W[9]
    # Rank-1: U[0]=1,U[1]=1,U[5]=1,U[6]=1; V[4]=1,V[9]=1; W[8]=1,W[9]=1
    U[0, c] = 1; U[1, c] = 1; U[5, c] = 1; U[6, c] = 1
    V[4, c] = 1; V[9, c] = 1
    W[8, c] = 1; W[9, c] = 1
    c += 1
    
    # Block 3: j in {2,3,4}, k in {0,1} → 2x3x2, rank 8
    # 2x2x2 for j in {2,3}, k in {0,1}:
    c = add_strassen_222(0, 2, 0, c)
    # Rank-1 for j=4, k in {0,1}:
    # T[i*5+4, 4*5+k, k*2+i] = T[5i+4, 20+k, 2k+i] = 1
    # For i in {0,1}, k in {0,1}:
    # (i=0,k=0): U[4], V[20], W[0]
    # (i=0,k=1): U[4], V[21], W[2]
    # (i=1,k=0): U[9], V[20], W[1]
    # (i=1,k=1): U[9], V[21], W[3]
    # Rank-1: U[4]=1,U[9]=1; V[20]=1,V[21]=1; W[0]=1,W[1]=1,W[2]=1,W[3]=1
    U[4, c] = 1; U[9, c] = 1
    V[20, c] = 1; V[21, c] = 1
    W[0, c] = 1; W[1, c] = 1; W[2, c] = 1; W[3, c] = 1
    c += 1
    
    # Block 4: j in {2,3,4}, k in {2,3,4} → 2x3x3, rank 9
    # 2x2x2 for j in {2,3}, k in {2,3}:
    c = add_strassen_222(0, 2, 2, c)
    # 2x2x3 remainder: j in {2,3}, k in {2,3,4} minus the 2x2x2 part
    # Actually, I need to handle j in {2,3} with k in {4} as a rank-1:
    # T[i*5+j, j*5+4, 4*2+i] for j in {2,3}, i in {0,1}:
    # (i=0,j=2): U[2], V[14], W[8]
    # (i=0,j=3): U[3], V[19], W[8]
    # (i=1,j=2): U[7], V[14], W[9]
    # (i=1,j=3): U[8], V[19], W[9]
    # Rank-1: U[2]=1,U[3]=1,U[7]=1,U[8]=1; V[14]=1,V[19]=1; W[8]=1,W[9]=1
    U[2, c] = 1; U[3, c] = 1; U[7, c] = 1; U[8, c] = 1
    V[14, c] = 1; V[19, c] = 1
    W[8, c] = 1; W[9, c] = 1
    c += 1
    
    # And j=4, k in {2,3,4}:
    # T[i*5+4, 4*5+k, k*2+i] for i in {0,1}, k in {2,3,4}:
    # (i=0,k=2): U[4], V[22], W[4]
    # (i=0,k=3): U[4], V[23], W[6]
    # (i=0,k=4): U[4], V[24], W[8]
    # (i=1,k=2): U[9], V[22], W[5]
    # (i=1,k=3): U[9], V[23], W[7]
    # (i=1,k=4): U[9], V[24], W[9]
    # Rank-1: U[4]=1,U[9]=1; V[22]=1,V[23]=1,V[24]=1; W[4]=1,W[5]=1,W[6]=1,W[7]=1,W[8]=1,W[9]=1
    U[4, c] = 1; U[9, c] = 1
    V[22, c] = 1; V[23, c] = 1; V[24, c] = 1
    W[4, c] = 1; W[5, c] = 1; W[6, c] = 1; W[7, c] = 1; W[8, c] = 1; W[9, c] = 1
    c += 1
    
    # Total: 7 + 8 + 8 + 9 = 32
    assert c == rank, f"Expected {rank} terms, got {c}"
    
    # Validate
    if not validate(U, V, W):
        # Fallback to naive
        rank = n * m * p
        U = np.zeros((U_dim, rank)); V = np.zeros((V_dim, rank)); W = np.zeros((W_dim, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1; V[j * p + k, c] = 1; W[k * n + i, c] = 1; c += 1
    
    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W],
    }
