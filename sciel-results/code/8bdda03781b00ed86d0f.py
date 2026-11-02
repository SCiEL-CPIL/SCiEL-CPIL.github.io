import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 71 solution:
    # Four Strassen blocks (rank 7 each) + Schoolbook for the rest (43 terms)
    # Blocks are defined by (i0, j0, k0) with size 2x2x2
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2: i in {0,1}, j in {2,3}, k in {0,1}
    # Block 3: i in {0,1}, j in {0,1}, k in {2,3}
    # Block 4: i in {0,1}, j in {2,3}, k in {2,3}
    
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    num_strassen_blocks = len(blocks)
    total_strassen_rank = num_strassen_blocks * 7
    
    # Calculate number of schoolbook terms
    total_terms = n * m * p
    strassen_terms_count = num_strassen_blocks * 8
    schoolbook_terms_count = total_terms - strassen_terms_count
    
    total_rank = total_strassen_rank + schoolbook_terms_count
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    def add_strassen_block(i0, j0, k0, col_offset):
        # Local indices 0,1 map to global i0, i0+1 etc.
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # M1 = (A00+A11)(B00+B11)
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M2 = (A10+A11)B00
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        # M3 = A00(B01-B11)
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M4 = A11(B10-B00)
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        # M5 = (A00+A01)B11
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        # M6 = (A10-A00)(B00+B01)
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M7 = (A01-A11)(B10+B11)
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    # Add Strassen blocks
    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    # Schoolbook for the rest
    # We need to identify which (i,j,k) triples were covered by Strassen
    # Covered if i in [i0, i0+1] and j in [j0, j0+1] and k in [k0, k0+1] for any block
    
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
                    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    # Check rank
    print(f"Rank: {U.shape[1]}")
    # Verify correctness
    # T_ijk = sum_c U_ic V_jc W_kc
    # Expected: delta_ik if j is such that A_ij B_jk = C_ik
    # Actually, the tensor T is such that C_ik = sum_j A_ij T_ijk B_jk
    # So T_ijk should be 1 if it corresponds to the multiplication term A_ij B_jk
    # My construction sets U, V, W such that the contraction gives the identity map on the tensor product space.
    # Let's verify a few entries.
    # T_ijk = sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}
    # For schoolbook terms, this is 1.
    # For Strassen terms, it should also reconstruct the identity.
    
    # Full verification
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Build the tensor T from the decomposition
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    max_error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Expected value is 1 for all i, j, k in matrix multiplication
                # Wait, the tensor for matrix multiplication C = AB is T_ijk = 1 for all i,j,k
                # Because C_ik = sum_j A_ij B_jk
                # So the coefficient is always 1.
                
                val = 0.0
                for c in range(U.shape[1]):
                    val += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                
                if abs(val - 1.0) > max_error:
                    max_error = abs(val - 1.0)
    
    print(f"Max error: {max_error}")
    if max_error < 1e-10:
        print("Correct!")
    else:
        print("INCORRECT!")

BET: {mechanism: "No change in rank; identical algorithm structure to parent. The only difference is the addition of a verification block in the __main__ guard, which does not affect the construct() output or the score.", q: 0.0, delta: [0.0, 0.0]}
