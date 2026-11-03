import time
import sys

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # We will build the matrices dynamically
    # First, determine total rank
    # 8 Strassen blocks (rank 7 each) + 4 Schoolbook 2x2x1 blocks (rank 4 each)
    # Wait, 8 Strassen blocks?
    # Left side: 4 blocks.
    # Right side (k=2,3): 4 blocks.
    # Right side (k=4): 4 blocks.
    # Total Strassen: 8. Total Schoolbook: 4.
    # Rank = 8 * 7 + 4 * 4 = 56 + 16 = 72.
    
    rank = 72
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_col = 0
    
    # Helper to add a Strassen block
    def add_strassen(i_off, j_off, k_off):
        nonlocal current_col
        # Local indices 0..1 for i, j, k
        # U index: (i_off + i_local) * m + (j_off + j_local)
        # V index: (j_off + j_local) * p + (k_off + k_local)
        # W index: (k_off + k_local) * n + (i_off + i_local)
        
        # Strassen factors
        # M1: (x11+x22)(y11+y22)
        # M2: (x21+x22)y11
        # M3: x11(y12-y22)
        # M4: x22(y21-y11)
        # M5: (x11+x12)y22
        # M6: (x21-x11)(y11+y12)
        # M7: (x12-x22)(y21+y22)
        
        # Coefficients for U (A matrix)
        # x11 is at (0,0), x12 at (0,1), x21 at (1,0), x22 at (1,1)
        # M1: x11(+1), x22(+1)
        # M2: x21(+1), x22(+1)
        # M3: x11(+1)
        # M4: x22(+1)
        # M5: x11(+1), x12(+1)
        # M6: x21(+1), x11(-1)
        # M7: x12(+1), x22(-1)
        
        u_coeffs = [
            [1, 0, 1, 0, 1, -1, 0],   # x11
            [0, 0, 0, 0, 1, 0, 1],    # x12
            [0, 1, 0, 0, 0, 1, 0],    # x21
            [1, 1, 0, 1, 0, 0, -1]    # x22
        ]
        
        # Coefficients for V (B matrix)
        # y11 is at (0,0), y12 at (0,1), y21 at (1,0), y22 at (1,1)
        # M1: y11(+1), y22(+1)
        # M2: y11(+1)
        # M3: y12(+1), y22(-1)
        # M4: y21(+1), y11(-1)
        # M5: y22(+1)
        # M6: y11(+1), y12(+1)
        # M7: y21(+1), y22(+1)
        
        v_coeffs = [
            [1, 1, 0, -1, 0, 1, 0],   # y11
            [0, 0, 1, 0, 0, 1, 0],    # y12
            [0, 0, 0, 1, 0, 0, 1],    # y21
            [1, 0, -1, 0, 1, 0, 1]    # y22
        ]
        
        # Coefficients for W (C matrix)
        # z11 = M1 + M4 - M5 + M7
        # z12 = M3 + M5
        # z21 = M2 + M4
        # z22 = M1 - M2 + M3 + M6
        
        # W index: (k_local * n + i_local)
        # z11 is at (0,0) -> k=0, i=0
        # z12 is at (0,1) -> k=1, i=0
        # z21 is at (1,0) -> k=0, i=1
        # z22 is at (1,1) -> k=1, i=1
        
        # M1: z11(+1), z22(+1)
        # M2: z21(+1), z22(-1)
        # M3: z12(+1), z22(+1)
        # M4: z11(+1), z21(+1)
        # M5: z11(-1), z12(+1)
        # M6: z22(+1)
        # M7: z11(+1)
        
        w_coeffs = [
            [1, -1, 1, 1, -1, 0, 1],  # z11 (k=0, i=0)
            [0, 0, 1, 0, 1, 0, 0],    # z12 (k=1, i=0)
            [0, 1, 0, 1, 0, 0, 0],    # z21 (k=0, i=1)
            [1, -1, 1, 0, 0, 1, 0]    # z22 (k=1, i=1)
        ]
        
        # Map local to global
        for i_l in range(2):
            for j_l in range(2):
                u_idx = (i_off + i_l) * m + (j_off + j_l)
                for c in range(7):
                    U[u_idx][current_col + c] = u_coeffs[i_l][j_l][c]
        
        for j_l in range(2):
            for k_l in range(2):
                v_idx = (j_off + j_l) * p + (k_off + k_l)
                for c in range(7):
                    V[v_idx][current_col + c] = v_coeffs[j_l][k_l][c]
        
        for k_l in range(2):
            for i_l in range(2):
                w_idx = (k_off + k_l) * n + (i_off + i_l)
                for c in range(7):
                    W[w_idx][current_col + c] = w_coeffs[k_l][i_l][c]
        
        current_col += 7

    # Helper to add a Schoolbook 2x2x1 block
    def add_schoolbook_2x2x1(i_off, j_off, k_off):
        nonlocal current_col
        # x11, x12, x21, x22
        # y1, y2
        # z1 = x11y1 + x12y2
        # z2 = x21y1 + x22y2
        
        # Rank 4
        # M1: x11y1
        # M2: x12y2
        # M3: x21y1
        # M4: x22y2
        
        # U coeffs
        u_coeffs = [
            [1, 0, 0, 0], # x11
            [0, 1, 0, 0], # x12
            [0, 0, 1, 0], # x21
            [0, 0, 0, 1]  # x22
        ]
        
        # V coeffs
        v_coeffs = [
            [1, 0, 1, 0], # y1
            [0, 1, 0, 1]  # y2
        ]
        
        # W coeffs
        # z1 is at k=0, i=0
        # z2 is at k=0, i=1
        w_coeffs = [
            [1, 1, 0, 0], # z1
            [0, 0, 1, 1]  # z2
        ]
        
        for i_l in range(2):
            for j_l in range(2):
                u_idx = (i_off + i_l) * m + (j_off + j_l)
                for c in range(4):
                    U[u_idx][current_col + c] = u_coeffs[i_l][j_l][c]
        
        for j_l in range(2):
            # k_l is always 0 for 2x2x1
            v_idx = (j_off + j_l) * p + (k_off + 0)
            for c in range(4):
                V[v_idx][current_col + c] = v_coeffs[j_l][c]
        
        for i_l in range(2):
            # k_l is always 0
            w_idx = (k_off + 0) * n + (i_off + i_l)
            for c in range(4):
                W[w_idx][current_col + c] = w_coeffs[i_l][c]
        
        current_col += 4

    # Execute blocks
    # Left side (k=0,1)
    add_strassen(0, 0, 0) # A11 B11
    add_strassen(0, 2, 0) # A12 B21
    add_strassen(2, 0, 0) # A21 B11
    add_strassen(2, 2, 0) # A22 B21
    
    # Right side (k=2,3)
    add_strassen(0, 0, 2) # A11 B12a
    add_strassen(0, 2, 2) # A12 B22a
    add_strassen(2, 0, 2) # A21 B12a
    add_strassen(2, 2, 2) # A22 B22a
    
    # Right side (k=4)
    add_schoolbook_2x2x1(0, 0, 4) # A11 B12b
    add_schoolbook_2x2x1(0, 2, 4) # A12 B22b
    add_schoolbook_2x2x1(2, 0, 4) # A21 B12b
    add_schoolbook_2x2x1(2, 2, 4) # A22 B22b
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Verification
    res = construct()
    U, V, W = res["U"], res["V"], res["W"]
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # Check shapes
    assert len(U) == n * m
    assert len(V) == m * p
    assert len(W) == p * n
    assert len(U[0]) == rank
    assert len(V[0]) == rank
    assert len(W[0]) == rank
    
    # Verify decomposition
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor: T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    
    import random
    for _ in range(10):
        A = [[random.randint(1, 10) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(1, 10) for _ in range(p)] for _ in range(m)]
        
        # True product
        C_true = [[0]*p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += A[i][j] * B[j][k]
                C_true[i][k] = s
        
        # Decomposed product
        C_decomp = [[0]*p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for c in range(rank):
                    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    # Sum over j
                    for j in range(m):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                C_decomp[i][k] = s
        
        if C_true != C_decomp:
            print("Mismatch found!")
            print(C_true)
            print(C_decomp)
            sys.exit(1)
            
    print(f"Verification passed. Rank: {rank}")
