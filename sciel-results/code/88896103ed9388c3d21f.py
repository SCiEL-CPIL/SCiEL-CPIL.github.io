import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # We will use:
    # 1x 3x3x3 Strassen block (Rank 23) at (0,0,0)
    # 3x 2x2x2 Strassen blocks (Rank 7 each) at (0,3,0), (0,3,2), (0,0,3)
    # Schoolbook for the rest
    
    total_rank = 23 + 3 * 7 + (75 - 27 - 3 * 8)
    # 23 + 21 + 24 = 68
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper for 2x2x2 Strassen
    def add_strassen_2x2x2(i0, j0, k0, col_offset):
        # ... (same as before) ...
        pass

    # Helper for 3x3x3 Strassen (Rank 23)
    def add_strassen_3x3x3(i0, j0, k0, col_offset):
        # Coefficients for 3x3x3 rank 23 decomposition
        # This is a known result. I will embed the coefficients.
        # The tensor T_ijk = sum_c U_ic V_jc W_kc
        # For the block, we map local indices 0,1,2 to global i0+di, etc.
        
        # I need to retrieve or hardcode the 23 terms.
        # Since I don't have the exact list memorized, I will use a fallback 
        # if I can't verify it easily, but the prompt implies I should find improvements.
        # However, hardcoding a 23-term 3x3x3 decomposition is risky if I make a typo.
        # Let's stick to the 2x2x2 blocks if I can't guarantee the 3x3x3 correctness.
        # BUT, the rank 68 is a significant improvement.
        # I will use a known 3x3x3 decomposition from literature (e.g. Bini et al).
        
        # Actually, to be safe and within time, I will implement the 2x2x2 blocks 
        # and check if I can fit a 5th 2x2x2 block in a different configuration.
        # We established max 4 disjoint 2x2x2 blocks.
        
        # Let's try to verify if I can just output the 2x2x2 solution (Rank 71) 
        # or if I should risk the 3x3x3.
        # The directive says "investigate improvements".
        # I will implement the 3x3x3 block.
        
        # Coefficients for 3x3x3 (Local indices 0,1,2)
        # U, V, W are 9x23 matrices for the block.
        # I will use the decomposition from "A 23-term formula for 3x3 matrix multiplication"
        
        # Due to complexity and risk of error in hardcoding 23*3*3=207 non-zero coefficients,
        # and the 120s limit, I will stick to the proven 2x2x2 blocks 
        # and optimize the placement if possible.
        
        # Wait, the 4x 2x2x2 solution is Rank 71.
        # Is there a better packing of 2x2x2 blocks?
        # We proved max 4.
        
        # So, if I can't safely implement 3x3x3, I should return 71.
        # But 71 is the current score.
        
        # Let's try to implement a 3x3x3 block using a verified source in my head?
        # No, I'll use the 2x2x2 blocks.
        
        # Actually, I can try to use the 3x3x3 block and verify it at runtime.
        # If it fails, fallback to 2x2x2.
        
        pass

    # Given the constraints, I will output the 2x2x2 solution (Rank 71) 
    # as it is guaranteed correct. 
    # If I were to guess the 3x3x3 coefficients, I might introduce bugs.
    
    # However, I can try to improve the 2x2x2 solution by checking 
    # if overlapping is allowed with linear combination? No, exact decomposition.
    
    # I will return the Rank 71 solution.
    
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    total_rank = 4 * 7 + (75 - 4 * 8)
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
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
    print(f"Rank: {U.shape[1]}")
