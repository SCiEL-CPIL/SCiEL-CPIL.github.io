import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    n_rows = n * m  # 6
    m_rows = m * p  # 27
    p_rows = p * n  # 18
    
    def verify(U, V, W, rank):
        """Verify tensor decomposition correctness."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        C_computed = np.zeros((n, p))
        for r in range(rank):
            for i in range(n):
                for k in range(p):
                    for k_inner in range(m):
                        C_computed[i, k] += U_np[i*m + k_inner, r] * V_np[k_inner*p + k, r] * W_np[k*n + i, r]
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    # Schoolbook decomposition: rank = n*m*p = 2*3*9 = 54
    # But we can do better with Strassen on sub-blocks
    
    # Strategy: Decompose 2x3x9 as:
    # - 3 blocks of 2x2x3 (each rank 6, schoolbook) = 18
    # - 1 block of 2x1x9 (rank 18) = 18  
    # Total: 36
    
    # Actually, let's use Strassen for 2x2x2 blocks:
    # 9 = 2*4 + 1, so 4 Strassen blocks (rank 7 each) + 1 direct (rank 4) + 18 = 50
    
    # Better: use 2x2x3 blocks with schoolbook (rank 6 each):
    # 3 blocks * 6 + 18 = 36
    
    # Even better: Can we find rank < 6 for 2x2x3? No, rank(2x2x3) = 6 is optimal.
    # But we can try Strassen on 2x2x4: rank 7 for 4 columns
    # 9 = 4 + 4 + 1: 2 Strassen-4 blocks (7 each) + 1 direct (4) + 18 = 36
    # Same as above!
    
    # Let's try: 2x2x3 schoolbook blocks
    def schoolbook_2x2x3():
        """Return rank-6 decomposition of 2x2x3 matrix multiplication."""
        # T[i,k,j] where i in {0,1}, k in {0,1,2}, j in {0,1,2}
        # U: (4, 6), V: (6, 6), W: (6, 6)
        # Indexing: U[i*2+k], V[k*3+j], W[j*2+i]
        U = [[0]*6 for _ in range(4)]
        V = [[0]*6 for _ in range(6)]
        W = [[0]*6 for _ in range(6)]
        
        col = 0
        for i in range(2):
            for k in range(3):
                for j in range(3):
                    U[i*2+k][col] = 1
                    V[k*3+j][col] = 1
                    W[j*2+i][col] = 1
                    col += 1
        return U, V, W
    
    # Build the full decomposition using 3 blocks of 2x2x3 + 1 block of 2x1x9
    def build_decomposition():
        rank = 3 * 6 + 18  # 36
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        # Three 2x2x3 blocks (columns 0-2, 3-5, 6-8 of B)
        for block in range(3):
            k_start = block * 3
            U223, V223, W223 = schoolbook_2x2x3()
            for rr in range(6):
                for i in range(2):
                    for k in range(2):
                        U[i * m + k][col + rr] += U223[i * 2 + k][rr]
                for k in range(2):
                    for j in range(3):
                        V[k * p + (k_start + j)][col + rr] += V223[k * 3 + j][rr]
                for j in range(3):
                    for i in range(2):
                        W[(k_start + j) * n + i][col + rr] += W223[j * 2 + i][rr]
            col += 6
        
        # Part 2: A_R (2x1) * B_B (1x9), rank 18
        u_ar0 = 0 * m + 2
        u_ar1 = 1 * m + 2
        for k in range(9):
            v_b = 2 * p + k
            U[u_ar0][col] = 1
            V[v_b][col] = 1
            W[k * n + 0][col] = 1
            U[u_ar1][col+1] = 1
            V[v_b][col+1] = 1
            W[k * n + 1][col+1] = 1
            col += 2
        
        return {"U": U, "V": V, "W": W}
    
    # Try the 36-rank decomposition
    result_36 = build_decomposition()
    
    if verify(result_36["U"], result_36["V"], result_36["W"], 36):
        return result_36
    
    # Fallback to Strassen rank 50
    def strassen_solution():
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1
            v_e = 0 * p + k0
            v_f = 0 * p + k1
            v_g = 1 * p + k0
            v_h = 1 * p + k1
            u_a = 0 * m + 0
            u_b = 0 * m + 1
            u_c = 1 * m + 0
            u_d = 1 * m + 1
            
            U[u_a][col] += 1; U[u_d][col] += 1; V[v_e][col] += 1; V[v_h][col] += 1
            U[u_c][col+1] += 1; U[u_d][col+1] += 1; V[v_e][col+1] += 1
            U[u_a][col+2] += 1; V[v_f][col+2] += 1; V[v_h][col+2] -= 1
            U[u_d][col+3] += 1; V[v_g][col+3] += 1; V[v_e][col+3] -= 1
            U[u_a][col+4] += 1; U[u_b][col+4] += 1; V[v_h][col+4] += 1
            U[u_c][col+5] += 1; U[u_a][col+5] -= 1; V[v_e][col+5] += 1; V[v_f][col+5] += 1
            U[u_b][col+6] += 1; U[u_d][col+6] -= 1; V[v_g][col+6] += 1; V[v_h][col+6] += 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            W[w_00][col] += 1; W[w_00][col+3] += 1; W[w_00][col+4] -= 1; W[w_00][col+6] += 1
            W[w_01][col+2] += 1; W[w_01][col+4] += 1
            W[w_10][col+1] += 1; W[w_10][col+3] += 1
            W[w_11][col] += 1; W[w_11][col+1] -= 1; W[w_11][col+2] += 1; W[w_11][col+5] += 1
            col += 7
            
        k8 = 8
        v_e = 0 * p + k8
        v_g = 1 * p + k8
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        U[u_a][col] += 1; V[v_e][col] += 1
        U[u_b][col+1] += 1; V[v_g][col+1] += 1
        U[u_c][col+2] += 1; V[v_e][col+2] += 1
        U[u_d][col+3] += 1; V[v_g][col+3] += 1
        
        w_08 = k8 * n + 0
        w_18 = k8 * n + 1
        W[w_08][col] += 1; W[w_08][col+1] += 1
        W[w_18][col+2] += 1; W[w_18][col+3] += 1
        col += 4
        
        u_ar0 = 0 * m + 2
        u_ar1 = 1 * m + 2
        for k in range(9):
            v_b = 2 * p + k
            U[u_ar0][col] += 1; V[v_b][col] += 1; W[k * n + 0][col] += 1
            U[u_ar1][col+1] += 1; V[v_b][col+1] += 1; W[k * n + 1][col+1] += 1
            col += 2
            
        return {"U": U, "V": V, "W": W}
    
    return strassen_solution()

