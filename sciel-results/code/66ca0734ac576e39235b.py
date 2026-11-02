"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 3
    
    # Start with Strassen+schoolbook rank-17 decomposition
    rank = 17
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Part 2: Schoolbook for middle column
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1
    
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1
    
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1
    
    # Part 3: Schoolbook for last column
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1
    
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1
    
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1
    
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1
    
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1
    
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    # Convert to Fraction for exact arithmetic
    U_frac = [[Fraction(int(U[i, r])) for r in range(rank)] for i in range(n * m)]
    V_frac = [[Fraction(int(V[i, r])) for r in range(rank)] for i in range(m * p)]
    W_frac = [[Fraction(int(W[i, r])) for r in range(rank)] for i in range(p * n)]
    
    # Verify initial decomposition
    def verify_frac(U_f, V_f, W_f, n_, m_, p_, rank_):
        for i in range(n_):
            for k in range(p_):
                for j1 in range(m_):
                    for j2 in range(m_):
                        s = Fraction(0)
                        for r in range(rank_):
                            s += W_f[k*n_+i, r] * U_f[i*m_+j1, r] * V_f[j2*p_+k, r]
                        target = Fraction(1) if j1 == j2 else Fraction(0)
                        if s != target:
                            return False
        return True
    
    if not verify_frac(U_frac, V_frac, W_frac, n, m, p, rank):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_fb = [[Fraction(0)] * rank_sb for _ in range(n * m)]
        V_fb = [[Fraction(0)] * rank_sb for _ in range(m * p)]
        W_fb = [[Fraction(0)] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col] = Fraction(1)
                    V_fb[j*p+k][col] = Fraction(1)
                    W_fb[k*n+i][col] = Fraction(1)
                    col += 1
        return {"U": [[float(x) for x in row] for row in U_fb], 
                "V": [[float(x) for x in row] for row in V_fb], 
                "W": [[float(x) for x in row] for row in W_fb]}
    
    # Systematic rank reduction using linear algebra over rationals
    # The tensor T is determined by the rank-1 terms u_r ⊗ v_r ⊗ w_r
    # We can remove term r if u_r ⊗ v_r ⊗ w_r is in the span of the others
    
    current_U = [row[:] for row in U_frac]
    current_V = [row[:] for row in V_frac]
    current_W = [row[:] for row in W_frac]
    current_rank = rank
    
    # Try to remove each term one at a time
    for r_remove in range(current_rank):
        if time.time() > deadline - 10:
            break
        
        # Remove term r_remove
        test_U = [row[:] for row in current_U]
        test_V = [row[:] for row in current_V]
        test_W = [row[:] for row in current_W]
        for i in range(n * m):
            test_U[i].pop(r_remove)
        for i in range(m * p):
            test_V[i].pop(r_remove)
        for i in range(p * n):
            test_W[i].pop(r_remove)
        
        if verify_frac(test_U, test_V, test_W, n, m, p, current_rank - 1):
            current_U = test_U
            current_V = test_V
            current_W = test_W
            current_rank -= 1
            # Restart the loop since indices shifted
            r_remove = -1
            continue
    
    # Try more aggressive reduction: check if any term is a linear combination
    # of others in a way that preserves the tensor
    # This is harder; we'll do a limited search for 2-term replacements
    
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        for r_remove in range(current_rank):
            if time.time() > deadline - 10:
                break
            
            test_U = [row[:] for row in current_U]
            test_V = [row[:] for row in current_V]
            test_W = [row[:] for row in current_W]
            for i in range(n * m):
                test_U[i].pop(r_remove)
            for i in range(m * p):
                test_V[i].pop(r_remove)
            for i in range(p * n):
                test_W[i].pop(r_remove)
            
            if verify_frac(test_U, test_V, test_W, n, m, p, current_rank - 1):
                current_U = test_U
                current_V = test_V
                current_W = test_W
                current_rank -= 1
                improved = True
                break
    
    # Convert back to float for output
    U_out = [[float(x) for x in row] for row in current_U]
    V_out = [[float(x) for x in row] for row in current_V]
    W_out = [[float(x) for x in row] for row in current_W]
    
    # Final verification with floats
    def verify_float(U_mat, V_mat, W_mat, n_, m_, p_, rank_):
        for i in range(n_):
            for k in range(p_):
                for j1 in range(m_):
                    for j2 in range(m_):
                        s = 0.0
                        for r in range(rank_):
                            s += W_mat[k*n_+i][r] * U_mat[i*m_+j1][r] * V_mat[j2*p_+k][r]
                        target = 1.0 if j1 == j2 else 0.0
                        if abs(s - target) > 1e-9:
                            return False
        return True
    
    if not verify_float(U_out, V_out, W_out, n, m, p, current_rank):
        # Fallback to original rank-17
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
