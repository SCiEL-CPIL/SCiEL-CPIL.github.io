"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses a structured construction that exploits the rectangular nature of the problem.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def to_nice_list(arr):
        """Convert array to list with nice integer/half-integer values."""
        result = []
        for row in arr:
            nice_row = []
            for val in row:
                val2 = val * 2
                if abs(val2 - round(val2)) < 1e-6:
                    nice_val = round(val2) / 2.0
                    if abs(nice_val - round(nice_val)) < 1e-9:
                        nice_val = int(round(nice_val))
                    nice_row.append(nice_val)
                else:
                    nice_row.append(float(val))
            result.append(nice_row)
        return result

    def verify_decomp(Ua, Va, Wa, nn, mm, pp):
        """Verify that the decomposition is exact."""
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    s = np.sum(Ua[i*mm+j] * Va[j*pp+k] * Wa[k*nn+i])
                    if abs(s - 1.0) > 1e-7:
                        return False
        return True

    # Build a decomposition using Strassen for 2x2 blocks
    # A is 3x3, B is 3x8
    # Split A into 2x2 block A11 and row/col A12, A21, A22
    # Split B into 2x4 block B11 and row B21, col B12, B22
    
    # Actually, let's use a more systematic approach:
    # Use Strassen for the 2x2 * 2x4 part (top-left 2x2 of A times top 2 rows of B)
    # Then handle the remaining parts directly
    
    U_t = []
    V_t = []
    W_t = []
    
    def add_term(U_entries, V_entries, W_entries):
        U_t.append(U_entries)
        V_t.append(V_entries)
        W_t.append(W_entries)
    
    # Strassen decomposition for 2x2 * 2x2
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # For A[0:2, 0:2] * B[0:2, k:k+2]
    for k_off in range(0, 8, 2):
        # a = A[0,0], b = A[0,1], c = A[1,0], d = A[1,1]
        # e = B[0,k_off], f = B[0,k_off+1], g = B[1,k_off], h = B[1,k_off+1]
        
        # M1 = (a+d)(e+h)
        add_term([((0,0), 1), ((1,1), 1)],
                 [((0,k_off), 1), ((1,k_off+1), 1)],
                 [((0,k_off), 1), ((1,k_off+1), 1)])
        
        # M2 = (c+d)e
        add_term([((1,0), 1), ((1,1), 1)],
                 [((0,k_off), 1)],
                 [((1,k_off), 1), ((1,k_off+1), -1)])
        
        # M3 = a(f-h)
        add_term([((0,0), 1)],
                 [((0,k_off+1), 1), ((1,k_off+1), -1)],
                 [((0,k_off+1), 1), ((1,k_off+1), 1)])
        
        # M4 = d(g-e)
        add_term([((1,1), 1)],
                 [((1,k_off), 1), ((0,k_off), -1)],
                 [((0,k_off), 1), ((1,k_off), 1)])
        
        # M5 = (a+b)h
        add_term([((0,0), 1), ((0,1), 1)],
                 [((1,k_off+1), 1)],
                 [((0,k_off), -1), ((0,k_off+1), 1)])
        
        # M6 = (c-a)(e+f)
        add_term([((1,0), 1), ((0,0), -1)],
                 [((0,k_off), 1), ((0,k_off+1), 1)],
                 [((1,k_off+1), 1)])
        
        # M7 = (b-d)(g+h)
        add_term([((0,1), 1), ((1,1), -1)],
                 [((1,k_off), 1), ((1,k_off+1), 1)],
                 [((0,k_off), 1)])
    
    # Now handle the remaining parts:
    # A[0:2, 2] * B[2, :] (2x1 * 1x8)
    for i in range(2):
        for k in range(8):
            add_term([((i, 2), 1)],
                     [((2, k), 1)],
                     [((i, k), 1)])
    
    # A[2, 0:2] * B[0:2, :] (1x2 * 2x8)
    for j in range(2):
        for k in range(8):
            add_term([((2, j), 1)],
                     [((j, k), 1)],
                     [((2, k), 1)])
    
    # A[2, 2] * B[2, :] (1x1 * 1x8)
    for k in range(8):
        add_term([((2, 2), 1)],
                 [((2, k), 1)],
                 [((2, k), 1)])
    
    rank = len(W_t)
    
    # Build factor matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for idx in range(rank):
        for (i, j), coeff in U_t[idx]:
            U[i * m + j, idx] = coeff
        for (j, k), coeff in V_t[idx]:
            V[j * p + k, idx] = coeff
        for (i, k), coeff in W_t[idx]:
            W[k * n + i, idx] = coeff

    if not verify_decomp(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=float)
        V_fb = np.zeros((m * p, rank_fb), dtype=float)
        W_fb = np.zeros((p * n, rank_fb), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": to_nice_list(U_fb), "V": to_nice_list(V_fb), "W": to_nice_list(W_fb)}
    
    # Try to reduce rank by finding linear dependencies
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Check for zero columns
    for r in range(rank - 1, -1, -1):
        if time.time() - start_time > DEADLINE:
            break
        if np.all(np.abs(U[:, r]) < 1e-12) and np.all(np.abs(V[:, r]) < 1e-12) and np.all(np.abs(W[:, r]) < 1e-12):
            U = np.delete(U, r, axis=1)
            V = np.delete(V, r, axis=1)
            W = np.delete(W, r, axis=1)
            rank -= 1
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = rank
    
    # Try to find if any term is in the span of others
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 10:
        for r in range(rank - 1, -1, -1):
            if time.time() - start_time > DEADLINE:
                break
            
            vec_r = np.kron(np.kron(U[:, r], V[:, r]), W[:, r])
            others = []
            for s in range(rank):
                if s != r:
                    others.append(np.kron(np.kron(U[:, s], V[:, s]), W[:, s]))
            
            if len(others) == 0:
                continue
            
            others_mat = np.column_stack(others)
            
            try:
                coeffs, residuals, rank_others, _ = np.linalg.lstsq(others_mat, vec_r, rcond=1e-10)
                reconstruction = others_mat @ coeffs
                if np.allclose(reconstruction, vec_r, atol=1e-8):
                    U = np.delete(U, r, axis=1)
                    V = np.delete(V, r, axis=1)
                    W = np.delete(W, r, axis=1)
                    rank -= 1
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = rank
            except:
                pass
    
    # Final verification
    if not verify_decomp(best_U, best_V, best_W, n, m, p):
        # Rebuild from scratch
        U_t2 = []
        V_t2 = []
        W_t2 = []
        
        for k_off in range(0, 8, 2):
            add_term([((0,0), 1), ((1,1), 1)],
                     [((0,k_off), 1), ((1,k_off+1), 1)],
                     [((0,k_off), 1), ((1,k_off+1), 1)])
            add_term([((1,0), 1), ((1,1), 1)],
                     [((0,k_off), 1)],
                     [((1,k_off), 1), ((1,k_off+1), -1)])
            add_term([((0,0), 1)],
                     [((0,k_off+1), 1), ((1,k_off+1), -1)],
                     [((0,k_off+1), 1), ((1,k_off+1), 1)])
            add_term([((1,1), 1)],
                     [((1,k_off), 1), ((0,k_off), -1)],
                     [((0,k_off), 1), ((1,k_off), 1)])
            add_term([((0,0), 1), ((0,1), 1)],
                     [((1,k_off+1), 1)],
                     [((0,k_off), -1), ((0,k_off+1), 1)])
            add_term([((1,0), 1), ((0,0), -1)],
                     [((0,k_off), 1), ((0,k_off+1), 1)],
                     [((1,k_off+1), 1)])
            add_term([((0,1), 1), ((1,1), -1)],
                     [((1,k_off), 1), ((1,k_off+1), 1)],
                     [((0,k_off), 1)])
        
        for i in range(2):
            for k in range(8):
                add_term([((i, 2), 1)],
                         [((2, k), 1)],
                         [((i, k), 1)])
        
        for j in range(2):
            for k in range(8):
                add_term([((2, j), 1)],
                         [((j, k), 1)],
                         [((2, k), 1)])
        
        for k in range(8):
            add_term([((2, 2), 1)],
                     [((2, k), 1)],
                     [((2, k), 1)])
        
        rank2 = len(W_t2)
        U2 = np.zeros((n * m, rank2), dtype=float)
        V2 = np.zeros((m * p, rank2), dtype=float)
        W2 = np.zeros((p * n, rank2), dtype=float)
        
        for idx in range(rank2):
            for (i, j), coeff in U_t2[idx]:
                U2[i * m + j, idx] = coeff
            for (j, k), coeff in V_t2[idx]:
                V2[j * p + k, idx] = coeff
            for (i, k), coeff in W_t2[idx]:
                W2[k * n + i, idx] = coeff
        
        return {"U": to_nice_list(U2), "V": to_nice_list(V2), "W": to_nice_list(W2)}
    
    return {"U": to_nice_list(best_U), "V": to_nice_list(best_V), "W": to_nice_list(best_W)}
# EVOLVE-BLOCK-END
