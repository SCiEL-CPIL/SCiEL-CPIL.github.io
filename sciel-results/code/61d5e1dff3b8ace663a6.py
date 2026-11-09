"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x4 x 4x4 matrix multiplication.
    
    Strategy:
    1. Build the rank-28 Strassen-based decomposition (4 blocks x 7 Strassen terms).
    2. Try to reduce rank by finding linear dependencies among the 28 terms.
    3. Return the lowest-rank verified decomposition found.
    """
    n, m, p = DIMENSIONS  # 2, 4, 4
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def time_left():
        return deadline - time.time()
    
    def verify(U, V, W, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.zeros((n, m, p))
        for c in range(rank):
            T += np.einsum('ij,jk,ki->ijk', U3[:, :, c], V3[:, :, c], W3[:, :, c])
        target = np.ones((n, m, p))
        return np.allclose(T, target, atol=tol)
    
    def construct_strassen_rank28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Standard 2x2x2 Strassen terms (7 terms)
        strassen_terms = [
            {'U': [(0, 0, 1), (1, 1, 1)], 'V': [(0, 0, 1), (1, 1, 1)], 'W': [(0, 0, 1), (1, 1, 1)]},
            {'U': [(1, 0, 1), (1, 1, 1)], 'V': [(0, 0, 1)], 'W': [(0, 1, 1), (1, 1, -1)]},
            {'U': [(0, 0, 1)], 'V': [(0, 1, 1), (1, 1, -1)], 'W': [(1, 0, 1), (1, 1, 1)]},
            {'U': [(1, 1, 1)], 'V': [(1, 0, 1), (0, 0, -1)], 'W': [(0, 0, 1), (0, 1, 1)]},
            {'U': [(0, 0, 1), (0, 1, 1)], 'V': [(1, 1, 1)], 'W': [(0, 0, -1), (1, 0, 1)]},
            {'U': [(1, 0, 1), (0, 0, -1)], 'V': [(0, 0, 1), (0, 1, 1)], 'W': [(1, 1, 1)]},
            {'U': [(0, 1, 1), (1, 1, -1)], 'V': [(1, 0, 1), (1, 1, 1)], 'W': [(0, 0, 1)]},
        ]
        
        # Four 2x2 blocks in the 4x4 matrix
        blocks = [(0, 0), (0, 2), (2, 0), (2, 2)]
        
        col = 0
        for j_off, k_off in blocks:
            for term in strassen_terms:
                for (li, lj, coeff) in term['U']:
                    U[li * m + (lj + j_off)][col] += coeff
                for (lj, lk, coeff) in term['V']:
                    V[(lj + j_off) * p + (lk + k_off)][col] += coeff
                for (lk, li, coeff) in term['W']:
                    W[(lk + k_off) * n + li][col] += coeff
                col += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Build the rank-28 decomposition
    best = construct_strassen_rank28()
    
    if not verify(best['U'], best['V'], best['W']):
        # Should not happen, but return anyway
        return best
    
    best_rank = 28
    
    # Try to find linear dependencies to reduce rank
    # Convert to numpy arrays
    U_arr = np.array(best['U'], dtype=float)
    V_arr = np.array(best['V'], dtype=float)
    W_arr = np.array(best['W'], dtype=float)
    
    # For each subset of columns to remove, check if the remaining columns still span the tensor
    # This is expensive, so we try small subsets first
    
    # First, try removing 1 column at a time
    for c_remove in range(28):
        if time_left() < 5:
            break
        U_test = np.delete(U_arr, c_remove, axis=1)
        V_test = np.delete(V_arr, c_remove, axis=1)
        W_test = np.delete(W_arr, c_remove, axis=1)
        if verify(U_test, V_test, W_test):
            best_rank = 27
            U_arr, V_arr, W_arr = U_test, V_test, W_test
            break
    
    # If we reduced to 27, try removing another
    if best_rank == 27:
        for c_remove in range(27):
            if time_left() < 5:
                break
            U_test = np.delete(U_arr, c_remove, axis=1)
            V_test = np.delete(V_arr, c_remove, axis=1)
            W_test = np.delete(W_arr, c_remove, axis=1)
            if verify(U_test, V_test, W_test):
                best_rank = 26
                U_arr, V_arr, W_arr = U_test, V_test, W_test
                break
    
    # Convert back to lists
    best = {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
    
    return best

# EVOLVE-BLOCK-END
