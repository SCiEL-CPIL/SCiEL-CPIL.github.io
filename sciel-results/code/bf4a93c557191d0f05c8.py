import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Build a rank-28 decomposition using Strassen on 2x2 blocks.
    # The multiplication tensor T(i,j,k) = 1 for all i,j,k.
    # We split the 3 intermediate columns into [0:2] and [2:3].
    # Split the 5 output columns into [0:2], [2:4], [4:5].
    # Use Strassen (rank 7) for 2x2*2x2 blocks, schoolbook for the rest.

    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0

    def strassen_block(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        # M2 = (A10+A11)B00
        U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
        col += 1
        # M3 = A00(B01-B11)
        U[i1*m+j1, col] = 1
        V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
        W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        # M4 = A11(B10-B00)
        U[i2*m+j2, col] = 1
        V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
        W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
        col += 1
        # M5 = (A00+A01)B11
        U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
        V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
        W[k2*n+i2, col] = 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
        V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1
        col += 1

    def schoolbook_block(i_list, j_list, k_list):
        nonlocal col
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    # Block 1: A[:,0:2] * B[0:2,0:2] -> C[0:2,0:2]  (Strassen, rank 7)
    strassen_block(0, 1, 0, 1, 0, 1)

    # Block 2: A[:,0:2] * B[0:2,2:4] -> C[0:2,2:4]  (Strassen, rank 7)
    strassen_block(0, 1, 0, 1, 2, 3)

    # Block 3: A[:,0:2] * B[0:2,4:5] -> C[0:2,4:5]  (Schoolbook, rank 4)
    schoolbook_block([0, 1], [0, 1], [4])

    # Block 4: A[:,2:3] * B[2:3,0:2] -> C[0:2,0:2]  (Schoolbook, rank 4)
    schoolbook_block([0, 1], [2], [0, 1])

    # Block 5: A[:,2:3] * B[2:3,2:4] -> C[0:2,2:4]  (Schoolbook, rank 4)
    schoolbook_block([0, 1], [2], [2, 3])

    # Block 6: A[:,2:3] * B[2:3,4:5] -> C[0:2,4:5]  (Schoolbook, rank 2)
    schoolbook_block([0, 1], [2], [4])

    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, r):
        tensor = np.zeros((n, m, p))
        for c in range(r):
            uc = U_mat[:, c].reshape(n, m)
            vc = V_mat[:, c].reshape(m, p)
            wc = W_mat[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
        return np.allclose(tensor, 1.0, atol=1e-9)

    # Try to reduce rank by peeling off redundant terms
    # Strategy: check if any two terms can be merged or if a term is redundant
    # For rank-1 peeling: try removing each term and see if remaining terms still work
    # This is expensive, so we'll use a smarter approach
    
    # First, verify our base decomposition
    if not verify(U, V, W, rank):
        # Fallback: full schoolbook rank 30
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank-1 peeling: check if we can remove any single term
    # This would reduce rank from 28 to 27
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Check each term for redundancy
    for remove_idx in range(rank):
        if time.time() > deadline - 10:
            break
            
        # Create modified matrices without term remove_idx
        U_test = np.delete(U, remove_idx, axis=1)
        V_test = np.delete(V, remove_idx, axis=1)
        W_test = np.delete(W, remove_idx, axis=1)
        
        if verify(U_test, V_test, W_test, rank - 1):
            best_rank = rank - 1
            best_U, best_V, best_W = U_test, V_test, W_test
            # Reset to try further reductions
            U, V, W = U_test, V_test, W_test
            rank = rank - 1
            # Restart the peeling process with the new smaller matrix
            for remove_idx2 in range(rank):
                if time.time() > deadline - 10:
                    break
                U_test2 = np.delete(U, remove_idx2, axis=1)
                V_test2 = np.delete(V, remove_idx2, axis=1)
                W_test2 = np.delete(W, remove_idx2, axis=1)
                
                if verify(U_test2, V_test2, W_test2, rank - 1):
                    best_rank = rank - 1
                    best_U, best_V, best_W = U_test2, V_test2, W_test2
                    U, V, W = U_test2, V_test2, W_test2
                    rank = rank - 1
                    break
    
    # If we found a reduction, use it
    if best_rank < 28:
        U, V, W = best_U, best_V, best_W
        rank = best_rank
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
