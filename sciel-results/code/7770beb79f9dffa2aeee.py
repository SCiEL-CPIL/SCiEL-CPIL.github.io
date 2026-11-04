import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Build the target tensor T[i,j,k] = 1 for all valid indices
    # In the trace convention: C is p x n, indexed by (k,i)
    # T[i,j,k] = 1 means A[i,j]*B[j,k] contributes to C[k,i]... 
    # Actually C[i,k] = sum_j A[i,j]*B[j,k], and C is stored as W[k*n+i]
    # So C[k*n+i] = C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor entry T[i,j,k] = 1 (coefficient of A[i,j]*B[j,k] in C[i,k])
    
    # Build U, V, W for the trivial rank-48 decomposition first
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i*m + j, r] = 1
                V_trivial[j*p + k, r] = 1
                W_trivial[k*n + i, r] = 1
                r += 1
    
    # Now try to find a lower rank decomposition
    # Strategy: use the Strassen-based rank-42 solution and try to reduce it
    
    # Build the rank-42 Strassen solution
    rank_s = 42
    U = np.zeros((n * m, rank_s), dtype=float)
    V = np.zeros((m * p, rank_s), dtype=float)
    W = np.zeros((p * n, rank_s), dtype=float)
    
    def add_strassen_block(i_start, j_start, k_start, offset):
        col = offset
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1

        col = offset + 1
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] -= 1

        col = offset + 2
        U[(i_start+0)*m + (j_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1

        col = offset + 3
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] -= 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1

        col = offset + 4
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1

        col = offset + 5
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+0)][col] -= 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1

        col = offset + 6
        U[(i_start+0)*m + (j_start+1)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] -= 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1

    add_strassen_block(0, 0, 0, 0)
    add_strassen_block(0, 0, 2, 7)
    add_strassen_block(0, 0, 4, 14)
    add_strassen_block(0, 2, 0, 21)
    add_strassen_block(0, 2, 2, 28)
    add_strassen_block(0, 2, 4, 35)
    
    # Verify the Strassen solution
    np.random.seed(42)
    A_test = np.random.rand(n, m)
    B_test = np.random.rand(m, p)
    C_test = np.random.rand(p, n)
    
    AB = A_test @ B_test
    LHS = np.sum(AB * C_test.T)
    
    RHS = 0
    for r in range(rank_s):
        a_dot = np.sum(A_test * U[:, r].reshape(n, m))
        b_dot = np.sum(B_test * V[:, r].reshape(m, p))
        c_dot = np.sum(C_test * W[:, r].reshape(p, n))
        RHS += a_dot * b_dot * c_dot
    
    if not np.isclose(LHS, RHS, atol=1e-9):
        # Fallback to trivial
        return {"U": U_trivial.tolist(), "V": V_trivial.tolist(), "W": W_trivial.tolist()}
    
    # Try to find linear dependencies to reduce rank
    # Build the tensor as a 3D array and check for rank reduction
    # The tensor T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # We want to find if any column of the "unfolding" is linearly dependent
    
    # Check if we can remove terms by solving for dependencies
    # For each term r, check if it can be expressed as a combination of others
    # This is equivalent to checking if the tensor has rank < 42
    
    # Use a greedy approach: try removing each term and see if we can still
    # reconstruct the tensor with the remaining terms (with modified coefficients)
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_s
    
    # Try removing terms one at a time
    for remove_r in range(rank_s):
        if time.time() > deadline:
            break
        
        # Try to see if term remove_r is redundant
        # We need: U[:,remove_r]*V[:,remove_r]*W[:,remove_r] = sum_{r' != remove_r} c_{r'} U[:,r']*V[:,r']*W[:,r']
        # This is a tensor equation. Hard to solve directly.
        
        # Alternative: check if the tensor formed by all terms except remove_r
        # still spans the same space. Use random testing.
        
        U_test = U[:, [i for i in range(rank_s) if i != remove_r]].copy()
        V_test = V[:, [i for i in range(rank_s) if i != remove_r]].copy()
        W_test = W[:, [i for i in range(rank_s) if i != remove_r]].copy()
        
        # Check if this reduced set still gives the correct tensor
        # by testing with random inputs
        valid = True
        for seed in range(3):
            np.random.seed(seed)
            A_t = np.random.rand(n, m)
            B_t = np.random.rand(m, p)
            C_t = np.random.rand(p, n)
            
            AB_t = A_t @ B_t
            LHS_t = np.sum(AB_t * C_t.T)
            
            RHS_t = 0
            for r in range(rank_s - 1):
                a_dot = np.sum(A_t * U_test[:, r].reshape(n, m))
                b_dot = np.sum(B_t * V_test[:, r].reshape(m, p))
                c_dot = np.sum(C_t * W_test[:, r].reshape(p, n))
                RHS_t += a_dot * b_dot * c_dot
            
            if not np.isclose(LHS_t, RHS_t, atol=1e-9):
                valid = False
                break
        
        if valid:
            best_U, best_V, best_W = U_test, V_test, W_test
            best_rank = rank_s - 1
            # Reset U,V,W to the reduced version and continue
            U, V, W = U_test.copy(), V_test.copy(), W_test.copy()
            rank_s = best_rank
            # Restart the loop since indices changed
            break
    
    # Try a few more rounds of reduction
    for _ in range(3):
        if time.time() > deadline:
            break
        progress = False
        for remove_r in range(rank_s):
            if time.time() > deadline:
                break
            
            U_test = U[:, [i for i in range(rank_s) if i != remove_r]].copy()
            V_test = V[:, [i for i in range(rank_s) if i != remove_r]].copy()
            W_test = W[:, [i for i in range(rank_s) if i != remove_r]].copy()
            
            valid = True
            for seed in range(3):
                np.random.seed(seed + rank_s)
                A_t = np.random.rand(n, m)
                B_t = np.random.rand(m, p)
                C_t = np.random.rand(p, n)
                
                AB_t = A_t @ B_t
                LHS_t = np.sum(AB_t * C_t.T)
                
                RHS_t = 0
                for r in range(rank_s - 1):
                    a_dot = np.sum(A_t * U_test[:, r].reshape(n, m))
                    b_dot = np.sum(B_t * V_test[:, r].reshape(m, p))
                    c_dot = np.sum(C_t * W_test[:, r].reshape(p, n))
                    RHS_t += a_dot * b_dot * c_dot
                
                if not np.isclose(LHS_t, RHS_t, atol=1e-9):
                    valid = False
                    break
            
            if valid:
                U, V, W = U_test, V_test, W_test
                rank_s -= 1
                progress = True
                break
        
        if not progress:
            break
    
    if time.time() > deadline:
        # Return current best
        pass
    
    # Final verification
    np.random.seed(123)
    A_test = np.random.rand(n, m)
    B_test = np.random.rand(m, p)
    C_test = np.random.rand(p, n)
    
    AB = A_test @ B_test
    LHS = np.sum(AB * C_test.T)
    
    RHS = 0
    for r in range(rank_s):
        a_dot = np.sum(A_test * U[:, r].reshape(n, m))
        b_dot = np.sum(B_test * V[:, r].reshape(m, p))
        c_dot = np.sum(C_test * W[:, r].reshape(p, n))
        RHS += a_dot * b_dot * c_dot
    
    if not np.isclose(LHS, RHS, atol=1e-9):
        # Fallback to the original Strassen
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
