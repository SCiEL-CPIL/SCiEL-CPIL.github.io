"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s to stay safely under 150s
    
    n, m, p = DIMENSIONS
    # Schoolbook decomposition of rank n*m*p
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_rank = schoolbook_rank
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Strategy: Decompose 2x3x6 into two 2x3x3 blocks.
    # If we find a rank R decomposition for 2x3x3, we can get rank 2R for 2x3x6.
    # Known rank for 2x3x3 is 12. Schoolbook is 18.
    # So if we find R < 18, we improve on 36.
    
    def construct_2x3x3_decomposition(rank):
        """
        Construct a rank-r decomposition for 2x3x3 matrix multiplication.
        Returns (U, V, W) as numpy arrays or None.
        """
        n2, m2, p2 = 2, 3, 3
        
        def compute_tensor(U, V, W):
            """Compute the full tensor T[i,j,k] = sum_r U[i*m2+j,r]*V[j*p2+k,r]*W[k*n2+i,r]"""
            T = np.zeros((n2, m2, p2))
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        T[i,j,k] = np.sum(U[i*m2+j, :] * V[j*p2+k, :] * W[k*n2+i, :])
            return T
        
        def compute_errors(U, V, W):
            """Compute errors = T - 1"""
            T = compute_tensor(U, V, W)
            return T - 1.0
        
        def compute_gradients(U, V, W):
            """Compute gradients of sum of squared errors"""
            errors = compute_errors(U, V, W)
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        e = errors[i,j,k]
                        for r in range(rank):
                            grad_U[i*m2+j, r] += 2 * e * V[j*p2+k, r] * W[k*n2+i, r]
                            grad_V[j*p2+k, r] += 2 * e * U[i*m2+j, r] * W[k*n2+i, r]
                            grad_W[k*n2+i, r] += 2 * e * U[i*m2+j, r] * V[j*p2+k, r]
            
            return grad_U, grad_V, grad_W
        
        best_result = None
        best_loss = float('inf')
        
        # Multiple restarts
        # We try a few seeds and learning rates
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(seed)
            U = np.random.randn(n2 * m2, rank) * 0.5
            V = np.random.randn(m2 * p2, rank) * 0.5
            W = np.random.randn(p2 * n2, rank) * 0.5
            
            lr = 0.01
            prev_loss = float('inf')
            patience = 0
            max_patience = 1000
            
            for iteration in range(2000):
                if time.time() > deadline - 5:
                    break
                
                errors = compute_errors(U, V, W)
                loss = np.sum(errors ** 2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_result = (U.copy(), V.copy(), W.copy())
                
                if loss < 1e-12:
                    break
                
                # Adaptive learning rate
                if loss < prev_loss * 0.99:
                    lr = min(lr * 1.05, 0.05)
                    patience = 0
                else:
                    patience += 1
                    if patience > max_patience:
                        lr *= 0.5
                        patience = 0
                        if lr < 1e-6:
                            break
                
                prev_loss = loss
                
                grad_U, grad_V, grad_W = compute_gradients(U, V, W)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
            
            # Final check
            if best_result is not None:
                U_final, V_final, W_final = best_result
                errors = compute_errors(U_final, V_final, W_final)
                if np.max(np.abs(errors)) < 1e-8:
                    return U_final, V_final, W_final
        
        return None

    # Try ranks from 12 up to 17
    # 12 is the known minimal rank
    for R in range(12, 18):
        if time.time() > deadline - 10:
            break
            
        decomp_2x3x3 = construct_2x3x3_decomposition(R)
        
        if decomp_2x3x3 is not None:
            U_2x3x3, V_2x3x3, W_2x3x3 = decomp_2x3x3
            
            # Construct 2x3x6 decomposition from two copies
            # U (6, 2R), V (18, 2R), W (12, 2R)
            R_total = 2 * R
            U_2x3x6 = np.zeros((n * m, R_total))
            V_2x3x6 = np.zeros((m * p, R_total))
            W_2x3x6 = np.zeros((p * n, R_total))
            
            # First copy (columns 0 to R-1)
            # Corresponds to B columns 0-2 (indices 0-2 in V, 0-5 in W? No, W is indexed by k*n+i)
            # For 2x3x3:
            # U is 2x3 (6 rows)
            # V is 3x3 (9 rows)
            # W is 3x2 (6 rows)
            
            # In 2x3x6:
            # U is 2x3 (6 rows)
            # V is 3x6 (18 rows)
            # W is 6x2 (12 rows)
            
            # Copy 1:
            # U rows 0-5: U_2x3x3
            # V rows 0-8: V_2x3x3 (cols 0-2 of B)
            # V rows 9-17: 0
            # W rows 0-5: W_2x3x3 (cols 0-2 of C, i.e., k=0,1,2)
            # W rows 6-11: 0
            
            U_2x3x6[:, 0:R] = U_2x3x3
            V_2x3x6[0:9, 0:R] = V_2x3x3
            W_2x3x6[0:6, 0:R] = W_2x3x3
            
            # Copy 2:
            # U rows 0-5: U_2x3x3 (same A)
            # V rows 0-8: 0
            # V rows 9-17: V_2x3x3 (cols 3-5 of B, shifted by 3)
            # W rows 0-5: 0
            # W rows 6-11: W_2x3x3 (cols 3-5 of C, shifted by 3)
            
            U_2x3x6[:, R:2*R] = U_2x3x3
            V_2x3x6[9:18, R:2*R] = V_2x3x3
            W_2x3x6[6:12, R:2*R] = W_2x3x3
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(R_total):
                            val += U_2x3x6[i*m+j, r] * V_2x3x6[j*p+k, r] * W_2x3x6[k*n+i, r]
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_U = U_2x3x6.tolist()
                best_V = V_2x3x6.tolist()
                best_W = W_2x3x6.tolist()
                best_rank = R_total
                break # Found a valid decomposition, stop searching for higher ranks

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
