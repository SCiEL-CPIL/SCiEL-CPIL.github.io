# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = 3, 5, 5
    start_time = time.time()
    deadline = 140
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = (U_school, V_school, W_school)
    best_rank = rank_school

    # Strategy: Find a decomposition for 1x5x5, then replicate for 3 blocks.
    # The tensor for 1x5x5 is T_{j,k} = 1 for all j,k in 0..4.
    # We need U (5 x r), V (5 x 5 x r? No, V is indexed by j,k in the tensor, but in the factorization V is (m*p, rank).
    # For 1x5x5, m=5, p=5. V is (25, r).
    # Equation: sum_c U[j, c] * V[j*5+k, c] * W[k, c] = 1 for all j,k.
    
    def find_1x5x5_decomp(target_rank, max_attempts, seed):
        rng = np.random.RandomState(seed)
        for _ in range(max_attempts):
            if time.time() > deadline:
                return None
            
            U_rand = rng.randint(-1, 2, size=(5, target_rank)).astype(float)
            V_rand = rng.randint(-1, 2, size=(25, target_rank)).astype(float)
            
            # Ensure rows are not zero
            for idx in range(5):
                if np.all(U_rand[idx] == 0):
                    U_rand[idx, rng.randint(target_rank)] = 1.0
            for idx in range(25):
                if np.all(V_rand[idx] == 0):
                    V_rand[idx, rng.randint(target_rank)] = 1.0
            
            W_rand = np.zeros((5, target_rank))
            valid = True
            
            # For each k, we have a system for j=0..4:
            # sum_c U[j,c] V[j*5+k,c] W[k,c] = 1
            # Let A_k[j, c] = U[j,c] * V[j*5+k,c].
            # Then A_k @ W[k,:] = 1.
            
            for k in range(5):
                A_k = np.zeros((5, target_rank))
                for j in range(5):
                    A_k[j, :] = U_rand[j, :] * V_rand[j*5+k, :]
                
                try:
                    W_row, residuals, rank_A, sv = np.linalg.lstsq(A_k, np.ones(5), rcond=1e-12)
                    if np.linalg.norm(A_k @ W_row - np.ones(5)) < 1e-10:
                        W_rand[k, :] = W_row
                    else:
                        valid = False
                        break
                except:
                    valid = False
                    break
            
            if valid:
                # Verify
                for j in range(5):
                    for k in range(5):
                        s = 0.0
                        for c in range(target_rank):
                            s += U_rand[j, c] * V_rand[j*5+k, c] * W_rand[k, c]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    return (U_rand, V_rand, W_rand)
        
        return None

    # Try to find low rank for 1x5x5
    # Start from 20 and go up
    # If we find rank r, total rank is 3*r
    # Schoolbook is 75. So if r < 25, we improve.
    
    # We will try ranks 20..24
    found_1x5x5 = None
    for r in range(20, 25):
        if time.time() > deadline - 10:
            break
        attempts = 100 if r < 23 else 50
        res = find_1x5x5_decomp(r, attempts, seed=42 + r)
        if res is not None:
            found_1x5x5 = (r, res)
            break
            
    if found_1x5x5 is not None:
        r_1, (U_1, V_1, W_1) = found_1x5x5
        total_rank = 3 * r_1
        
        if total_rank < best_rank:
            U_new = [[0] * total_rank for _ in range(n * m)]
            V_new = [[0] * total_rank for _ in range(m * p)]
            W_new = [[0] * total_rank for _ in range(p * n)]
            
            for i in range(n):
                for j in range(m):
                    for c in range(r_1):
                        U_new[i * m + j][i * r_1 + c] = U_1[j, c]
                        
                for j in range(m):
                    for k in range(p):
                        for c in range(r_1):
                            V_new[j * p + k][i * r_1 + c] = V_1[j * p + k, c]
                            
                for k in range(p):
                    for c in range(r_1):
                        W_new[k * n + i][i * r_1 + c] = W_1[k, c]
            
            # Verify full solution
            if _verify(U_new, V_new, W_new, n, m, p):
                best_result = (U_new, V_new, W_new)
                best_rank = total_rank
    else:
        # If no improvement found, try a generic random search for the full tensor with rank < 75
        # Just a few attempts to see if we get lucky
        for r in range(70, 75):
             if time.time() > deadline - 5:
                 break
             # Re-use the logic from before but maybe smarter?
             # For now, stick to the block structure if it fails, return schoolbook.
             pass

    U, V, W = best_result
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # Check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    
    rank = len(U[0]) if U else 0
    
    # Optimization: Vectorized check using numpy
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Reshape U to (n, m, rank)
    U_reshaped = U_np.reshape(n, m, rank)
    # Reshape V to (m, p, rank)
    V_reshaped = V_np.reshape(m, p, rank)
    # Reshape W to (p, n, rank)
    W_reshaped = W_np.reshape(p, n, rank)
    
    # We need T[i, j, k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # For each i, j, k:
    # Sum over c: U_reshaped[i,j,c] * V_reshaped[j,k,c] * W_reshaped[k,i,c]
    
    # This is a bit tricky to do in one go without broadcasting to full n*m*p*rank which is 3*5*5*75 ~ 56k, which is fine.
    
    # Let's just loop i, j, k for simplicity and speed since rank is small-ish
    # Actually, let's use einsum if possible or just loops.
    # n=3, m=5, p=5. Total 75 checks.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
