import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = 3, 3, 4
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Target rank to try
    target_ranks = [32, 33, 34, 35]
    
    best_U = None
    best_V = None
    best_W = None
    best_rank = n * m * p # 36
    
    # Initialize with schoolbook
    U = np.zeros((n * m, best_rank))
    V = np.zeros((m * p, best_rank))
    W = np.zeros((p * n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    best_U, best_V, best_W = U, V, W

    def verify(U_mat, V_mat, W_mat):
        n, m, p = 3, 3, 4
        rank = U_mat.shape[1]
        
        # Check non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        
        # Check zero entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    
                    if (i == i2 and j == j2 and k == k2):
                        continue # Already checked
                    
                    s = 0
                    for r in range(rank):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s) > 1e-9:
                        return False
        
        return True

    # Precompute valid (a,b,c) triples for the multiplication tensor
    valid_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_triples.append((a, b, c))
    
    valid_set = set(valid_triples)
    
    # Precompute all (a,b,c) triples
    all_triples = []
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                all_triples.append((a, b, c))
    
    def compute_error_fast(U_mat, V_mat, W_mat):
        """Compute error using vectorized operations."""
        rank = U_mat.shape[1]
        err = 0.0
        
        # For each rank component, compute the contribution to each tensor entry
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # We want T[a,b,c] = 1 if (a,b,c) in valid_set, else 0
        
        # Efficient approach: for each r, compute outer product contributions
        # But let's use a simpler approach: iterate over valid and invalid triples
        
        # For valid triples:
        for a, b, c in valid_triples:
            s = 0
            for r in range(rank):
                s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
            err += (s - 1.0) ** 2
        
        # For invalid triples:
        for a, b, c in all_triples:
            if (a, b, c) in valid_set:
                continue
            s = 0
            for r in range(rank):
                s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
            err += s ** 2
        
        return err

    # Try to find a lower rank decomposition
    for target_rank in target_ranks:
        if time.time() > deadline - 20:
            break
            
        # Multiple random restarts
        for restart in range(5):
            if time.time() > deadline - 10:
                break
                
            # Initialize with random half-integers
            U_rand = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, target_rank))
            V_rand = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, target_rank))
            W_rand = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, target_rank))
            
            current_error = compute_error_fast(U_rand, V_rand, W_rand)
            
            # Local search with proper revert
            for iteration in range(500):
                if time.time() > deadline - 5:
                    break
                
                # Pick a random element to perturb
                mat_idx = random.choice([0, 1, 2])
                row = random.randint(0, (n*m, m*p, p*n)[mat_idx] - 1)
                col = random.randint(0, target_rank - 1)
                
                # Store old value for revert
                if mat_idx == 0:
                    old_val = U_rand[row, col]
                    U_rand[row, col] += random.choice([-0.5, 0.5])
                elif mat_idx == 1:
                    old_val = V_rand[row, col]
                    V_rand[row, col] += random.choice([-0.5, 0.5])
                else:
                    old_val = W_rand[row, col]
                    W_rand[row, col] += random.choice([-0.5, 0.5])
                
                new_error = compute_error_fast(U_rand, V_rand, W_rand)
                
                if new_error < current_error:
                    current_error = new_error
                    if verify(U_rand, V_rand, W_rand):
                        best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_rand.copy()
                        best_rank = target_rank
                        break
                else:
                    # Revert exactly
                    if mat_idx == 0:
                        U_rand[row, col] = old_val
                    elif mat_idx == 1:
                        V_rand[row, col] = old_val
                    else:
                        W_rand[row, col] = old_val
                
                if best_rank == target_rank:
                    break
            
            if best_rank == target_rank:
                break
    
    # If no better solution found, return schoolbook
    if best_rank == 36:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
