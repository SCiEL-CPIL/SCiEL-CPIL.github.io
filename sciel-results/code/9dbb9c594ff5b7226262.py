import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_sb
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Use a richer set of half-integer values
    values = np.array([Fraction(x, 2) for x in range(-8, 9)])
    
    def to_float(vals):
        return np.array([float(v) for v in vals])
    
    # Simulated annealing for a given rank
    def simulate_annealing(r, max_iters, deadline):
        # Greedy construction initialization
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        # Initialize with a structured pattern
        for c in range(r):
            # Assign random half-integers to create diversity
            for i in range(n):
                for j in range(m):
                    U_3d[i, j, c] = random.choice(values)
            for j in range(m):
                for k in range(p):
                    V_3d[j, k, c] = random.choice(values)
            for k in range(p):
                for i in range(n):
                    W_3d[k, i, c] = random.choice(values)
        
        # Convert to float for computation
        U_f = to_float(U_3d.flatten().reshape(n*m, r))
        V_f = to_float(V_3d.flatten().reshape(m*p, r))
        W_f = to_float(W_3d.flatten().reshape(p*n, r))
        
        U_3df = U_f.reshape(n, m, r)
        V_3df = V_f.reshape(m, p, r)
        W_3df = W_f.reshape(p, n, r)
        
        # Compute T and error
        T = np.einsum('ijc,jkc,kic->ijk', U_3df, V_3df, W_3df)
        error = float(np.sum((T - 1) ** 2))
        
        best_error = error
        best_U_3d = U_3df.copy()
        best_V_3d = V_3df.copy()
        best_W_3d = W_3df.copy()
        
        # Simulated annealing parameters
        T_start = 10.0
        T_end = 0.01
        cooling_rate = (T_end / T_start) ** (1.0 / max(1, max_iters - 1))
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Temperature
            temp = T_start * (cooling_rate ** it)
            
            # Randomly select a variable to change
            var_type = np.random.randint(3)
            
            if var_type == 0:  # Change U[i,j,c]
                i = np.random.randint(n)
                j = np.random.randint(m)
                c = np.random.randint(r)
                old_val = U_3df[i, j, c]
                
                # Try a random new value
                new_val = random.choice(values)
                if new_val == old_val:
                    continue
                new_val_f = float(new_val)
                diff = new_val_f - old_val
                if diff == 0:
                    continue
                
                # Compute delta in error
                delta_T = diff * (V_3df[j, :, c] * W_3df[:, i, c])
                T_new = T[i, j, :] + delta_T
                new_err_slice = float(np.sum((T_new - 1) ** 2))
                old_err_slice = float(np.sum((T[i, j, :] - 1) ** 2))
                new_err = error + new_err_slice - old_err_slice
                
                # Acceptance criterion
                if new_err < error or random.random() < np.exp(-(new_err - error) / temp):
                    T[i, j, :] = T_new
                    U_3df[i, j, c] = new_val_f
                    error = new_err
                    
                    if error < best_error - 1e-9:
                        best_error = error
                        best_U_3d = U_3df.copy()
                        best_V_3d = V_3df.copy()
                        best_W_3d = W_3df.copy()
                        
            elif var_type == 1:  # Change V[j,k,c]
                j = np.random.randint(m)
                k = np.random.randint(p)
                c = np.random.randint(r)
                old_val = V_3df[j, k, c]
                
                new_val = random.choice(values)
                if new_val == old_val:
                    continue
                new_val_f = float(new_val)
                diff = new_val_f - old_val
                if diff == 0:
                    continue
                
                delta_T = (U_3df[:, j, c] * W_3df[k, :, c]) * diff
                T_new = T[:, j, k] + delta_T
                new_err_slice = float(np.sum((T_new - 1) ** 2))
                old_err_slice = float(np.sum((T[:, j, k] - 1) ** 2))
                new_err = error + new_err_slice - old_err_slice
                
                if new_err < error or random.random() < np.exp(-(new_err - error) / temp):
                    T[:, j, k] = T_new
                    V_3df[j, k, c] = new_val_f
                    error = new_err
                    
                    if error < best_error - 1e-9:
                        best_error = error
                        best_U_3d = U_3df.copy()
                        best_V_3d = V_3df.copy()
                        best_W_3d = W_3df.copy()
                        
            else:  # Change W[k,i,c]
                k = np.random.randint(p)
                i = np.random.randint(n)
                c = np.random.randint(r)
                old_val = W_3df[k, i, c]
                
                new_val = random.choice(values)
                if new_val == old_val:
                    continue
                new_val_f = float(new_val)
                diff = new_val_f - old_val
                if diff == 0:
                    continue
                
                delta_T = (U_3df[i, :, c] * V_3df[:, k, c]) * diff
                T_new = T[i, :, k] + delta_T
                new_err_slice = float(np.sum((T_new - 1) ** 2))
                old_err_slice = float(np.sum((T[i, :, k] - 1) ** 2))
                new_err = error + new_err_slice - old_err_slice
                
                if new_err < error or random.random() < np.exp(-(new_err - error) / temp):
                    T[i, :, k] = T_new
                    W_3df[k, i, c] = new_val_f
                    error = new_err
                    
                    if error < best_error - 1e-9:
                        best_error = error
                        best_U_3d = U_3df.copy()
                        best_V_3d = V_3df.copy()
                        best_W_3d = W_3df.copy()
        
        # Check if we found an exact solution
        if best_error < 1e-9:
            # Verify with exact arithmetic
            U_exact = np.zeros((n, m, r))
            V_exact = np.zeros((m, p, r))
            W_exact = np.zeros((p, n, r))
            for i in range(n):
                for j in range(m):
                    for c in range(r):
                        U_exact[i, j, c] = Fraction(best_U_3d[i, j, c]).limit_denominator(2)
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        V_exact[j, k, c] = Fraction(best_V_3d[j, k, c]).limit_denominator(2)
            for k in range(p):
                for i in range(n):
                    for c in range(r):
                        W_exact[k, i, c] = Fraction(best_W_3d[k, i, c]).limit_denominator(2)
            
            # Verify T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == 1
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = Fraction(0)
                        for c in range(r):
                            s += U_exact[i, j, c] * V_exact[j, k, c] * W_exact[k, i, c]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert to float arrays for return
                U_ret = np.array([[float(U_exact[i, j, c]) for c in range(r)] for i in range(n) for j in range(m)])
                V_ret = np.array([[float(V_exact[j, k, c]) for c in range(r)] for j in range(m) for k in range(p)])
                W_ret = np.array([[float(W_exact[k, i, c]) for c in range(r)] for k in range(p) for i in range(n)])
                return r, U_ret, V_ret, W_ret
        
        return None
    
    # Search for lower rank using simulated annealing
    # Try ranks from 50 down to 20
    for r in range(50, 19, -1):
        if time.time() > deadline:
            break
        
        found = False
        # Multiple restarts with simulated annealing
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Determine iterations based on remaining time
            remaining = deadline - time.time()
            max_iters = min(5000, max(500, int(remaining * 50)))
            
            result = simulate_annealing(r, max_iters, deadline)
            if result is not None:
                best_rank, best_U, best_V, best_W = result
                found = True
                break
        
        if found:
            continue
    
    def to_list(mat):
        return mat.tolist()

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
