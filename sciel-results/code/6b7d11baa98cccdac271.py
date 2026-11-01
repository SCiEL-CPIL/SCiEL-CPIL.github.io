"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor as a 24-dim vector
    # T[(i,j),(j,k),(k,i)] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Flatten index for each mode:
    #   U index: i*m + j  (range [0, nm))
    #   V index: j*p + k  (range [0, mp))
    #   W index: k*n + i  (range [0, pn))
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                tensor_entries.append((u_idx, v_idx, w_idx))
    
    num_entries = len(tensor_entries)  # should be 24
    
    # Generate schoolbook as fallback (rank 24)
    best_rank = n * m * p
    U_sb = [[0] * best_rank for _ in range(nm)]
    V_sb = [[0] * best_rank for _ in range(mp)]
    W_sb = [[0] * best_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Strategy: Exploit algebraic structure
    # For i=0: entries are (u_idx=j, v_idx=j*p+k, w_idx=k) for j in [0,3), k in [0,4)
    # This is a 1x3x4 tensor with 12 nonzero entries
    # For i=1: entries are (u_idx=3+j, v_idx=j*p+k, w_idx=4+k) for j in [0,3), k in [0,4)
    
    # The 1x3x4 tensor has rank at most 12 (schoolbook), but can be lower
    # Known result: 1xn x nx1 has rank n^2 - n + 1 for n >= 3
    # For n=3: rank = 7, for n=4: rank = 13
    # But we have 1x3x4, not square
    
    # Let's try to find rank-9 decomposition for the full tensor
    # by exploiting the structure
    
    # Key insight: We can write T as sum over i of T_i where T_i is the i-slice
    # T_0: u_idx=j, v_idx=j*p+k, w_idx=k
    # T_1: u_idx=3+j, v_idx=j*p+k, w_idx=4+k
    
    # If we find rank-r decompositions for each slice, we get rank-2r for the whole
    # But we might do better by sharing structure
    
    # Try rank 15-18 with structured approach
    
    np.random.seed(42)
    
    # For each target rank, try multiple approaches:
    # 1. Random initialization + coordinate descent
    # 2. Structured initialization based on slice decompositions
    
    for target_rank in range(15, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        found = False
        
        # Approach 1: Random restarts with coordinate descent
        num_restarts = 30
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random half-integer values
            U_init = np.random.choice([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], size=(nm, target_rank))
            V_init = np.random.choice([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], size=(mp, target_rank))
            W_init = np.random.choice([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], size=(pn, target_rank))
            
            U_curr = U_init.copy()
            V_curr = V_init.copy()
            W_curr = W_init.copy()
            
            # Iterative refinement with coordinate descent
            for iteration in range(100):
                if time.time() > deadline - 3:
                    break
                
                # Solve for W given U, V (linear system)
                A_mat = np.zeros((num_entries, pn * target_rank))
                b_vec = np.ones(num_entries)
                
                for e_idx, (u_idx, v_idx, w_idx) in enumerate(tensor_entries):
                    for r in range(target_rank):
                        col_idx = w_idx * target_rank + r
                        A_mat[e_idx, col_idx] = U_curr[u_idx, r] * V_curr[v_idx, r]
                
                W_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W_curr = W_flat.reshape(pn, target_rank)
                W_curr = np.round(W_curr * 2) / 2
                
                # Solve for U given V, W
                A_mat_U = np.zeros((num_entries, nm * target_rank))
                for e_idx, (u_idx, v_idx, w_idx) in enumerate(tensor_entries):
                    for r in range(target_rank):
                        col_idx = u_idx * target_rank + r
                        A_mat_U[e_idx, col_idx] = V_curr[v_idx, r] * W_curr[w_idx, r]
                
                U_flat, _, _, _ = np.linalg.lstsq(A_mat_U, b_vec, rcond=None)
                U_curr = U_flat.reshape(nm, target_rank)
                U_curr = np.round(U_curr * 2) / 2
                
                # Solve for V given U, W
                A_mat_V = np.zeros((num_entries, mp * target_rank))
                for e_idx, (u_idx, v_idx, w_idx) in enumerate(tensor_entries):
                    for r in range(target_rank):
                        col_idx = v_idx * target_rank + r
                        A_mat_V[e_idx, col_idx] = U_curr[u_idx, r] * W_curr[w_idx, r]
                
                V_flat, _, _, _ = np.linalg.lstsq(A_mat_V, b_vec, rcond=None)
                V_curr = V_flat.reshape(mp, target_rank)
                V_curr = np.round(V_curr * 2) / 2
                
                # Check if we have an exact solution
                error = 0.0
                for e_idx, (u_idx, v_idx, w_idx) in enumerate(tensor_entries):
                    val = 0.0
                    for r in range(target_rank):
                        val += U_curr[u_idx, r] * V_curr[v_idx, r] * W_curr[w_idx, r]
                    error += abs(val - 1.0)
                
                if error < 1e-9:
                    # Found exact solution!
                    best_rank = target_rank
                    U_list = [[float(x) for x in U_curr[i]] for i in range(nm)]
                    V_list = [[float(x) for x in V_curr[i]] for i in range(mp)]
                    W_list = [[float(x) for x in W_curr[i]] for i in range(pn)]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    found = True
                    break
            
            if found:
                break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
