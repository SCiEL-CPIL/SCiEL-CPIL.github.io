"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin

    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def validate(decomp):
        U = np.array(decomp["U"], dtype=float)
        V = np.array(decomp["V"], dtype=float)
        W = np.array(decomp["W"], dtype=float)
        
        n, m, p = DIMENSIONS
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        C_reconstructed = np.zeros((n, p))
        
        for r in range(U.shape[1]):
            u_r = U[:, r]
            v_r = V[:, r]
            w_r = W[:, r]
            for i in range(n):
                for k in range(p):
                    s = 0
                    for j in range(m):
                        u_val = u_r[i*m + j]
                        v_val = v_r[j*p + k]
                        w_val = w_r[k*n + i]
                        if u_val != 0 and v_val != 0 and w_val != 0:
                            s += u_val * v_val * w_val * A[i,j] * B[j,k]
                    C_reconstructed[i,k] += s
                    
        if not np.allclose(C_reconstructed, C_expected, atol=1e-9):
            return False
        return True

    # Try to find a valid decomposition with rank < 36
    # Use a structured search: try to find a 2x3x3 decomposition and reuse it
    
    def find_2x3x3_decomp(max_trials=2000):
        """Find a valid decomposition for 2x3x3 with rank <= 18 using structured search"""
        np.random.seed(42)
        
        for target_rank in range(15, 19):
            for trial in range(max_trials):
                if time.time() > deadline - 20:
                    return None
                
                # Use a more structured initialization based on combinatorial patterns
                U = np.zeros((6, target_rank))
                V = np.zeros((9, target_rank))
                W = np.zeros((6, target_rank))
                
                # Strategy: each rank-1 term should have a "support" pattern
                # that covers specific (i,j,k) tuples
                for r in range(target_rank):
                    # Randomly assign which indices are active in each matrix
                    # Use biased sampling to encourage overlap
                    u_nz = np.random.choice([2, 3, 4, 5], p=[0.2, 0.3, 0.3, 0.2])
                    v_nz = np.random.choice([2, 3, 4, 5], p=[0.2, 0.3, 0.3, 0.2])
                    w_nz = np.random.choice([2, 3, 4, 5], p=[0.2, 0.3, 0.3, 0.2])
                    
                    u_idx = np.random.choice(6, size=u_nz, replace=False)
                    v_idx = np.random.choice(9, size=v_nz, replace=False)
                    w_idx = np.random.choice(6, size=w_nz, replace=False)
                    
                    # Use values from a small set including fractions
                    vals = [-1, -0.5, 0.5, 1, 2, -2]
                    U[u_idx, r] = np.random.choice(vals, size=u_nz)
                    V[v_idx, r] = np.random.choice(vals, size=v_nz)
                    W[w_idx, r] = np.random.choice(vals, size=w_nz)
                
                # Check validity: T[i,j,k] = sum_r U[i*3+j,r] * V[j*3+k,r] * W[k*2+i,r] = 1
                valid = True
                for i in range(2):
                    for j in range(3):
                        for k in range(3):
                            t_val = 0.0
                            for r in range(target_rank):
                                t_val += U[i*3+j, r] * V[j*3+k, r] * W[k*2+i, r]
                            if abs(t_val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    return U, V, W
        
        # Try a deterministic construction: use all rank-1 terms with specific patterns
        # Based on the idea that we need to cover 18 (i,j,k) tuples
        for target_rank in range(15, 19):
            if time.time() > deadline - 20:
                return None
            
            # Try systematic patterns: each term covers a subset of indices
            U = np.zeros((6, target_rank))
            V = np.zeros((9, target_rank))
            W = np.zeros((6, target_rank))
            
            # For each rank term, choose a pattern
            for r in range(target_rank):
                # Pattern r: choose which i, j, k values are active
                # i in {0,1}, j in {0,1,2}, k in {0,1,2}
                # U index = i*3+j, V index = j*3+k, W index = k*2+i
                
                # Randomly select a combination
                i_val = np.random.choice(2)
                j_val = np.random.choice(3)
                k_val = np.random.choice(3)
                
                # Add some randomization: sometimes include neighboring indices
                extra_i = np.random.choice(2) if np.random.rand() < 0.3 else None
                extra_j = np.random.choice(3) if np.random.rand() < 0.3 else None
                extra_k = np.random.choice(3) if np.random.rand() < 0.3 else None
                
                u_indices = set([i_val*3 + j_val])
                v_indices = set([j_val*3 + k_val])
                w_indices = set([k_val*2 + i_val])
                
                if extra_i is not None:
                    u_indices.add(extra_i*3 + j_val)
                    w_indices.add(k_val*2 + extra_i)
                if extra_j is not None:
                    u_indices.add(i_val*3 + extra_j)
                    v_indices.add(extra_j*3 + k_val)
                if extra_k is not None:
                    v_indices.add(j_val*3 + extra_k)
                    w_indices.add(extra_k*2 + i_val)
                
                u_list = list(u_indices)
                v_list = list(v_indices)
                w_list = list(w_indices)
                
                # Assign values
                U[u_list[0], r] = 1.0
                V[v_list[0], r] = 1.0
                W[w_list[0], r] = 1.0
                
                # If there are extra indices, assign small values
                if len(u_list) > 1:
                    U[u_list[1], r] = 0.5
                if len(v_list) > 1:
                    V[v_list[1], r] = 0.5
                if len(w_list) > 1:
                    W[w_list[1], r] = 0.5
                
                # Check validity
                valid = True
                for i in range(2):
                    for j in range(3):
                        for k in range(3):
                            t_val = 0.0
                            for r2 in range(target_rank):
                                t_val += U[i*3+j, r2] * V[j*3+k, r2] * W[k*2+i, r2]
                            if abs(t_val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    return U, V, W
        
        return None

    # Try to find a 2x3x3 decomposition
    result = find_2x3x3_decomp()
    if result is not None:
        U_11, V_11, W_11 = result
        rank_11 = U_11.shape[1]
        
        # Extend to 2x3x6 by using two copies
        U_full = np.zeros((6, 2*rank_11))
        V_full = np.zeros((18, 2*rank_11))
        W_full = np.zeros((12, 2*rank_11))
        
        # Block 1: k in 0..2
        U_full[:, 0:rank_11] = U_11
        V_full[0:9, 0:rank_11] = V_11
        W_full[0:6, 0:rank_11] = W_11
        
        # Block 2: k in 3..5
        U_full[:, rank_11:2*rank_11] = U_11
        for j in range(3):
            for k_prime in range(3):
                V_full[j*6 + (3+k_prime), rank_11:2*rank_11] = V_11[j*3 + k_prime]
        for k_prime in range(3):
            for i in range(2):
                W_full[(3+k_prime)*2 + i, rank_11:2*rank_11] = W_11[k_prime*2 + i]
        
        decomp = {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
        if validate(decomp):
            return decomp

    # Fallback to Schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
