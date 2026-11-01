import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Fallback solution: Rank 21
    rank = 21
    strassen = [
        {'U': [1, 0, 0, 1], 'V': [1, 0, 0, 1], 'W': [1, 0, 0, 1]},
        {'U': [0, 0, 1, 1], 'V': [1, 0, 0, 0], 'W': [0, 1, 0, -1]},
        {'U': [1, 0, 0, 0], 'V': [0, 1, 0, -1], 'W': [0, 0, 1, 1]},
        {'U': [0, 0, 0, 1], 'V': [-1, 0, 1, 0], 'W': [1, 1, 0, 0]},
        {'U': [1, 1, 0, 0], 'V': [0, 0, 0, 1], 'W': [-1, 0, 1, 0]},
        {'U': [-1, 0, 1, 0], 'V': [1, 1, 0, 0], 'W': [0, 0, 0, 1]},
        {'U': [0, 1, 0, -1], 'V': [0, 0, 1, 1], 'W': [1, 0, 0, 0]},
    ]
    
    U_fb = [[0] * rank for _ in range(4)]
    V_fb = [[0] * rank for _ in range(12)]
    W_fb = [[0] * rank for _ in range(12)]
    
    for b in range(3):
        for s in range(7):
            col = b * 7 + s
            sc = strassen[s]
            for idx in range(4):
                if sc['U'][idx] != 0:
                    U_fb[idx][col] = sc['U'][idx]
            v_idx = [2*b, 2*b+1, 6+2*b, 6+2*b+1]
            for idx in range(4):
                if sc['V'][idx] != 0:
                    V_fb[v_idx[idx]][col] = sc['V'][idx]
            w_idx = [4*b, 4*b+1, 4*b+2, 4*b+3]
            for idx in range(4):
                if sc['W'][idx] != 0:
                    W_fb[w_idx[idx]][col] = sc['W'][idx]
                    
    # Try to find rank 10 for M_{2,2,3}
    # If found, use it to build rank 20 for M_{2,2,6}
    
    n_sub, m_sub, p_sub = 2, 2, 3
    target_rank = 10
    
    # Define the tensor T for M_{2,2,3}
    # T[(i,j), (j',k), (i',k')] = 1 if i==i', j==j', k==k'
    # We want sum_s U[(i,j),s] V[(j',k),s] W[(i',k'),s] == T[(i,j), (j',k), (i',k')]
    
    # Residual function
    def residuals(params):
        # params is a flat vector of length target_rank * (4 + 6 + 6)
        U_s = params[0:target_rank*4].reshape(target_rank, 4)
        V_s = params[target_rank*4:target_rank*10].reshape(target_rank, 6)
        W_s = params[target_rank*10:].reshape(target_rank, 6)
        
        # Compute the tensor
        # Residuals are the differences from the target tensor
        # We only need to check the non-zero and zero constraints
        # Total entries: 4 * 6 * 6 = 144
        # But we can just compute the full tensor and subtract the target
        
        # Efficient computation:
        # For each s, U_s[:, :, None] * V_s[:, :, None] * W_s[:, :, None]?
        # U_s is (rank, 4). V_s is (rank, 6). W_s is (rank, 6).
        # Tensor S is (4, 6, 6).
        # S[a, b, c] = sum_s U_s[s, a] * V_s[s, b] * W_s[s, c]
        
        S = np.einsum('sa,sb,sc->abc', U_s, V_s, W_s)
        
        # Target tensor T
        # T[a, b, c] = 1 if a//2 == c//2 (i==i'), a%2 == b//2 (j==j'), b%2 == c%2 (k==k')?
        # Indices:
        # A index a = i*2 + j
        # B index b = j'*3 + k
        # C index c = i'*3 + k'
        
        # Condition: i == i', j == j', k == k'
        # i = a//2, j = a%2
        # j' = b//3, k = b%3
        # i' = c//3, k' = c%3
        
        T = np.zeros((4, 6, 6))
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    a = i*2 + j
                    b = j*3 + k
                    c = i*3 + k
                    T[a, b, c] = 1.0
                    
        return (S - T).flatten()
    
    found = False
    best_params = None
    
    # Try a few random initializations
    np.random.seed(42)
    for _ in range(20):
        if time.time() > deadline - 20:
            break
            
        # Initialize with small random values
        init = np.random.randn(target_rank * 16) * 0.1
        # Add some structure?
        
        try:
            result = least_squares(residuals, init, method='lm', max_nfev=1000)
            if result.cost < 1e-8:
                found = True
                best_params = result.x
                break
        except Exception:
            pass
            
    if found and best_params is not None:
        # Extract U, V, W for M_{2,2,3}
        U_sub = best_params[0:target_rank*4].reshape(target_rank, 4)
        V_sub = best_params[target_rank*4:target_rank*10].reshape(target_rank, 6)
        W_sub = best_params[target_rank*10:].reshape(target_rank, 6)
        
        # Construct M_{2,2,6} using two blocks of M_{2,2,3}
        rank_total = 20
        U = [[0] * rank_total for _ in range(4)]
        V = [[0] * rank_total for _ in range(12)]
        W = [[0] * rank_total for _ in range(12)]
        
        for b in range(2):
            for s in range(target_rank):
                col = b * target_rank + s
                for idx in range(4):
                    U[idx][col] = U_sub[s, idx]
                
                # V mapping:
                # V_sub indices: 0..5 correspond to (j', k) for k=0,1,2
                # V_total indices: j'*6 + k
                # Block b corresponds to k=3b, 3b+1, 3b+2
                for j_prime in range(2):
                    for k_local in range(3):
                        v_sub_idx = j_prime * 3 + k_local
                        v_total_idx = j_prime * 6 + (3 * b + k_local)
                        V[v_total_idx][col] = V_sub[s, v_sub_idx]
                        
                # W mapping:
                # W_sub indices: 0..5 correspond to (i', k') for k'=0,1,2
                # W_total indices: i'*6 + k'? No, W is (p*n, rank) = (12, rank).
                # Wait, W shape is (p*n, rank).
                # In the problem description: W[k*n+i, r].
                # So W index is k*n + i.
                # For M_{2,2,3}, p=3, n=2. W index is k'*2 + i'.
                # For M_{2,2,6}, p=6, n=2. W index is k''*2 + i''.
                # Block b corresponds to k'' = 3b, 3b+1, 3b+2.
                
                for i_prime in range(2):
                    for k_local in range(3):
                        w_sub_idx = k_local * 2 + i_prime
                        w_total_idx = (3 * b + k_local) * 2 + i_prime
                        W[w_total_idx][col] = W_sub[s, w_sub_idx]
                        
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": U_fb, "V": V_fb, "W": W_fb}
