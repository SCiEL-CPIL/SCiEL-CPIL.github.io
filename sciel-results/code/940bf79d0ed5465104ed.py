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
                    
    # Try to find rank 20 for M_{2,2,6} directly
    target_rank = 20
    
    # Define the tensor T for M_{2,2,6}
    # T[(i,j), (j',k), (i',k')] = 1 if i==i', j==j', k==k'
    # Indices:
    # A index a = i*2 + j (0..3)
    # B index b = j'*6 + k (0..11)
    # C index c = i'*6 + k (0..11) -- Wait, W index is k*n+i.
    # Problem says: W[k*n+i, r].
    # So C index c = k*2 + i.
    # k is 0..5, i is 0..1.
    # c ranges 0..11.
    
    T = np.zeros((4, 12, 12))
    for i in range(2):
        for j in range(2):
            for k in range(6):
                a = i*2 + j
                b = j*6 + k
                c = k*2 + i
                T[a, b, c] = 1.0

    def residuals(params):
        U_s = params[0:target_rank*4].reshape(target_rank, 4)
        V_s = params[target_rank*4:target_rank*16].reshape(target_rank, 12)
        W_s = params[target_rank*16:].reshape(target_rank, 12)
        
        S = np.einsum('sa,sb,sc->abc', U_s, V_s, W_s)
        
        return (S - T).flatten()
    
    found = False
    best_params = None
    
    np.random.seed(42)
    
    # Try a few random initializations
    # Use small random values
    for _ in range(50):
        if time.time() > deadline - 20:
            break
            
        init = np.random.randn(target_rank * 28) * 0.1
        
        try:
            result = least_squares(residuals, init, method='lm', max_nfev=2000)
            if result.cost < 1e-8:
                found = True
                best_params = result.x
                break
        except Exception:
            pass
            
    if found and best_params is not None:
        U_sub = best_params[0:target_rank*4].reshape(target_rank, 4)
        V_sub = best_params[target_rank*4:target_rank*16].reshape(target_rank, 12)
        W_sub = best_params[target_rank*16:].reshape(target_rank, 12)
        
        U = [[0] * target_rank for _ in range(4)]
        V = [[0] * target_rank for _ in range(12)]
        W = [[0] * target_rank for _ in range(12)]
        
        for s in range(target_rank):
            for idx in range(4):
                U[idx][s] = U_sub[s, idx]
            for idx in range(12):
                V[idx][s] = V_sub[s, idx]
                W[idx][s] = W_sub[s, idx]
                
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": U_fb, "V": V_fb, "W": W_fb}
