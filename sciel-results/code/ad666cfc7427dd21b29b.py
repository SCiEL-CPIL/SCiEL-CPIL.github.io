import time
import numpy as np

def construct():
    n, m, p = 3, 5, 7
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Schoolbook decomposition: rank = n*m*p = 105
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    def verify(U, V, W, n, m, p):
        rank = U.shape[1]
        for k in range(p):
            for i in range(n):
                a = k * n + i
                for ii in range(n):
                    for j in range(m):
                        b = ii * m + j
                        for jj in range(m):
                            for kk in range(p):
                                c = jj * p + kk
                                expected = 1.0 if (i == ii and j == jj and k == kk) else 0.0
                                actual = 0.0
                                for r in range(rank):
                                    actual += U[b, r] * V[c, r] * W[a, r]
                                if abs(actual - expected) > 1e-9:
                                    return False
        return True
    
    # The schoolbook solution is guaranteed to work
    best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    best_rank = rank
    
    # Try to find a better solution with less rank using random search
    # Lower bound: max(n*m, m*p, p*n) = 35
    lower_bound = max(n * m, m * p, p * n)
    
    # Attempt random search for lower ranks
    # This is very unlikely to succeed, but worth trying
    for trial_rank in range(100, lower_bound, -5):
        if time.time() > deadline - 30:
            break
        
        for attempt in range(5):
            if time.time() > deadline - 30:
                break
            
            # Random initialization with half-integers
            np.random.seed(trial_rank * 1000 + attempt)
            U_r = np.random.randint(0, 4, size=(n * m, trial_rank)) * 0.5
            V_r = np.random.randint(0, 4, size=(m * p, trial_rank)) * 0.5
            W_r = np.random.randint(0, 4, size=(p * n, trial_rank)) * 0.5
            
            # Verify
            if verify(U_r, V_r, W_r, n, m, p):
                best_rank = trial_rank
                best_result = {'U': U_r.tolist(), 'V': V_r.tolist(), 'W': W_r.tolist()}
                break
    
    return best_result
