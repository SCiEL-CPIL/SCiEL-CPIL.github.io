import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    start_time = time.time()
    
    # Function to build the rank-32 fallback
    def build_rank32():
        terms = []
        # Strassen blocks for k-pairs: (0,1), (2,3), (4,5), (6,7)
        for block_idx in range(4):
            k0 = 2 * block_idx
            k1 = k0 + 1
            
            U1 = {0: 1, 3: 1}
            V1 = {k0: 1, 9+k1: 1}
            W1 = {2*k0: 1, 2*k1+1: 1}
            terms.append((U1, V1, W1))
            
            U2 = {2: 1, 3: 1}
            V2 = {k0: 1}
            W2 = {2*k0+1: 1, 2*k1+1: -1}
            terms.append((U2, V2, W2))
            
            U3 = {0: 1}
            V3 = {k1: 1, 9+k1: -1}
            W3 = {2*k1: 1, 2*k1+1: 1}
            terms.append((U3, V3, W3))
            
            U4 = {3: 1}
            V4 = {9+k0: 1, k0: -1}
            W4 = {2*k0: 1, 2*k0+1: 1}
            terms.append((U4, V4, W4))
            
            U5 = {0: 1, 1: 1}
            V5 = {9+k1: 1}
            W5 = {2*k0: -1, 2*k1: 1}
            terms.append((U5, V5, W5))
            
            U6 = {2: 1, 0: -1}
            V6 = {k0: 1, k1: 1}
            W6 = {2*k1+1: 1}
            terms.append((U6, V6, W6))
            
            U7 = {1: 1, 3: -1}
            V7 = {9+k0: 1, 9+k1: 1}
            W7 = {2*k0: 1}
            terms.append((U7, V7, W7))
        
        # Schoolbook for k=8
        k = 8
        terms.append(({0: 1}, {k: 1}, {2*k: 1}))
        terms.append(({1: 1}, {9+k: 1}, {2*k: 1}))
        terms.append(({2: 1}, {k: 1}, {2*k+1: 1}))
        terms.append(({3: 1}, {9+k: 1}, {2*k+1: 1}))
        
        rank = len(terms)
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(18)]
        W = [[0] * rank for _ in range(18)]
        
        for t, (Ud, Vd, Wd) in enumerate(terms):
            for idx, val in Ud.items():
                U[idx][t] = val
            for idx, val in Vd.items():
                V[idx][t] = val
            for idx, val in Wd.items():
                W[idx][t] = val
        
        return U, V, W

    # Function to try to find rank-8 for 2x2x3
    def try_find_rank8_223():
        # Target: T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,1,2}
        # U: 4x8, V: 6x8, W: 6x8
        # Flatten variables
        # We fix some variables to break symmetry and help convergence
        # e.g., set U[0,0,0] = 1
        
        def func(params):
            U = params[0:32].reshape(4, 8)
            V = params[32:80].reshape(6, 8)
            W = params[80:128].reshape(6, 8)
            
            # Reshape for einsum
            # U: i,j,r -> 2,2,8
            # V: j,k,r -> 2,3,8
            # W: k,i,r -> 3,2,8
            T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
            # Target is all ones
            diff = T - 1.0
            return diff.flatten()
        
        # Initial guess: random
        np.random.seed(42)
        x0 = np.random.randn(128) * 0.1
        # Set U[0,0,0] = 1 to fix scale
        x0[0] = 1.0
        
        try:
            result = least_squares(func, x0, method='lm', max_nfev=10000)
            if result.success and np.allclose(result.fun, 0, atol=1e-6):
                U = result.x[0:32].reshape(4, 8)
                V = result.x[32:80].reshape(6, 8)
                W = result.x[80:128].reshape(6, 8)
                # Verify
                T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
                if np.allclose(T, 1.0, atol=1e-5):
                    return U, V, W
        except Exception:
            pass
        return None

    # Try to build rank-24 solution
    U_223, V_223, W_223 = try_find_rank8_223()
    
    if U_223 is not None:
        # Build full rank-24 decomposition
        rank = 24
        U = np.zeros((4, rank))
        V = np.zeros((18, rank))
        W = np.zeros((18, rank))
        
        # Block 1: k=0,1,2
        U[:, 0:8] = U_223
        V[0:6, 0:8] = V_223
        W[0:6, 0:8] = W_223
        
        # Block 2: k=3,4,5
        U[:, 8:16] = U_223
        V[6:12, 8:16] = V_223
        W[6:12, 8:16] = W_223
        
        # Block 3: k=6,7,8
        U[:, 16:24] = U_223
        V[12:18, 16:24] = V_223
        W[12:18, 16:24] = W_223
        
        # Verify full decomposition
        T = np.einsum('ijr,jkr,kir->ijk', U.reshape(2,2,-1), V.reshape(2,9,-1), W.reshape(9,2,-1))
        if np.allclose(T, 1.0, atol=1e-5):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Fallback if verification fails (should not happen if local check passed)
            pass

    # Fallback to rank 32
    U, V, W = build_rank32()
    return {"U": U, "V": V, "W": W}
