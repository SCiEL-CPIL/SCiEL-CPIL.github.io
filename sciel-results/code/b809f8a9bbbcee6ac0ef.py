import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    def verify(U, V, W, r):
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        try:
            T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
            return np.allclose(T, 1.0, atol=1e-9)
        except:
            return False

    def to_half_int(x):
        return round(x * 2) / 2

    def try_alternating_min(U, V, W, r, max_iters=200):
        """Run alternating minimization and return (U, V, W, success)"""
        for it in range(max_iters):
            if time.time() > deadline - 15:
                return U, V, W, False
            
            U_reshaped = U.reshape(n, m, r)
            V_reshaped = V.reshape(m, p, r)
            W_reshaped = W.reshape(p, n, r)
            
            # Update U
            try:
                M = np.einsum('jkc,kic->ijkc', V_reshaped, W_reshaped, optimize=True)
                M_flat = M.reshape(n*m, p, r)
                U_flat = U.reshape(n*m, r)
                b = np.ones((n*m, p))
                for idx in range(n*m):
                    try:
                        U_flat[idx, :] = np.linalg.lstsq(M_flat[idx], b[idx], rcond=None)[0]
                    except:
                        pass
                U = U_flat.reshape(n*m, r)
            except:
                pass
            
            # Update V
            try:
                U_reshaped = U.reshape(n, m, r)
                M_V = np.einsum('ijc,kic->jkic', U_reshaped, W_reshaped, optimize=True)
                M_V_flat = M_V.reshape(m*p, n, r)
                V_flat = V.reshape(m*p, r)
                b_V = np.ones((m*p, n))
                for idx in range(m*p):
                    try:
                        V_flat[idx, :] = np.linalg.lstsq(M_V_flat[idx], b_V[idx], rcond=None)[0]
                    except:
                        pass
                V = V_flat.reshape(m*p, r)
            except:
                pass
            
            # Update W
            try:
                U_reshaped = U.reshape(n, m, r)
                V_reshaped = V.reshape(m, p, r)
                M_W = np.einsum('ijc,jkc->kijc', U_reshaped, V_reshaped, optimize=True)
                M_W_flat = M_W.reshape(p*n, m, r)
                W_flat = W.reshape(p*n, r)
                b_W = np.ones((p*n, m))
                for idx in range(p*n):
                    try:
                        W_flat[idx, :] = np.linalg.lstsq(M_W_flat[idx], b_W[idx], rcond=None)[0]
                    except:
                        pass
                W = W_flat.reshape(p*n, r)
            except:
                pass
            
            # Check convergence
            if it % 5 == 0:
                if verify(U, V, W, r):
                    return U, V, W, True
        
        return U, V, W, False

    def try_round_and_verify(U, V, W, r):
        """Try rounding to half-integers and verify"""
        U_r = np.array([[to_half_int(U[i*m+j, c]) for c in range(r)] for i in range(n) for j in range(m)])
        V_r = np.array([[to_half_int(V[j*p+k, c]) for c in range(r)] for j in range(m) for k in range(p)])
        W_r = np.array([[to_half_int(W[k*n+i, c]) for c in range(r)] for k in range(p) for i in range(n)])
        if verify(U_r, V_r, W_r, r):
            return U_r, V_r, W_r, True
        return None, None, None, False

    best = None
    best_rank = None
    
    # Try decreasing ranks
    for r in range(59, 49, -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        
        # Strategy 1: Random initializations with various scales
        for restart in range(30):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(restart * 7919 + r * 104729)
            scale = 0.5 + 0.1 * (restart % 5)  # Vary scale from 0.5 to 0.9
            
            U = np.random.randn(n*m, r) * scale
            V = np.random.randn(m*p, r) * scale
            W = np.random.randn(p*n, r) * scale
            
            U, V, W, success = try_alternating_min(U, V, W, r, max_iters=150)
            if success:
                U_r, V_r, W_r, verified = try_round_and_verify(U, V, W, r)
                if verified:
                    if best_rank is None or r < best_rank:
                        best_rank = r
                        best = (U_r.tolist(), V_r.tolist(), W_r.tolist())
                    found = True
                    break
        
        if found:
            continue
        
        # Strategy 2: Structured initializations based on partial schoolbook
        # Use a subset of schoolbook terms as initialization
        if time.time() > deadline - 15:
            break
        
        for struct_init in range(10):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(struct_init * 31 + r)
            
            # Initialize with a random subset of schoolbook pattern
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            # Randomly select some (i,j,k) triples to initialize
            num_terms = min(r, n*m*p)
            for c in range(num_terms):
                i = np.random.randint(n)
                j = np.random.randint(m)
                k = np.random.randint(p)
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
            
            # Add small random perturbation to break symmetry
            U += np.random.randn(n*m, r) * 0.1
            V += np.random.randn(m*p, r) * 0.1
            W += np.random.randn(p*n, r) * 0.1
            
            U, V, W, success = try_alternating_min(U, V, W, r, max_iters=150)
            if success:
                U_r, V_r, W_r, verified = try_round_and_verify(U, V, W, r)
                if verified:
                    if best_rank is None or r < best_rank:
                        best_rank = r
                        best = (U_r.tolist(), V_r.tolist(), W_r.tolist())
                    found = True
                    break
        
        if found:
            continue
        
        # Strategy 3: Binary initialization (entries are -1, 0, or 1)
        if time.time() > deadline - 15:
            break
        
        for binary_init in range(10):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(binary_init * 17 + r * 3)
            
            U = np.random.choice([-1, 0, 1], size=(n*m, r)) * 0.5
            V = np.random.choice([-1, 0, 1], size=(m*p, r)) * 0.5
            W = np.random.choice([-1, 0, 1], size=(p*n, r)) * 0.5
            
            U, V, W, success = try_alternating_min(U, V, W, r, max_iters=150)
            if success:
                U_r, V_r, W_r, verified = try_round_and_verify(U, V, W, r)
                if verified:
                    if best_rank is None or r < best_rank:
                        best_rank = r
                        best = (U_r.tolist(), V_r.tolist(), W_r.tolist())
                    found = True
                    break
    
    if best is not None:
        U_list, V_list, W_list = best
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
