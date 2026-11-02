import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    nm, mp, pn = n * m, m * p, p * n  # 4, 24, 24

    # Build the target tensor
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(nm):
        row_a = i // m
        col_a = i % m
        for col_b in range(p):
            row_b = col_a
            j = row_b * p + col_b
            row_c = row_a
            col_c = col_b
            k = row_c * p + col_c
            T[i, j, k] = 1.0

    def verify(U, V, W):
        """Check if U,V,W exactly decompose T."""
        r = U.shape[1]
        T_decomp = np.einsum('ic,jc,kc->ijk', U, V, W)
        max_err = np.max(np.abs(T - T_decomp))
        return max_err

    # Strategy 1: Strassen rank-42 construction (exact, fast)
    def strassen_rank42():
        rank = 42
        U = np.zeros((nm, rank), dtype=np.float64)
        V = np.zeros((mp, rank), dtype=np.float64)
        W = np.zeros((pn, rank), dtype=np.float64)
        
        num_blocks = p // 2  # 6
        for t in range(num_blocks):
            base = t * 7
            k0 = 2 * t
            k1 = 2 * t + 1
            
            # M1 = (A00 + A11) * (B00 + B11)
            c = base + 0
            U[0, c] += 1; U[3, c] += 1
            V[k0, c] += 1; V[p + k1, c] += 1
            W[k0 * n + 0, c] += 1; W[k1 * n + 1, c] += 1
            
            # M2 = (A10 + A11) * B00
            c = base + 1
            U[2, c] += 1; U[3, c] += 1
            V[k0, c] += 1
            W[k0 * n + 1, c] += 1; W[k1 * n + 1, c] += -1
            
            # M3 = A00 * (B01 - B11)
            c = base + 2
            U[0, c] += 1
            V[k1, c] += 1; V[p + k1, c] += -1
            W[k1 * n + 0, c] += 1; W[k1 * n + 1, c] += 1
            
            # M4 = A11 * (B10 - B00)
            c = base + 3
            U[3, c] += 1
            V[p + k0, c] += 1; V[k0, c] += -1
            W[k0 * n + 0, c] += 1; W[k0 * n + 1, c] += 1
            
            # M5 = (A00 + A01) * B11
            c = base + 4
            U[0, c] += 1; U[1, c] += 1
            V[p + k1, c] += 1
            W[k0 * n + 0, c] += -1; W[k1 * n + 0, c] += 1
            
            # M6 = (A10 - A00) * (B00 + B01)
            c = base + 5
            U[2, c] += 1; U[0, c] += -1
            V[k0, c] += 1; V[k1, c] += 1
            W[k1 * n + 1, c] += 1
            
            # M7 = (A01 - A11) * (B10 + B11)
            c = base + 6
            U[1, c] += 1; U[3, c] += -1
            V[p + k0, c] += 1; V[p + k1, c] += 1
            W[k0 * n + 0, c] += 1
        
        return U, V, W, rank

    # Strategy 2: Schoolbook rank-48 (guaranteed)
    def schoolbook_rank48():
        rank = 48
        U = np.zeros((nm, rank), dtype=np.float64)
        V = np.zeros((mp, rank), dtype=np.float64)
        W = np.zeros((pn, rank), dtype=np.float64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
        
        return U, V, W, rank

    # Strategy 3: Try to find lower rank via optimization
    def try_optimize(rank, max_iter=100):
        """Try to find rank-r decomposition via L-BFGS-B."""
        if time.time() > deadline - 10:
            return None
        
        nm, mp, pn = n * m, m * p, p * n
        total_params = nm * rank + mp * rank + pn * rank
        
        def objective(params):
            U = params[0:nm*rank].reshape(nm, rank)
            V = params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
            W = params[nm*rank+mp*rank:].reshape(pn, rank)
            T_decomp = np.einsum('ic,jc,kc->ijk', U, V, W)
            err = T - T_decomp
            return np.sum(err * err)
        
        def gradient(params):
            U = params[0:nm*rank].reshape(nm, rank)
            V = params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
            W = params[nm*rank+mp*rank:].reshape(pn, rank)
            T_decomp = np.einsum('ic,jc,kc->ijk', U, V, W)
            diff = T_decomp - T
            
            tmp1 = np.einsum('ijk,kr->ijr', diff, W)
            dU = np.einsum('ijr,jr->ir', tmp1, V)
            
            tmp2 = np.einsum('ijk,ir->jkr', diff, U)
            dV = np.einsum('jkr,kr->jr', tmp2, W)
            
            tmp3 = np.einsum('ijk,jr->ikr', diff, V)
            dW = np.einsum('ikr,ir->kr', tmp3, U)
            
            return 2.0 * np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        
        best_params = None
        best_fun = np.inf
        
        np.random.seed(42)
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            
            params0 = np.random.randn(total_params) * 0.5
            from scipy.optimize import minimize
            res = minimize(objective, params0, method='L-BFGS-B',
                          jac=gradient, options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.fun < best_fun:
                best_fun = res.fun
                best_params = res.x
            
            if best_fun < 1e-6:
                break
        
        if best_params is None or best_fun > 1e-4:
            return None
        
        U = best_params[0:nm*rank].reshape(nm, rank)
        V = best_params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
        W = best_params[nm*rank+mp*rank:].reshape(pn, rank)
        
        # Verify
        T_decomp = np.einsum('ic,jc,kc->ijk', U, V, W)
        max_err = np.max(np.abs(T - T_decomp))
        
        if max_err < 1e-6:
            return U, V, W, rank
        return None

    # First, get the guaranteed rank-42 Strassen solution
    U42, V42, W42, rank42 = strassen_rank42()
    err42 = verify(U42, V42, W42)
    
    if err42 > 1e-10:
        # Something wrong with Strassen, fall back to schoolbook
        U, V, W, rank = schoolbook_rank48()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W, best_rank = U42, V42, W42, rank42
    
    # Try to improve below rank 42
    # Lower bound: at least 7 (from embedded 2x2x2)
    # Try ranks from 30 to 41
    if time.time() < deadline - 30:
        for r in range(30, 42):
            if time.time() > deadline - 15:
                break
            
            result = try_optimize(r, max_iter=80)
            if result is not None:
                U, V, W, rank = result
                if rank < best_rank:
                    best_U, best_V, best_W, best_rank = U, V, W, rank
                    print(f"Found rank {rank} decomposition")
                    break  # Found improvement, stop searching lower
    
    # If we didn't find better than 42, return the Strassen solution
    if best_rank <= 42:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback: schoolbook rank 48
    U, V, W, rank = schoolbook_rank48()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
