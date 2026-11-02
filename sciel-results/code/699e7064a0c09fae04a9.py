"""Find a rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication."""
import time
import numpy as np
from itertools import product
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    dim_u = n * m  # 10
    dim_v = m * p  # 25
    dim_w = p * n  # 10
    
    # Build the target tensor
    target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0
    
    def get_schoolbook():
        """Return schoolbook decomposition as dict of lists."""
        rank = n * m * p  # 50
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify_decomposition(U, V, W, atol=1e-9):
        """Verify that U, V, W exactly decompose the target tensor."""
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T, target, atol=atol)
    
    def verify_and_clean(U, V, W, atol=1e-8):
        """Verify and try to clean values to half-integers."""
        if not verify_decomposition(U, V, W, atol):
            return None
        
        # Try snapping to half-integers
        u_snap = np.round(U * 2.0) / 2.0
        v_snap = np.round(V * 2.0) / 2.0
        w_snap = np.round(W * 2.0) / 2.0
        if verify_decomposition(u_snap, v_snap, w_snap, atol=1e-7):
            u_snap = np.where(np.abs(u_snap) < 1e-6, 0, u_snap)
            v_snap = np.where(np.abs(v_snap) < 1e-6, 0, v_snap)
            w_snap = np.where(np.abs(w_snap) < 1e-6, 0, w_snap)
            if verify_decomposition(u_snap, v_snap, w_snap, atol=1e-6):
                return (u_snap, v_snap, w_snap)
        
        # Try integer snap
        u_int = np.round(U)
        v_int = np.round(V)
        w_int = np.round(W)
        if verify_decomposition(u_int, v_int, w_int, atol=1e-6):
            return (u_int, v_int, w_int)
        
        # Return as-is if valid
        return (U, V, W)
    
    def try_rank_with_optimization(rank, max_restarts=3, max_iter=80, time_per_restart=8.0):
        """
        Try to find a decomposition of given rank using gradient-based optimization
        with multiple restarts and structured initialization.
        """
        if time.time() > deadline - 10.0:
            return None
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5.0:
                return None
            
            np.random.seed(10000 + restart * 31 + rank * 7)
            
            # Initialize with a mix of structured and random components
            U = np.zeros((dim_u, rank))
            V = np.zeros((dim_v, rank))
            W = np.zeros((dim_w, rank))
            
            # Strategy 1: Use structured line patterns as base
            # For each rank term, assign a structured pattern
            for r in range(rank):
                # Randomly choose which type of line pattern to use
                pattern_type = np.random.randint(3)
                
                if pattern_type == 0:
                    # Fix (i,j), vary k
                    i = np.random.randint(n)
                    j = np.random.randint(m)
                    U[i * m + j, r] = 1.0
                    for k in range(p):
                        V[j * p + k, r] = 1.0
                        W[k * n + i, r] = 1.0
                elif pattern_type == 1:
                    # Fix (j,k), vary i
                    j = np.random.randint(m)
                    k = np.random.randint(p)
                    V[j * p + k, r] = 1.0
                    for i in range(n):
                        U[i * m + j, r] = 1.0
                        W[k * n + i, r] = 1.0
                else:
                    # Fix (i,k), vary j
                    i = np.random.randint(n)
                    k = np.random.randint(p)
                    W[k * n + i, r] = 1.0
                    for j in range(m):
                        U[i * m + j, r] = 1.0
                        V[j * p + k, r] = 1.0
                
                # Add small random perturbations
                U[:, r] += np.random.randn(dim_u) * 0.1
                V[:, r] += np.random.randn(dim_v) * 0.1
                W[:, r] += np.random.randn(dim_w) * 0.1
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def fun(params):
                u_flat = params[:dim_u * rank]
                v_flat = params[dim_u * rank : dim_u * rank + dim_v * rank]
                w_flat = params[dim_u * rank + dim_v * rank : ]
                
                U_m = u_flat.reshape(dim_u, rank)
                V_m = v_flat.reshape(dim_v, rank)
                W_m = w_flat.reshape(dim_w, rank)
                
                T = np.einsum('ar,br,cr->abc', U_m, V_m, W_m)
                diff = T - target
                loss = np.sum(diff * diff)
                
                grad_u = 2.0 * np.einsum('abc,br,cr->ar', diff, V_m, W_m)
                grad_v = 2.0 * np.einsum('abc,ar,cr->br', diff, U_m, W_m)
                grad_w = 2.0 * np.einsum('abc,ar,br->cr', diff, U_m, V_m)
                
                grad = np.concatenate([grad_u.flatten(), grad_v.flatten(), grad_w.flatten()])
                return loss, grad
            
            try:
                res = minimize(fun, params, method='L-BFGS-B', jac=True,
                               options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 20})
                
                if res.fun < 1e-10:
                    u_final = res.x[:dim_u * rank].reshape(dim_u, rank)
                    v_final = res.x[dim_u * rank : dim_u * rank + dim_v * rank].reshape(dim_v, rank)
                    w_final = res.x[dim_u * rank + dim_v * rank : ].reshape(dim_w, rank)
                    
                    result = verify_and_clean(u_final, v_final, w_final)
                    if result is not None:
                        return result
                        
            except Exception:
                continue
        
        return None
    
    def try_pattern_subset(rank, max_time=10.0):
        """
        Try to find a subset of structured line patterns that exactly decomposes the tensor.
        """
        if time.time() > deadline - 5.0:
            return None
        
        # Generate all possible line patterns
        all_patterns = []
        
        # Type 1: Fix (i,j), vary k
        for i in range(n):
            for j in range(m):
                u_sup = frozenset([i * m + j])
                v_sup = frozenset([j * p + k for k in range(p)])
                w_sup = frozenset([k * n + i for k in range(p)])
                all_patterns.append((u_sup, v_sup, w_sup))
        
        # Type 2: Fix (j,k), vary i
        for j in range(m):
            for k in range(p):
                u_sup = frozenset([i * m + j for i in range(n)])
                v_sup = frozenset([j * p + k])
                w_sup = frozenset([k * n + i for i in range(n)])
                all_patterns.append((u_sup, v_sup, w_sup))
        
        # Type 3: Fix (i,k), vary j
        for i in range(n):
            for k in range(p):
                u_sup = frozenset([i * m + j for j in range(m)])
                v_sup = frozenset([j * p + k for j in range(m)])
                w_sup = frozenset([k * n + i])
                all_patterns.append((u_sup, v_sup, w_sup))
        
        # Try random subsets of size 'rank'
        np.random.seed(42)
        for attempt in range(50):
            if time.time() > deadline - 5.0:
                return None
            
            indices = np.random.choice(len(all_patterns), size=min(rank, len(all_patterns)), replace=False)
            selected = [all_patterns[i] for i in indices]
            
            # Build linear system
            all_positions = set()
            for r in range(len(selected)):
                u_sup, v_sup, w_sup = selected[r]
                for a in u_sup:
                    for b in v_sup:
                        for c in w_sup:
                            all_positions.add((a, b, c))
            
            pos_list = list(all_positions)
            if not pos_list:
                continue
            
            A = np.zeros((len(pos_list), len(selected)))
            b_vec = np.zeros(len(pos_list))
            
            for pos_idx, (a, b, c) in enumerate(pos_list):
                b_vec[pos_idx] = target[a, b, c]
                for r in range(len(selected)):
                    u_sup, v_sup, w_sup = selected[r]
                    if a in u_sup and b in v_sup and c in w_sup:
                        A[pos_idx, r] = 1.0
            
            try:
                coeffs, residuals, rank_A, sv = np.linalg.lstsq(A, b_vec, rcond=None)
                
                # Check exactness
                T = np.zeros((dim_u, dim_v, dim_w))
                for r in range(len(selected)):
                    if abs(coeffs[r]) > 1e-10:
                        u_sup, v_sup, w_sup = selected[r]
                        for a in u_sup:
                            for b in v_sup:
                                for c in w_sup:
                                    T[a, b, c] += coeffs[r]
                
                if np.allclose(T, target, atol=1e-8):
                    # Build U, V, W
                    U = np.zeros((dim_u, len(selected)))
                    V = np.zeros((dim_v, len(selected)))
                    W = np.zeros((dim_w, len(selected)))
                    
                    for r in range(len(selected)):
                        u_sup, v_sup, w_sup = selected[r]
                        for a in u_sup:
                            U[a, r] = coeffs[r]
                        for b in v_sup:
                            V[b, r] = 1.0
                        for c in w_sup:
                            W[c, r] = 1.0
                    
                    result = verify_and_clean(U, V, W)
                    if result is not None:
                        return result
                        
            except Exception:
                continue
        
        return None
    
    # Try various ranks from low to high
    best_result = None
    
    # Try pattern-based approach first for various ranks
    for target_rank in [35, 38, 40, 42, 44, 45, 46, 48]:
        if time.time() > deadline - 15.0:
            break
        
        result = try_pattern_subset(target_rank)
        if result is not None:
            U, V, W = result
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
    
    # If pattern approach failed, try optimization
    if best_result is None:
        for target_rank in [35, 38, 40, 42, 44, 45, 46, 48]:
            if time.time() > deadline - 10.0:
                break
            
            result = try_rank_with_optimization(target_rank, max_restarts=2, max_iter=60)
            if result is not None:
                U, V, W = result
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
    
    if best_result is not None:
        return best_result
    else:
        return get_schoolbook()
