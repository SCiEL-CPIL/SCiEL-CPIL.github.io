import numpy as np
import time
from itertools import product

def construct():
    n, m, p = 2, 3, 3
    nm, mp, pn = n * m, m * p, p * n  # 6, 9, 6
    deadline = time.time() + 145
    
    # Precompute tensor entries
    T = np.zeros((nm, mp, pn))
    entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
                entries.append((a, b, c))
    
    a_idx = np.array([e[0] for e in entries])
    b_idx = np.array([e[1] for e in entries])
    c_idx = np.array([e[2] for e in entries])
    num_entries = len(entries)
    
    def verify(U, V, W, tol=1e-9):
        """Check half-integer constraint and exact decomposition."""
        if not np.allclose(U * 2, np.round(U * 2), atol=tol):
            return False
        if not np.allclose(V * 2, np.round(V * 2), atol=tol):
            return False
        if not np.allclose(W * 2, np.round(W * 2), atol=tol):
            return False
        R = np.einsum('as,bs,cs->abc', U, V, W)
        for (a, b, c) in entries:
            if not np.isclose(R[a, b, c], 1.0, atol=tol):
                return False
        return True
    
    def compute_cost(U, V, W):
        R_nz = np.zeros(num_entries)
        for s in range(U.shape[1]):
            R_nz += U[a_idx, s] * V[b_idx, s] * W[c_idx, s]
        return np.sum((R_nz - 1.0) ** 2)
    
    def als_decompose(U_init, V_init, W_init, rank, max_iters=500, tol=1e-14):
        """ALS with tight convergence."""
        U = U_init.copy().astype(np.float64)
        V = V_init.copy().astype(np.float64)
        W = W_init.copy().astype(np.float64)
        
        prev_cost = np.inf
        best_cost = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            # Update U
            for a in range(nm):
                mask = (a_idx == a)
                if not np.any(mask):
                    continue
                b_sub = b_idx[mask]
                c_sub = c_idx[mask]
                D = V[b_sub, :] * W[c_sub, :]
                try:
                    U[a, :] = np.linalg.lstsq(D, np.ones(len(b_sub)), rcond=None)[0]
                except:
                    pass
            
            # Update V
            for b in range(mp):
                mask = (b_idx == b)
                if not np.any(mask):
                    continue
                a_sub = a_idx[mask]
                c_sub = c_idx[mask]
                D = U[a_sub, :] * W[c_sub, :]
                try:
                    V[b, :] = np.linalg.lstsq(D, np.ones(len(a_sub)), rcond=None)[0]
                except:
                    pass
            
            # Update W
            for c in range(pn):
                mask = (c_idx == c)
                if not np.any(mask):
                    continue
                a_sub = a_idx[mask]
                b_sub = b_idx[mask]
                D = U[a_sub, :] * V[b_sub, :]
                try:
                    W[c, :] = np.linalg.lstsq(D, np.ones(len(a_sub)), rcond=None)[0]
                except:
                    pass
            
            cost = compute_cost(U, V, W)
            if cost < best_cost:
                best_cost = cost
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if abs(prev_cost - cost) < tol:
                break
            prev_cost = cost
        
        return best_U, best_V, best_W, best_cost
    
    def polish(U, V, W, rank, max_iters=100, tol=1e-15):
        """High-precision polish after ALS."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for it in range(max_iters):
            U_old = U.copy()
            V_old = V.copy()
            W_old = W.copy()
            
            for a in range(nm):
                mask = (a_idx == a)
                if not np.any(mask):
                    continue
                b_sub = b_idx[mask]
                c_sub = c_idx[mask]
                D = V[b_sub, :] * W[c_sub, :]
                try:
                    U[a, :] = np.linalg.lstsq(D, np.ones(len(b_sub)), rcond=None)[0]
                except:
                    pass
            
            for b in range(mp):
                mask = (b_idx == b)
                if not np.any(mask):
                    continue
                a_sub = a_idx[mask]
                c_sub = c_idx[mask]
                D = U[a_sub, :] * W[c_sub, :]
                try:
                    V[b, :] = np.linalg.lstsq(D, np.ones(len(a_sub)), rcond=None)[0]
                except:
                    pass
            
            for c in range(pn):
                mask = (c_idx == c)
                if not np.any(mask):
                    continue
                a_sub = a_idx[mask]
                b_sub = b_idx[mask]
                D = U[a_sub, :] * V[b_sub, :]
                try:
                    W[c, :] = np.linalg.lstsq(D, np.ones(len(a_sub)), rcond=None)[0]
                except:
                    pass
            
            if np.allclose(U, U_old, atol=tol) and np.allclose(V, V_old, atol=tol) and np.allclose(W, W_old, atol=tol):
                break
        
        return U, V, W
    
    def snap_and_verify(U, V, W):
        """Try to snap to half-integers and verify with multiple strategies."""
        # Try all combinations of rounding modes for each matrix
        rounding_modes = [
            lambda x: np.round(x * 2) / 2,
            lambda x: np.floor(x * 2 + 0.5) / 2,
            lambda x: np.round(x),
            lambda x: np.ceil(x * 2) / 2,
        ]
        
        for mode_u in rounding_modes:
            for mode_v in rounding_modes:
                for mode_w in rounding_modes:
                    Uc = mode_u(U)
                    Vc = mode_v(V)
                    Wc = mode_w(W)
                    if verify(Uc, Vc, Wc):
                        return Uc, Vc, Wc
        
        # Try small perturbations around the rounded values
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Try adding small offsets
        offsets = [0, 0.25, -0.25, 0.5, -0.5]
        for ou in offsets:
            for ov in offsets:
                for ow in offsets:
                    Uc = U_rounded + ou
                    Vc = V_rounded + ov
                    Wc = W_rounded + ow
                    if verify(Uc, Vc, Wc):
                        return Uc, Vc, Wc
        
        return None
    
    def local_search(U, V, W, rank, max_steps=5000):
        """Local search over half-integer perturbations."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        # Snap to nearest half-integers first
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        current_cost = compute_cost(U, V, W)
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            if current_cost < 1e-10:
                break
            
            # Try perturbing a random entry by ±0.5
            choice = np.random.choice(3)
            if choice == 0:
                idx = np.random.randint(nm * rank)
                i, j = divmod(idx, rank)
                old_val = U[i, j]
                U[i, j] = old_val + 0.5
                cost = compute_cost(U, V, W)
                if cost >= current_cost:
                    U[i, j] = old_val
                else:
                    current_cost = cost
            elif choice == 1:
                idx = np.random.randint(mp * rank)
                i, j = divmod(idx, rank)
                old_val = V[i, j]
                V[i, j] = old_val + 0.5
                cost = compute_cost(U, V, W)
                if cost >= current_cost:
                    V[i, j] = old_val
                else:
                    current_cost = cost
            else:
                idx = np.random.randint(pn * rank)
                i, j = divmod(idx, rank)
                old_val = W[i, j]
                W[i, j] = old_val + 0.5
                cost = compute_cost(U, V, W)
                if cost >= current_cost:
                    W[i, j] = old_val
                else:
                    current_cost = cost
        
        if verify(U, V, W):
            return U, V, W
        return None
    
    def structured_init(rank, seed, strategy=0):
        """Create structured initializations with multiple strategies."""
        np.random.seed(seed)
        
        if strategy == 0:
            # Random
            U = np.random.randn(nm, rank) * 0.3
            V = np.random.randn(mp, rank) * 0.3
            W = np.random.randn(pn, rank) * 0.3
        elif strategy == 1:
            # Sparse random with half-integers
            vals = [-1, -0.5, 0.5, 1]
            U = np.random.choice(vals, size=(nm, rank))
            V = np.random.choice(vals, size=(mp, rank))
            W = np.random.choice(vals, size=(pn, rank))
        elif strategy == 2:
            # Binary initialization
            U = np.random.randint(0, 2, size=(nm, rank)).astype(float)
            V = np.random.randint(0, 2, size=(mp, rank)).astype(float)
            W = np.random.randint(0, 2, size=(pn, rank)).astype(float)
        elif strategy == 3:
            # Mixed: some entries zero, others random
            U = np.random.randn(nm, rank) * 0.5
            U[np.random.rand(nm, rank) < 0.5] = 0
            V = np.random.randn(mp, rank) * 0.5
            V[np.random.rand(mp, rank) < 0.5] = 0
            W = np.random.randn(pn, rank) * 0.5
            W[np.random.rand(pn, rank) < 0.5] = 0
        else:
            U = np.random.randn(nm, rank) * 0.3
            V = np.random.randn(mp, rank) * 0.3
            W = np.random.randn(pn, rank) * 0.3
        
        return U, V, W
    
    def try_rank(rank, num_restarts=10, time_budget=25):
        start_time = time.time()
        best_solution = None
        best_cost = np.inf
        
        for restart in range(num_restarts):
            if time.time() - start_time > time_budget:
                break
            if time.time() > deadline - 10:
                break
            
            seed = 10000 + rank * 1000 + restart * 73
            strategy = restart % 4
            U_init, V_init, W_init = structured_init(rank, seed, strategy)
            
            # ALS
            U, V, W, cost = als_decompose(U_init, V_init, W_init, rank, max_iters=300)
            
            if cost < best_cost:
                best_cost = cost
            
            # Polish
            if cost < 1e-6:
                U, V, W = polish(U, V, W, rank, max_iters=200)
                cost = compute_cost(U, V, W)
                if cost < best_cost:
                    best_cost = cost
            
            # Try to snap
            if cost < 1e-4:
                result = snap_and_verify(U, V, W)
                if result is not None:
                    return result
            
            # Local search
            if cost < 1e-3 and time.time() < deadline - 15:
                result = local_search(U, V, W, rank, max_steps=2000)
                if result is not None:
                    return result
        
        return None
    
    # Try ranks from low to high
    # Theoretical lower bound is around 11-13 for 2x3x3
    for target_rank in [13, 14, 15, 16, 17]:
        if time.time() > deadline - 20:
            break
        
        result = try_rank(target_rank, num_restarts=12, time_budget=22)
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback: schoolbook rank 18
    rank = n * m * p  # 18
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    for col, (a, b, c) in enumerate(entries):
        U[a, col] = 1
        V[b, col] = 1
        W[c, col] = 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
