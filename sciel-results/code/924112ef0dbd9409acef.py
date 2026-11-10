# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _build_idx():
    X, Y = np.meshgrid(np.arange(10), np.arange(10))  # X[y,x]=x, Y[y,x]=y
    idx_a = X.astype(np.int64).ravel()                 # x
    idx_b = (X + Y).astype(np.int64).ravel()           # x+y
    idx_c = (X - Y + 9).astype(np.int64).ravel()       # x-y  (shifted >=0)
    idx_d = (X + 2 * Y).astype(np.int64).ravel()       # x+2y
    return idx_a, idx_b, idx_c, idx_d

def _entropy(q):
    m = q > 1e-300
    if not np.any(m):
        return 0.0
    return -np.sum(q[m] * np.log2(q[m]))

def _value_from_P(P, idx_a, idx_b, idx_c, idx_d):
    pf = P.ravel()
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    lc = idx_c.max() + 1
    ld = idx_d.max() + 1
    
    qa = np.bincount(idx_a, weights=pf, minlength=la)
    qb = np.bincount(idx_b, weights=pf, minlength=lb)
    qc = np.bincount(idx_c, weights=pf, minlength=lc)
    qd = np.bincount(idx_d, weights=pf, minlength=ld)
    
    D = _entropy(qc)
    M = max(_entropy(qa), _entropy(qb), _entropy(qd))
    
    if M < 1e-12:
        return -1e9
    return D / M

def _refine_weights_fixed_support(support, idx_a, idx_b, idx_c, idx_d, deadline):
    """
    Refines weights on a fixed set of support indices.
    """
    if len(support) == 0:
        return None, -1e9
        
    w = np.ones(len(support)) / len(support)
    
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    lc = idx_c.max() + 1
    ld = idx_d.max() + 1
    
    sup_a = idx_a[support]
    sup_b = idx_b[support]
    sup_c = idx_c[support]
    sup_d = idx_d[support]
    
    def get_q(idx_map, w, length):
        q = np.zeros(length)
        np.add.at(q, idx_map, w)
        return q
        
    def fun_and_grad(w):
        qa = get_q(sup_a, w, la)
        qb = get_q(sup_b, w, lb)
        qc = get_q(sup_c, w, lc)
        qd = get_q(sup_d, w, ld)
        
        Ha = _entropy(qa)
        Hb = _entropy(qb)
        Hc = _entropy(qc)
        Hd = _entropy(qd)
        
        M = max(Ha, Hb, Hd)
        if M < 1e-12:
            return 1e9, np.zeros_like(w)
            
        val = Hc / M
        # Numerical gradient is expensive, but support size is small.
        # However, L-BFGS-B with numerical jac on small dims is okay.
        # Or we can just return -val and let scipy handle jac.
        return -val, None

    # Simplex constraint: sum(w)=1, w>=0
    # L-BFGS-B can handle bounds.
    # We need to ensure sum(w)=1. 
    # Standard trick: optimize w[0..k-2] and set w[k-1] = 1 - sum(w[0..k-2]).
    # But L-BFGS-B doesn't handle equality constraints directly.
    # Alternative: use softmax parameterization for weights.
    # z = log(w). w = softmax(z).
    
    z_init = np.zeros(len(support))
    
    def fun_z(z):
        if time.time() > deadline - 5:
            raise TimeoutError()
        z_norm = z - np.max(z)
        w = np.exp(z_norm)
        s = w.sum()
        if s < 1e-300:
            return 1e9
        w = w / s
        
        qa = get_q(sup_a, w, la)
        qb = get_q(sup_b, w, lb)
        qc = get_q(sup_c, w, lc)
        qd = get_q(sup_d, w, ld)
        
        Ha = _entropy(qa)
        Hb = _entropy(qb)
        Hc = _entropy(qc)
        Hd = _entropy(qd)
        
        M = max(Ha, Hb, Hd)
        if M < 1e-12:
            return 1e9
        return -(Hc / M)

    try:
        res = minimize(fun_z, z_init, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10})
        z_opt = res.x
        z_norm = z_opt - np.max(z_opt)
        w_opt = np.exp(z_norm)
        w_opt = w_opt / w_opt.sum()
        
        P = np.zeros(100)
        P[support] = w_opt
        val = -res.fun
        return P, val
    except Exception:
        return None, -1e9

def _greedy_support_search(idx_a, idx_b, idx_c, idx_d, deadline, target_size=20):
    """
    Greedy construction of support set.
    Start with empty set, add cell that maximizes gain in D / M.
    """
    support = []
    best_val = -1e9
    
    # Precompute indices for all cells
    all_indices = np.arange(100)
    
    for step in range(target_size):
        if time.time() > deadline - 10:
            break
            
        best_candidate = -1
        best_gain = -1e9
        
        # Try adding each non-supported cell
        current_support_arr = np.array(support) if support else np.array([], dtype=int)
        
        # For speed, we can evaluate candidates in batches or just iterate if target_size is small
        # 100 candidates * 20 steps = 2000 evaluations. Each evaluation is O(100).
        # Total ~200k ops. Very fast.
        
        for cell in all_indices:
            if cell in support:
                continue
                
            new_support = list(support) + [cell]
            P = np.zeros(100)
            P[new_support] = 1.0
            P = P / P.sum()
            
            val = _value_from_P(P, idx_a, idx_b, idx_c, idx_d)
            
            if val > best_gain:
                best_gain = val
                best_candidate = cell
                
        if best_candidate != -1:
            support.append(best_candidate)
            best_val = max(best_val, best_gain)
        else:
            break
            
    return np.array(support), best_val

def _generate_initial_candidates(idx_a, idx_b, idx_c, idx_d, deadline):
    """
    Generate a few diverse support sets to start with.
    """
    candidates = []
    
    # 1. Greedy search
    sup_greedy, val_greedy = _greedy_support_search(idx_a, idx_b, idx_c, idx_d, deadline, target_size=25)
    if len(sup_greedy) > 0:
        candidates.append(sup_greedy)
        
    # 2. Cross structure
    sup_cross = []
    for i in range(10):
        sup_cross.append(i * 10) # y=0, x=i
        sup_cross.append(i)      # x=0, y=i
    sup_cross = list(set(sup_cross))
    candidates.append(np.array(sup_cross))
    
    # 3. Random sparse
    rng = np.random.default_rng(42)
    for _ in range(5):
        sup_rand = rng.choice(100, 20, replace=False)
        candidates.append(sup_rand)
        
    # 4. "Diagonal-ish" structures
    # x-y = k
    for k in [-5, -2, 0, 2, 5]:
        sup_diag = []
        for x in range(10):
            y = x - k
            if 0 <= y < 10:
                sup_diag.append(y * 10 + x)
        if len(sup_diag) > 0:
             candidates.append(np.array(sup_diag))
             
    return candidates

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    idx_a, idx_b, idx_c, idx_d = _build_idx()
    
    start_time = time.time()
    deadline = start_time + 620.0
    
    best_val = -1e9
    best_P = np.full((n, n), 1.0 / 100.0)
    
    # 1. Generate candidate supports
    candidates = _generate_initial_candidates(idx_a, idx_b, idx_c, idx_d, deadline)
    
    # 2. Refine weights for each candidate support
    for sup in candidates:
        if time.time() > deadline - 30:
            break
            
        if len(sup) == 0:
            continue
            
        P_opt, val_opt = _refine_weights_fixed_support(sup, idx_a, idx_b, idx_c, idx_d, deadline)
        
        if P_opt is not None and val_opt > best_val:
            best_val = val_opt
            best_P = P_opt.reshape(n, n)
            
    # 3. If greedy search didn't find something great, try a broader random search with refinement
    # This acts as a safety net
    if best_val < 1.5:
        rng = np.random.default_rng(123)
        for i in range(20):
            if time.time() > deadline - 30:
                break
            sup_rand = rng.choice(100, 15, replace=False)
            P_opt, val_opt = _refine_weights_fixed_support(sup_rand, idx_a, idx_b, idx_c, idx_d, deadline)
            if P_opt is not None and val_opt > best_val:
                best_val = val_opt
                best_P = P_opt.reshape(n, n)
                
    # Final Polish on the best P found
    if time.time() < deadline - 10 and best_val > -1e8:
        # Convert best_P to z
        z_best = np.log(best_P.ravel() + 1e-12)
        
        def fun_final(z):
            if time.time() > deadline - 5:
                raise TimeoutError()
            P = np.exp(z - np.max(z))
            s = P.sum()
            if s < 1e-300:
                return 1e9
            P = P / s
            v = _value_from_P(P, idx_a, idx_b, idx_c, idx_d)
            return -v

        try:
            res = minimize(fun_final, z_best, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
            z_opt = res.x
            P_opt = np.exp(z_opt - np.max(z_opt))
            P_opt = P_opt / P_opt.sum()
            P_opt = P_opt.reshape(n, n)
            v_opt = _value_from_P(P_opt, idx_a, idx_b, idx_c, idx_d)
            if v_opt > best_val:
                best_val = v_opt
                best_P = P_opt
        except Exception:
            pass

    # Finalize
    best_P = np.clip(best_P, 0.0, None)
    s = best_P.sum()
    if s <= 1e-9:
        return np.full((n, n), 1.0 / 100.0)
    best_P = best_P / s
    
    # Check validity
    pf = best_P.ravel()
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    ld = idx_d.max() + 1
    qa = np.bincount(idx_a, weights=pf, minlength=la)
    qb = np.bincount(idx_b, weights=pf, minlength=lb)
    qd = np.bincount(idx_d, weights=pf, minlength=ld)
    M = max(_entropy(qa), _entropy(qb), _entropy(qd))
    if M <= 1e-9:
        return np.full((n, n), 1.0 / 100.0)
        
    return best_P.astype(np.float64)

# EVOLVE-BLOCK-END
