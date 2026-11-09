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

def _softmax(z):
    z2 = z - np.max(z)
    e = np.exp(z2)
    s = e.sum()
    if s < 1e-300 or not np.isfinite(s):
        return np.full_like(z, 1.0 / len(z))
    return e / s

def _entropy(q):
    m = q > 1e-300
    if not np.any(m):
        return 0.0
    return -np.sum(q[m] * np.log2(q[m]))

def _entropy_grad_P(q):
    # Returns gradient of entropy w.r.t probabilities q
    # H = -sum(q * log2(q))
    # dH/dq = -log2(q) - 1/ln(2)
    g = np.zeros_like(q)
    m = q > 1e-300
    g[m] = -(np.log2(q[m]) + 1.0 / np.log(2.0))
    return g

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

def _make_f_and_grad(idx_a, idx_b, idx_c, idx_d, beta):
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    lc = idx_c.max() + 1
    ld = idx_d.max() + 1

    def fun_and_grad(z):
        P = _softmax(z)
        qa = np.bincount(idx_a, weights=P, minlength=la)
        qb = np.bincount(idx_b, weights=P, minlength=lb)
        qc = np.bincount(idx_c, weights=P, minlength=lc)
        qd = np.bincount(idx_d, weights=P, minlength=ld)
        
        # Entropies
        Ha = _entropy(qa)
        Hb = _entropy(qb)
        Hc = _entropy(qc)
        Hd = _entropy(qd)
        
        # Smooth Max
        Hs = np.array([Ha, Hb, Hd])
        max_H = np.max(Hs)
        exps = np.exp(beta * (Hs - max_H))
        sum_exps = np.sum(exps)
        M_smooth = (max_H + np.log(sum_exps)) / beta
        
        if M_smooth < 1e-12:
            return -1e9, np.zeros_like(z)
            
        value = Hc / M_smooth
        
        # Gradients w.r.t P
        ga = _entropy_grad_P(qa)
        gb = _entropy_grad_P(qb)
        gc = _entropy_grad_P(qc)
        gd = _entropy_grad_P(qd)
        
        # Map gradients from q-space to P-space
        # grad_P H_k = P * (g_k[idx] - dot(g_k[idx], P))
        # Note: g_k is defined on the bins. We need to expand it to P size.
        # g_k_expanded[i] = g_k[idx_k[i]]
        
        ga_exp = ga[idx_a]
        gb_exp = gb[idx_b]
        gc_exp = gc[idx_c]
        gd_exp = gd[idx_d]
        
        grad_Ha = P * (ga_exp - np.dot(ga_exp, P))
        grad_Hb = P * (gb_exp - np.dot(gb_exp, P))
        grad_Hc = P * (gc_exp - np.dot(gc_exp, P))
        grad_Hd = P * (gd_exp - np.dot(gd_exp, P))
        
        # Gradient of M_smooth w.r.t Hs
        # dM/dH_j = exp(beta * (H_j - max_H)) / sum_exps
        w_a = exps[0] / sum_exps
        w_b = exps[1] / sum_exps
        w_d = exps[2] / sum_exps
        
        grad_M = w_a * grad_Ha + w_b * grad_Hb + w_d * grad_Hd
        
        # Gradient of value = Hc / M_smooth
        # d(value)/dz = (1/M) * grad_Hc - (Hc/M^2) * grad_M
        
        grad_val = (1.0 / M_smooth) * grad_Hc - (Hc / (M_smooth * M_smooth)) * grad_M
        
        return -value, -grad_val

    return fun_and_grad

def _finalize(P, idx_a, idx_b, idx_c, idx_d):
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s <= 1e-9 or not np.isfinite(s):
        return np.full((10, 10), 1.0 / 100.0)
    P = P / s
    # recompute M to ensure validity
    pf = P.ravel()
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    ld = idx_d.max() + 1
    qa = np.bincount(idx_a, weights=pf, minlength=la)
    qb = np.bincount(idx_b, weights=pf, minlength=lb)
    qd = np.bincount(idx_d, weights=pf, minlength=ld)
    M = max(_entropy(qa), _entropy(qb), _entropy(qd))
    if M <= 1e-9:
        return np.full((10, 10), 1.0 / 100.0)
    return P.astype(np.float64)

def _local_refine(z, idx_a, idx_b, idx_c, idx_d, deadline, best_val, best_z):
    """
    Refines a single starting point z using a sequence of L-BFGS-B optimizations
    with increasing beta (sharpness of the soft-max).
    """
    current_z = z.copy()
    # Beta schedule: start smooth, end sharp
    beta_schedule = [1.0, 5.0, 20.0, 100.0, 500.0, 2000.0]
    
    for beta in beta_schedule:
        if time.time() > deadline - 5.0:
            break
            
        fun_and_grad = _make_f_and_grad(idx_a, idx_b, idx_c, idx_d, beta)
        
        try:
            # Increase iterations for higher beta to allow convergence
            max_iter = 100 if beta < 10 else 500
            res = minimize(fun_and_grad, current_z, jac=True, method='L-BFGS-B', 
                           options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': max_iter * 10})
            current_z = res.x
            
            # Evaluate with hard max for true score
            P = _softmax(current_z).reshape(10, 10)
            val = _value_from_P(P, idx_a, idx_b, idx_c, idx_d)
            
            if val > best_val[0]:
                best_val[0] = val
                best_z[0] = current_z.copy()
        except Exception:
            pass
            
    return best_val[0], best_z[0]

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    idx_a, idx_b, idx_c, idx_d = _build_idx()
    
    start_time = time.time()
    # 630s limit, leave 10s margin
    deadline = start_time + 620.0

    best_val = -1e9
    best_z = np.zeros(100)
    
    # 1. Generate diverse initial population
    # We use log-probability parameters z, P = softmax(z)
    pop_size = 80
    pop = []
    
    # Uniform
    pop.append(np.zeros(100))
    
    # Heuristic starts based on geometry
    X, Y = np.meshgrid(np.arange(n), np.arange(n))
    
    # Vertical lines (x fixed)
    for x0 in [0, 4, 9]:
        z = np.full(100, -100.0)
        for y in range(10):
            z[y*10 + x0] = 0.0
        pop.append(z)
        
    # Horizontal lines (y fixed)
    for y0 in [0, 4, 9]:
        z = np.full(100, -100.0)
        for x in range(10):
            z[y0*10 + x] = 0.0
        pop.append(z)
        
    # Main diagonal
    z_diag = np.full(100, -100.0)
    for y in range(10):
        x = y
        z_diag[y*10 + x] = 0.0
    pop.append(z_diag)
    
    # Anti-diagonal
    z_anti = np.full(100, -100.0)
    for y in range(10):
        x = 9 - y
        z_anti[y*10 + x] = 0.0
    pop.append(z_anti)
    
    # Random starts
    rng = np.random.default_rng(42)
    for _ in range(pop_size - len(pop)):
        pop.append(rng.normal(0, 2.0, 100))
        
    # Evaluate initial population to find a good start
    vals = []
    for z in pop:
        P = _softmax(z).reshape(10, 10)
        v = _value_from_P(P, idx_a, idx_b, idx_c, idx_d)
        vals.append(v)
        if v > best_val:
            best_val = v
            best_z = z.copy()
            
    # 2. Genetic Algorithm Loop
    # We will refine the top candidates locally, and also evolve the population.
    # To save time, we don't run full local refinement on every individual every generation.
    # Instead, we do a quick evaluation, keep the best, and do local refinement on the best few.
    
    generation = 0
    temp = 1.0
    cooling = 0.995
    
    # Pre-allocate best_val and best_z as lists to allow modification in helper function if needed, 
    # but here we just manage them in the main loop for clarity and speed.
    
    while time.time() < deadline - 20.0:
        generation += 1
        
        # Identify top candidates for local refinement (elitism + local search)
        # Sort population by value
        sorted_idx = np.argsort(vals)[::-1]
        
        # Refine the best 3 candidates from the previous generation (or initial)
        # This allows us to converge to local optima while the GA explores.
        refine_count = min(3, len(pop))
        for i in range(refine_count):
            idx = sorted_idx[i]
            if time.time() > deadline - 10.0:
                break
            # Run local refinement on this candidate
            val, z_res = _local_refine(pop[idx], idx_a, idx_b, idx_c, idx_d, deadline, [best_val, best_z])
            # Update global best
            if val > best_val:
                best_val = val
                best_z = z_res.copy()
            # Update the population member with the refined result
            pop[idx] = z_res
            vals[idx] = val
            
        # If time is running out, break
        if time.time() > deadline - 15.0:
            break

        # Create new population
        new_pop = []
        new_vals = []
        
        # Elitism: keep the top 10% (or at least 1)
        elite_count = max(1, pop_size // 10)
        for i in range(elite_count):
            idx = sorted_idx[i]
            new_pop.append(pop[idx].copy())
            new_vals.append(vals[idx])
            
        # Fill the rest with offspring
        while len(new_pop) < pop_size:
            # Tournament selection
            i = np.random.choice(len(pop))
            j = np.random.choice(len(pop))
            if vals[i] >= vals[j]:
                parent1 = pop[i]
            else:
                parent1 = pop[j]
                
            k = np.random.choice(len(pop))
            l = np.random.choice(len(pop))
            if vals[k] >= vals[l]:
                parent2 = pop[k]
            else:
                parent2 = pop[l]
                
            # Uniform Crossover
            mask = np.random.rand(100) > 0.5
            child = np.where(mask, parent1, parent2)
            
            # Mutation
            # Gaussian noise
            noise = np.random.normal(0, temp, 100)
            child = child + noise
            
            # Occasionally, do a "restart" mutation: blend with a random start
            if np.random.rand() < 0.1:
                random_z = np.random.normal(0, 2.0, 100)
                child = 0.5 * child + 0.5 * random_z
            
            new_pop.append(child)
            
            # Evaluate
            P = _softmax(child).reshape(10, 10)
            v = _value_from_P(P, idx_a, idx_b, idx_c, idx_d)
            new_vals.append(v)
            
            if v > best_val:
                best_val = v
                best_z = child.copy()
                
        pop = new_pop
        vals = new_vals
        temp *= cooling
        
        # If time is running out, break
        if time.time() > deadline - 15.0:
            break

    # Final polish on the global best
    if time.time() < deadline - 5.0:
        val, z_res = _local_refine(best_z, idx_a, idx_b, idx_c, idx_d, deadline, [best_val, best_z])
        if val > best_val:
            best_val = val
            best_z = z_res.copy()

    P = _softmax(best_z).reshape(n, n)
    P = _finalize(P, idx_a, idx_b, idx_c, idx_d)
    return P
