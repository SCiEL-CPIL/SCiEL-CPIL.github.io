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

def _entropy_grad(q):
    m = q > 1e-300
    H = -np.sum(q[m] * np.log2(q[m]))
    dq = np.zeros_like(q)
    dq[m] = -(np.log2(q[m]) + 1.0 / np.log(2.0))
    return H, dq

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

def _make_f_and_grad(idx_a, idx_b, idx_c, idx_d):
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    lc = idx_c.max() + 1
    ld = idx_d.max() + 1
    log2 = np.log(2.0)
    
    def f_and_grad(z, smooth_beta=100.0):
        P = _softmax(z)
        qa = np.bincount(idx_a, weights=P, minlength=la)
        qb = np.bincount(idx_b, weights=P, minlength=lb)
        qc = np.bincount(idx_c, weights=P, minlength=lc)
        qd = np.bincount(idx_d, weights=P, minlength=ld)
        Ha, ga = _entropy_grad(qa)
        Hb, gb = _entropy_grad(qb)
        Hc, gc = _entropy_grad(qc)
        Hd, gd = _entropy_grad(qd)
        D = Hc
        
        # Smooth max using log-sum-exp with temperature
        Hs = np.array([Ha, Hb, Hd])
        if smooth_beta > 1e6:
            # Hard max
            M = np.max(Hs)
            idx_max = np.argmax(Hs)
            gs_list = [ga, gb, gd]
            gM = gs_list[idx_max]
        else:
            # Soft max: M = (1/beta) * log(sum(exp(beta * Hi)))
            beta = smooth_beta
            exps = np.exp(beta * (Hs - np.max(Hs)))
            M = (np.max(Hs) + np.log(exps.sum())) / beta
            # Gradient of soft max wrt each Hi
            weights = exps / exps.sum()
            gM = weights[0] * ga + weights[1] * gb + weights[2] * gd
        
        if M < 1e-12:
            return -1e9, np.zeros_like(z)
        
        # df/dP where f = D / M
        df_dP = gc / M - (D / (M * M)) * gM
        mean = np.dot(df_dP, P)
        grad_z = P * (df_dP - mean)
        return D / M, grad_z

    def fun(z):
        return -f_and_grad(z)[0]

    def fprime(z):
        return -f_and_grad(z)[1]

    return fun, fprime, f_and_grad

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

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    idx_a, idx_b, idx_c, idx_d = _build_idx()
    fun, fprime, f_and_grad = _make_f_and_grad(idx_a, idx_b, idx_c, idx_d)
    
    start_time = time.time()
    deadline = start_time + 620.0  # safety margin
    
    # Phase 1: GA for global exploration (use ~40% of time)
    ga_deadline = start_time + 250.0
    
    pop_size = 60
    pop = []
    best_z = None
    best_val = -1e9
    
    def init_pop():
        p = []
        # Uniform
        p.append(np.zeros(100))
        
        # Heuristic starts
        X, Y = np.meshgrid(np.arange(n), np.arange(n))
        
        # Vertical lines
        for x0 in [0, 4, 9]:
            z = np.full(100, -100.0)
            for y in range(10):
                z[y*10 + x0] = 0.0
            p.append(z)
            
        # Horizontal lines
        for y0 in [0, 4, 9]:
            z = np.full(100, -100.0)
            for x in range(10):
                z[y0*10 + x] = 0.0
            p.append(z)
            
        # Diagonal x-y = c
        for c in [-4, 0, 4]:
            z = np.full(100, -100.0)
            for y in range(10):
                x = y + c
                if 0 <= x < 10:
                    z[y*10 + x] = 0.0
            if np.sum(z > -50) > 0:
                p.append(z)
                
        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(pop_size - 9):
            p.append(rng.normal(0, 2.0, 100))
            
        return p

    pop = init_pop()
    
    # Evaluate initial population
    vals = []
    for z in pop:
        P = _softmax(z).reshape(10, 10)
        v = _value_from_P(P, idx_a, idx_b, idx_c, idx_d)
        vals.append(v)
        if v > best_val:
            best_val = v
            best_z = z.copy()
            
    # Genetic Algorithm Loop
    generation = 0
    temp = 1.0
    cooling = 0.999
    
    while time.time() < ga_deadline:
        generation += 1
        new_pop = []
        new_vals = []
        
        # Elitism: keep best
        idx_best = np.argmax(vals)
        new_pop.append(pop[idx_best].copy())
        new_vals.append(vals[idx_best])
        
        # Crossover and Mutation
        while len(new_pop) < pop_size:
            # Tournament selection
            i = np.random.choice(pop_size)
            j = np.random.choice(pop_size)
            if vals[i] >= vals[j]:
                parent1 = pop[i]
            else:
                parent1 = pop[j]
                
            k = np.random.choice(pop_size)
            l = np.random.choice(pop_size)
            if vals[k] >= vals[l]:
                parent2 = pop[k]
            else:
                parent2 = pop[l]
                
            # Uniform Crossover
            mask = np.random.rand(100) > 0.5
            child = np.where(mask, parent1, parent2)
            
            # Mutation
            noise = np.random.normal(0, temp, 100)
            child = child + noise
            
            # Occasionally, do a "restart" mutation
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

    # Phase 2: Gradient-based refinement with smooth max (use remaining time)
    # Take the best z from GA and refine with L-BFGS-B
    z0 = best_z.copy()
    
    # Run multiple optimization passes with decreasing smoothness (increasing beta)
    # Start with smooth max for better gradients, then tighten
    passes = [
        {'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-8, 'gtol': 1e-6, 'beta': 50.0},
        {'maxiter': 300, 'maxfun': 3000, 'ftol': 1e-10, 'gtol': 1e-8, 'beta': 100.0},
        {'maxiter': 400, 'maxfun': 4000, 'ftol': 1e-12, 'gtol': 1e-10, 'beta': 500.0},
        {'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-13, 'gtol': 1e-11, 'beta': 1e7},  # Hard max
    ]
    
    for opts in passes:
        if time.time() > deadline - 10.0:
            break
        beta = opts.pop('beta')
        try:
            # Create closure with specific beta
            def fun_beta(z, b=beta):
                return -f_and_grad(z, smooth_beta=b)[0]
            def fprime_beta(z, b=beta):
                return -f_and_grad(z, smooth_beta=b)[1]
            
            res = minimize(fun_beta, z0, jac=fprime_beta, method='L-BFGS-B', options=opts)
            zf = res.x
            f = -res.fun
            if np.isfinite(f) and f > best_val:
                best_val = f
                best_z = zf.copy()
            z0 = zf
        except Exception:
            break
    
    # Final check: also try refining from a few other good GA candidates
    if time.time() < deadline - 30.0:
        # Get top 3 from GA
        top_indices = np.argsort(vals)[-3:]
        for idx in top_indices:
            if time.time() > deadline - 15.0:
                break
            z0 = pop[idx].copy()
            try:
                # Use hard max for final polish
                res = minimize(fun, z0, jac=fprime, method='L-BFGS-B',
                             options={'maxiter': 200, 'maxfun': 2000,
                                      'ftol': 1e-10, 'gtol': 1e-8})
                zf = res.x
                f = -res.fun
                if np.isfinite(f) and f > best_val:
                    best_val = f
                    best_z = zf.copy()
            except Exception:
                pass
    
    # Phase 3: High-precision polish with very tight tolerances
    if time.time() < deadline - 5.0:
        try:
            res = minimize(fun, best_z, jac=fprime, method='L-BFGS-B',
                         options={'maxiter': 1000, 'maxfun': 10000,
                                  'ftol': 1e-15, 'gtol': 1e-12})
            zf = res.x
            f = -res.fun
            if np.isfinite(f) and f > best_val:
                best_val = f
                best_z = zf.copy()
        except Exception:
            pass

    P = _softmax(best_z).reshape(n, n)
    P = _finalize(P, idx_a, idx_b, idx_c, idx_d)
    return P

# EVOLVE-BLOCK-END
