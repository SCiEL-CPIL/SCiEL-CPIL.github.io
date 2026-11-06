import numpy as np
import time

def construct():
    N = 24
    start_time = time.time()
    deadline = start_time + 145.0
    sqrt_N = np.sqrt(N)
    
    # Dense sampling for evaluation
    M = 4096
    theta = 2.0 * np.pi * np.arange(M) / M
    k_idx = np.arange(N)
    V = np.exp(1j * np.outer(theta, k_idx))
    
    # Precompute for fast evaluation
    def fast_score(c):
        vals = V @ c
        return np.abs(vals).max() / sqrt_N
    
    # Accurate score with golden-section refinement
    def accurate_score(c):
        vals = V @ c
        mags = np.abs(vals)
        top_k = 5
        top_idx = np.argsort(mags)[-top_k:]
        
        best_refined = mags.max()
        
        for idx in top_idx:
            idx = int(idx)
            dtheta = 2.0 * np.pi / M
            a = theta[idx] - dtheta
            b = theta[idx] + dtheta
            
            def eval_p(t):
                return np.abs(np.sum(c * np.exp(1j * k_idx * t)))
            
            invphi = (np.sqrt(5.0) - 1.0) / 2.0
            t1 = a + (1.0 - invphi) * (b - a)
            t2 = a + invphi * (b - a)
            f1 = eval_p(t1)
            f2 = eval_p(t2)
            
            for _ in range(30):
                if f1 < f2:
                    b = t2
                    t2 = t1
                    f2 = f1
                    t1 = a + (1.0 - invphi) * (b - a)
                    f1 = eval_p(t1)
                else:
                    a = t1
                    t1 = t2
                    f1 = f2
                    t2 = a + invphi * (b - a)
                    f2 = eval_p(t2)
            
            tm = (a + b) / 2.0
            fm = eval_p(tm)
            if fm > best_refined:
                best_refined = fm
        
        return best_refined / sqrt_N
    
    # Efficient local search with incremental updates
    def local_search_efficient(c, time_limit):
        c = c.copy()
        best_c = c.copy()
        best_score = fast_score(c)
        
        # Precompute column norms and current values
        vals = V @ c
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_i = -1
            best_new_score = best_score
            
            for i in range(N):
                # Flip bit i
                delta = -2.0 * c[i] * V[:, i]
                new_vals = vals + delta
                new_score = np.abs(new_vals).max() / sqrt_N
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_i = i
            
            if best_i != -1:
                # Apply the flip
                delta = -2.0 * c[best_i] * V[:, best_i]
                vals = vals + delta
                c[best_i] = -c[best_i]
                best_c = c.copy()
                best_score = best_new_score
                improved = True
        
        return best_c, best_score
    
    # Generate diverse initial candidates
    np.random.seed(42)
    starts = []
    
    # Random starts
    for _ in range(15):
        starts.append(np.random.choice([-1, 1], N))
    
    # Structured starts
    starts.append(np.array([-1, 1] * (N // 2)))  # Alternating
    starts.append(np.array([1] * 12 + [-1] * 12))  # Block
    starts.append(np.array([1, 1, -1, 1] * 6))  # Pattern
    starts.append(np.array([1, -1, -1, 1] * 6))  # Pattern
    starts.append(np.array([1, 1, 1, -1] * 6))  # Pattern
    
    best_global_c = None
    best_global_score = float('inf')
    
    # Phase 1: Local search from diverse starts
    for start in starts:
        if time.time() > deadline - 30.0:
            break
        
        c, s_fast = local_search_efficient(start, deadline - 30.0)
        s_acc = accurate_score(c)
        
        if s_acc < best_global_score:
            best_global_score = s_acc
            best_global_c = c.copy()
    
    # Phase 2: Intensive iterative local search (ILS) on the best candidate
    if best_global_c is not None:
        current_c = best_global_c.copy()
        current_score = best_global_score
        
        # ILS with adaptive perturbation
        while time.time() < deadline - 10.0:
            # Perturb: flip 2-6 random bits
            c_pert = current_c.copy()
            k_pert = np.random.randint(2, 7)
            idxs = np.random.choice(N, k_pert, replace=False)
            c_pert[idxs] = -c_pert[idxs]
            
            c_new, s_new_fast = local_search_efficient(c_pert, deadline - 10.0)
            s_new_acc = accurate_score(c_new)
            
            if s_new_acc < current_score:
                current_c = c_new
                current_score = s_new_acc
                if s_new_acc < best_global_score:
                    best_global_score = s_new_acc
                    best_global_c = c_new.copy()
            elif np.random.rand() < 0.15:
                # Accept with probability (simulated annealing-like)
                current_c = c_new
                current_score = s_new_acc
    
    # Final refinement
    if best_global_c is not None:
        c_final, _ = local_search_efficient(best_global_c, deadline - 5.0)
        s_final = accurate_score(c_final)
        if s_final < best_global_score:
            best_global_score = s_final
            best_global_c = c_final
    
    return list(best_global_c)
