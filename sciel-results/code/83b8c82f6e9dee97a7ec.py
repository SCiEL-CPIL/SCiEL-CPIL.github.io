import numpy as np
import time
import sys

def construct():
    N = 24
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    sqrt_N = np.sqrt(N)
    
    # Precompute evaluation matrix for dense sampling
    M = 4096
    theta = 2.0 * np.pi * np.arange(M) / M
    # V is M x N, V[m, k] = exp(1j * k * theta[m])
    k_idx = np.arange(N)
    V = np.exp(1j * np.outer(theta, k_idx))
    
    # Helper: fast score using only grid (no refinement)
    def fast_score(c):
        vals = V @ c
        mags = np.abs(vals)
        return mags.max() / sqrt_N
    
    # Helper: accurate score with golden-section refinement of top peaks
    def accurate_score(c):
        vals = V @ c
        mags = np.abs(vals)
        # Get top 5 peaks
        top_k = 5
        if M < top_k:
            top_idx = np.argsort(mags)[-M:]
        else:
            top_idx = np.argsort(mags)[-top_k:]
        
        best_refined = mags.max()
        
        for idx in top_idx:
            idx = int(idx)
            dtheta = 2.0 * np.pi / M
            # Refine on [theta[idx] - dtheta, theta[idx] + dtheta]
            a = theta[idx] - dtheta
            b = theta[idx] + dtheta
            
            def eval_p(t):
                return np.sum(c * np.exp(1j * k_idx * t))
            
            # Golden section
            invphi = (np.sqrt(5.0) - 1.0) / 2.0
            t1 = a + (1.0 - invphi) * (b - a)
            t2 = a + invphi * (b - a)
            f1 = np.abs(eval_p(t1))
            f2 = np.abs(eval_p(t2))
            
            for _ in range(30):
                if f1 < f2:
                    b = t2
                    t2 = t1
                    f2 = f1
                    t1 = a + (1.0 - invphi) * (b - a)
                    f1 = np.abs(eval_p(t1))
                else:
                    a = t1
                    t1 = t2
                    f1 = f2
                    t2 = a + invphi * (b - a)
                    f2 = np.abs(eval_p(t2))
            
            tm = (a + b) / 2.0
            fm = np.abs(eval_p(tm))
            if fm > best_refined:
                best_refined = fm
        
        return best_refined / sqrt_N
    
    # Generate Golay-Rudin-Shapiro sequence
    def golay_rudin_shapiro(n):
        g = [1]
        while len(g) < n:
            g = g + [x if y == 1 else -x for x, y in zip(g, g)]
            # Actually the standard construction is:
            # next = g + [x if y==1 else -x for x,y in zip(g, g)] is wrong.
            # Correct: if g = g1 + i*g2, next = g1 + g2 + i(g1 - g2)
            # In ±1 terms: if g has even length, next = g + [g_i if g_i==g_{i+1} else -g_i for i in range(len(g)//2)] ... no.
            # Let's use the known property: G_{n+1} = G_n + G_n' where G_n' is the conjugate (last bit negated for the second half? No, bit-reversed conjugate).
            # Simpler: just use the known optimal polynomials or a good heuristic.
            # The known optimal for N=24 has sup norm ~1.258 * sqrt(24) ~ 6.16.
            # Let's just use random restarts and local search.
        return g[:n]
    
    # Actually, let's just use random restarts + local search with fast scoring
    # and refine only the best candidates.
    
    def local_search(c, time_limit):
        c = c.copy()
        best_c = c.copy()
        best_score = fast_score(c)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_i = -1
            best_new_score = best_score
            for i in range(N):
                # Flip bit i
                old_val = c[i]
                c[i] = -old_val
                new_score = fast_score(c)
                c[i] = old_val
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_i = i
            
            if best_i != -1:
                c[best_i] = -c[best_i]
                best_c = c.copy()
                best_score = best_new_score
                improved = True
        
        return best_c, best_score
    
    # Initial starts
    starts = []
    
    # 1. Random starts
    np.random.seed(42)
    for _ in range(10):
        starts.append(np.random.choice([-1, 1], N))
    
    # 2. Golay-Rudin-Shapiro-like sequence
    # The GRS sequence of length 24:
    # Let's hardcode a good starting point or generate it properly.
    # GRS sequence generation:
    def gen_grs(n):
        # Start with [1]
        seq = [1]
        while len(seq) < n:
            # Next sequence: append the conjugate of the current sequence
            # Conjugate means flip the sign of the second half? No.
            # G_{k+1} = G_k G_k' where G_k' is the bit-reversed conjugate.
            # For ±1 sequences, this is equivalent to:
            # new_seq = seq + [seq[i] if i % 2 == 0 else -seq[i] for i in range(len(seq))] ? No.
            # Let's use the property: G_{k+1} = G_k + [G_k[i] * (-1)**i for i in range(len(G_k))]
            # Actually, the simplest is to look up or use a known good sequence.
            # Let's just use a simple heuristic: maximize the number of sign changes? No.
            # Let's just use random for now, but try a specific known good one if possible.
            # For N=24, a known optimal polynomial has coefficients:
            # [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1] ?
            # That's alternating, which gives high sup norm at z=1.
            # Let's stick with random restarts.
            break
        return seq[:n]
    
    # Add a few structured starts
    # Alternating
    starts.append(np.array([-1, 1] * (N // 2)))
    # Block pattern
    starts.append(np.array([1] * 12 + [-1] * 12))
    
    best_global_c = None
    best_global_score = float('inf')
    
    # Phase 1: Random restarts with local search
    for start in starts:
        if time.time() > deadline - 30.0:
            break
        
        c, s = local_search(start, deadline - 30.0)
        s_acc = accurate_score(c)
        
        if s_acc < best_global_score:
            best_global_score = s_acc
            best_global_c = c.copy()
    
    # Phase 2: Intensive local search / ILS on the best candidate
    if best_global_c is not None:
        current_c = best_global_c.copy()
        current_score = best_global_score
        
        # ILS loop
        while time.time() < deadline - 10.0:
            # Perturb
            c_pert = current_c.copy()
            # Flip 2-4 random bits
            k_pert = np.random.randint(2, 5)
            idxs = np.random.choice(N, k_pert, replace=False)
            c_pert[idxs] = -c_pert[idxs]
            
            c_new, s_new_fast = local_search(c_pert, deadline - 10.0)
            s_new_acc = accurate_score(c_new)
            
            if s_new_acc < current_score:
                current_c = c_new
                current_score = s_new_acc
                if s_new_acc < best_global_score:
                    best_global_score = s_new_acc
                    best_global_c = c_new.copy()
            else:
                # Accept with probability (simulated annealing-like)
                if np.random.rand() < 0.1:
                    current_c = c_new
                    current_score = s_new_acc
    
    # Final refinement
    if best_global_c is not None:
        # One more local search from the best
        c_final, _ = local_search(best_global_c, deadline - 5.0)
        s_final = accurate_score(c_final)
        if s_final < best_global_score:
            best_global_score = s_final
            best_global_c = c_final
    
    return list(best_global_c)
