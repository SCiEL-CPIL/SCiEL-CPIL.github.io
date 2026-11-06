import numpy as np
import time

def construct():
    deadline = time.time() + 115.0  # Safety margin for 120s limit
    
    N_COEFF = 24
    sqrt_N = np.sqrt(N_COEFF)
    
    # Precompute z^k for k=0..23 on a dense grid for fast scoring
    N_GRID = 2048
    theta = 2 * np.pi * np.arange(N_GRID) / N_GRID
    z = np.exp(1j * theta)
    # Z[k, m] = z_m^k
    Z = z[None, :] ** np.arange(N_COEFF)[:, None]
    
    # Precompute powers for golden-section refinement
    KC = np.arange(N_COEFF)
    
    def score_fast(coeffs):
        """Fast evaluation using dense grid only."""
        vals = coeffs @ Z
        return np.max(np.abs(vals))

    def score_precise(coeffs):
        """Precise evaluation with golden-section refinement of top peaks."""
        vals = coeffs @ Z
        abs_vals = np.abs(vals)
        m = np.max(abs_vals)
        
        top_k = 10
        idx = np.argpartition(abs_vals, -top_k)[-top_k:]
        best = m
        
        dtheta = 2 * np.pi / N_GRID
        gr = (np.sqrt(5) - 1) / 2
        
        for i in idx:
            lo_idx = (i - 1) % N_GRID
            hi_idx = (i + 1) % N_GRID
            a = theta[lo_idx]
            b = theta[hi_idx]
            if a > b:
                b += 2 * np.pi
            
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            for _ in range(50):
                zc = np.exp(1j * c)
                zd = np.exp(1j * d)
                pc = np.dot(coeffs, zc ** KC)
                pd = np.dot(coeffs, zd ** KC)
                ac = abs(pc)
                ad = abs(pd)
                
                if ac < ad:
                    b = d
                    d = c
                    ad = ac
                    c = b - gr * (b - a)
                else:
                    a = c
                    c = d
                    ac = ad
                    d = a + gr * (b - a)
                
                if time.time() > deadline - 5:
                    break
            
            zm = np.exp(1j * (a + b) / 2)
            pm = np.dot(coeffs, zm ** KC)
            am = abs(pm)
            if am > best:
                best = am
        
        return best

    best_coeffs = None
    best_score = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Generate Seeds
    # Seed 1: Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < N_COEFF:
        p, q = p + q, p + [-c for c in q]
    gr_coeffs = np.array(p[:N_COEFF], dtype=np.float64)
    
    seeds = [gr_coeffs]
    for _ in range(20):
        seeds.append(rng.choice([-1.0, 1.0], size=N_COEFF))
    
    # Local search from each seed with single and double flips
    for s_idx, seed in enumerate(seeds):
        if time.time() > deadline - 10:
            break
        
        coeffs = seed.copy()
        cur_score = score_precise(coeffs)
        cur_fast_score = score_fast(coeffs)
        
        if cur_score < best_score:
            best_score = cur_score
            best_coeffs = coeffs.copy()
            
        P_vals = coeffs @ Z
        
        improved = True
        flip_type = 0
        while improved and time.time() < deadline - 15:
            improved = False
            best_flip_score = cur_score
            best_flip_fast = cur_fast_score
            best_flip_indices = None
            
            if flip_type == 0:
                # Single flip
                for k in range(N_COEFF):
                    P_new = P_vals - 2.0 * coeffs[k] * Z[k]
                    fast_score = np.max(np.abs(P_new))
                    
                    if fast_score < best_flip_fast:
                        new_coeffs = coeffs.copy()
                        new_coeffs[k] *= -1
                        precise_score = score_precise(new_coeffs)
                        
                        if precise_score < best_flip_score:
                            best_flip_score = precise_score
                            best_flip_fast = fast_score
                            best_flip_indices = [k]
            else:
                # Double flip - try all pairs
                for i in range(N_COEFF):
                    if time.time() > deadline - 15:
                        break
                    for j in range(i + 1, N_COEFF):
                        P_new = P_vals - 2.0 * coeffs[i] * Z[i] - 2.0 * coeffs[j] * Z[j]
                        fast_score = np.max(np.abs(P_new))
                        
                        if fast_score < best_flip_fast:
                            new_coeffs = coeffs.copy()
                            new_coeffs[i] *= -1
                            new_coeffs[j] *= -1
                            precise_score = score_precise(new_coeffs)
                            
                            if precise_score < best_flip_score:
                                best_flip_score = precise_score
                                best_flip_fast = fast_score
                                best_flip_indices = [i, j]
            
            if best_flip_indices is not None:
                for k in best_flip_indices:
                    coeffs[k] *= -1
                cur_score = best_flip_score
                cur_fast_score = best_flip_fast
                P_vals = coeffs @ Z
                improved = True
                if cur_score < best_score:
                    best_score = cur_score
                    best_coeffs = coeffs.copy()
            
            if not improved:
                flip_type = 1 - flip_type
    
    # Genetic algorithm phase for broader exploration
    if time.time() < deadline - 20 and best_coeffs is not None:
        pop_size = 30
        population = [best_coeffs.copy()]
        for _ in range(pop_size - 1):
            ind = best_coeffs.copy()
            # Mutate: flip 1-3 random coefficients
            n_mut = rng.integers(1, 4)
            for _ in range(n_mut):
                k = rng.integers(0, N_COEFF)
                ind[k] *= -1
            population.append(ind)
        
        # Evaluate initial population
        pop_scores = [score_precise(ind) for ind in population]
        
        generations = 0
        while time.time() < deadline - 10:
            generations += 1
            
            # Sort by score
            sorted_idx = np.argsort(pop_scores)
            population = [population[i] for i in sorted_idx]
            pop_scores = [pop_scores[i] for i in sorted_idx]
            
            # Update best
            if pop_scores[0] < best_score:
                best_score = pop_scores[0]
                best_coeffs = population[0].copy()
            
            # Create next generation
            new_pop = [population[0].copy()]  # Elitism
            
            while len(new_pop) < pop_size:
                if time.time() > deadline - 10:
                    break
                
                # Tournament selection
                def tournament():
                    t_idx = rng.integers(0, pop_size, size=3)
                    best_t = min(t_idx, key=lambda x: pop_scores[x])
                    return population[best_t].copy()
                
                parent1 = tournament()
                parent2 = tournament()
                
                # Uniform crossover
                mask = rng.random(N_COEFF) < 0.5
                child = np.where(mask, parent1, parent2)
                
                # Mutation: flip 1-2 random coefficients
                n_mut = rng.integers(1, 3)
                for _ in range(n_mut):
                    k = rng.integers(0, N_COEFF)
                    child[k] *= -1
                
                new_pop.append(child)
            
            # Evaluate new population
            population = new_pop
            pop_scores = [score_precise(ind) for ind in population]
    
    # Final polish: try random perturbations of the best solution
    if best_coeffs is not None and time.time() < deadline - 5:
        for _ in range(100):
            if time.time() > deadline - 2:
                break
            cand = best_coeffs.copy()
            n_flips = rng.integers(1, 4)
            for _ in range(n_flips):
                k = rng.integers(0, N_COEFF)
                cand[k] *= -1
            s = score_precise(cand)
            if s < best_score:
                best_score = s
                best_coeffs = cand.copy()

    if best_coeffs is None:
        return [1] * 24
    return [int(c) for c in best_coeffs]
