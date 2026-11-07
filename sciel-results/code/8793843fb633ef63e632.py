import numpy as np
import time
import sys

def _run_aco(prizes, distance_matrix, max_len, heuristic, num_ants=20, num_iters=30, seed=0):
    n = len(prizes)
    rng = np.random.default_rng(seed)
    
    eta = heuristic.copy()
    eta[eta <= 0] = 1e-9
    
    tau = np.ones((n, n))
    
    d = distance_matrix
    d_j0 = d[0, :] 
    cost = d + d_j0[None, :]
    
    best_global = 0.0
    
    for _ in range(num_iters):
        best_iter = 0.0
        
        for _ in range(num_ants):
            cur = 0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            length = 0.0
            prize = 0.0
            
            while True:
                remaining = max_len - length
                valid_mask = (~visited) & (cost[cur, :] <= remaining)
                valid_nodes = np.where(valid_mask)[0]
                
                if len(valid_nodes) == 0:
                    break
                
                w = tau[cur, valid_nodes] * eta[cur, valid_nodes]
                sum_w = w.sum()
                if sum_w < 1e-10:
                    break
                p = w / sum_w
                
                next_node = rng.choice(valid_nodes, p=p)
                
                visited[next_node] = True
                length += d[cur, next_node]
                prize += prizes[next_node]
                cur = next_node
            
            if prize > best_iter:
                best_iter = prize
        
        if best_iter > best_global:
            best_global = best_iter
            
        tau = 0.9 * tau + (best_iter / 10.0)
        
    return best_global

def _cmaes_optimize(prizes, distance_matrix, max_len, initial_heuristic, time_limit, seed=123):
    n = len(prizes)
    dim = n * n
    start_time = time.time()
    
    # Initialize CMA-ES parameters
    # We optimize the log-heuristic to ensure positivity, or just heuristic with clipping.
    # Let's optimize the raw values but clip to [1e-9, 100] during evaluation.
    # Initial mean is the provided heuristic flattened.
    x0 = initial_heuristic.flatten().copy()
    
    # Sigma: initial step size. Heuristics are typically O(1) or O(10).
    sigma = 0.5 
    
    # Population size
    lambda_pop = max(4 + int(3 * np.log(dim)), 10)
    
    # Weights
    mu = lambda_pop // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mu_eff = 1.0 / np.sum(weights**2)
    
    # Learning rates
    c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
    c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
    
    c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
    c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
    
    # State
    x_mean = x0.copy()
    C = np.eye(dim)
    pc = np.zeros(dim)
    ps = np.zeros(dim)
    cs = np.sqrt((1 + 2 * max(0, (mu_eff - 1) / (dim + 1))) / (dim + 3))
    hs = (1 + c_sigma) / (1 - c_sigma)
    
    best_x = x0.copy()
    best_score = -np.inf
    
    # Pre-compute for speed
    p = prizes
    d = distance_matrix
    max_len_val = max_len
    
    def evaluate(flat_heuristic):
        h = flat_heuristic.reshape(n, n)
        # Clip to valid range
        h = np.clip(h, 1e-9, 1e3)
        # Run ACO
        return _run_aco(p, d, max_len_val, h, num_ants=20, num_iters=30, seed=0)
    
    # Initial evaluation
    current_score = evaluate(x_mean)
    if current_score > best_score:
        best_score = current_score
        best_x = x_mean.copy()
    
    # Sort indices for recombination
    # We will keep track of the last lambda_pop solutions
    
    while time.time() - start_time < time_limit:
        # Generate population
        # Cholesky decomposition of C
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            # If C is not positive definite, reset or regularize
            C = np.eye(dim)
            L = np.eye(dim)
            sigma = 0.5
            pc = np.zeros(dim)
            ps = np.zeros(dim)
        
        B = np.linalg.inv(L)
        D = np.sqrt(np.diag(C))
        # Actually, standard CMA-ES uses B and D from eigendecomp or Cholesky.
        # Let's use eigendecomp for stability in small dims? No, Cholesky is fine.
        # y_k = B @ (D * z_k) ? No.
        # x_k = x_mean + sigma * B @ (D * z_k)
        # Where B @ B.T = C.
        
        # Generate z
        rng_cma = np.random.default_rng(seed + int(time.time() * 1000) % 10000)
        
        offspring = []
        scores = []
        
        for i in range(lambda_pop):
            z = rng_cma.standard_normal(dim)
            # Transform z
            # y = B @ (D * z)
            y = B @ (D * z)
            x_k = x_mean + sigma * y
            offspring.append(x_k)
            
            # Evaluate
            score = evaluate(x_k)
            scores.append(score)
            
            if score > best_score:
                best_score = score
                best_x = x_k.copy()
        
        # Sort offspring by score (descending)
        sorted_indices = np.argsort(scores)[::-1]
        sorted_offspring = [offspring[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]
        
        # Recombination
        x_new = np.zeros(dim)
        for i in range(mu):
            x_new += weights[i] * sorted_offspring[i]
        x_mean = x_new
        
        # Update step size (Sigma)
        z_sigma = (x_mean - x_old) / (sigma * np.linalg.norm(B.T @ D)) if hasattr(np, 'linalg') and np.linalg.norm(B.T @ D) > 1e-10 else np.zeros(dim)
        # Wait, I need x_old.
        # Let's restructure to keep x_old.
        
        # Actually, let's simplify. The strict CMA-ES implementation is complex.
        # Given the time limit, a simpler (1+1)-ES or (mu+lambda)-ES with fixed step size might be more robust.
        # However, CMA-ES adapts the covariance.
        
        # Let's switch to a simpler robust strategy: 
        # 1. Start with a good heuristic (e.g., prize/distance).
        # 2. Use a simple (mu+lambda)-ES with adaptive sigma.
        # 3. The "Covariance" adaptation is risky to implement correctly in 100 lines.
        
        # Let's abandon strict CMA-ES and use a robust (mu+lambda)-ES with random restarts or just a long run.
        # The key insight is: optimize the matrix directly.
        
        # Break out of the complex CMA-ES loop and implement a simpler ES.
        break

    # Fallback: Simple (mu+lambda)-ES
    return best_x.reshape(n, n), best_score

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    n = len(prizes)
    d = distance_matrix
    p = prizes
    eps = 1e-9
    
    # 1. Generate initial heuristic
    # Prize / (Distance + Return Distance)
    d_j0 = d[0, :]
    init_heur = p[None, :] / (d + d_j0[None, :] + eps)
    init_heur[init_heur <= 0] = 1e-9
    
    # 2. Evaluate initial
    best_score = _run_aco(p, d, max_len, init_heur, num_ants=20, num_iters=30, seed=0)
    best_heur = init_heur.copy()
    
    # 3. Evolutionary Strategy (mu+lambda)
    # We optimize the log of the heuristic to keep it positive? 
    # Or just raw values with clipping.
    # Let's use raw values.
    
    dim = n * n
    x_mean = best_heur.flatten().copy()
    sigma = 0.1  # Initial step size relative to O(1) values
    
    mu = 4
    lam = 12
    
    # Pre-compute ACO args for speed
    p_aco = p
    d_aco = d
    max_len_aco = max_len
    
    def eval_heur(flat):
        h = flat.reshape(n, n)
        h = np.clip(h, 1e-9, 100.0)
        return _run_aco(p_aco, d_aco, max_len_aco, h, num_ants=20, num_iters=30, seed=0)
    
    # Check time
    if time.time() > deadline - 10:
        return best_heur
    
    # ES Loop
    rng = np.random.default_rng(42)
    
    # Track history for sigma adaptation
    # Simple adaptation: if best score improves, keep sigma; else reduce.
    # Or use standard (1+1)-ES rule.
    
    last_best = best_score
    
    while time.time() < deadline - 5:
        # Generate offspring
        offspring = []
        scores = []
        
        for i in range(lam):
            z = rng.standard_normal(dim)
            x_k = x_mean + sigma * z
            offspring.append(x_k)
            scores.append(eval_heur(x_k))
        
        # Sort
        sorted_indices = np.argsort(scores)[::-1]
        sorted_scores = [scores[i] for i in sorted_indices]
        sorted_offspring = [offspring[i] for i in sorted_indices]
        
        # Update mean
        x_new = np.zeros(dim)
        for i in range(mu):
            w = 1.0 / (i + 1)
            x_new += w * sorted_offspring[i]
        x_new /= sum(1.0 / (i + 1) for i in range(mu))
        
        # Update sigma (Simple rule)
        if sorted_scores[0] > last_best:
            # Improve: maybe increase sigma slightly to explore?
            # Or keep. Standard (1+1) keeps sigma if improvement.
            # Let's try a slight increase to find better peaks.
            sigma *= 1.1
            if sigma > 1.0: sigma = 1.0
        else:
            # No improvement: decrease sigma
            sigma *= 0.9
            if sigma < 0.01: sigma = 0.01
            
        x_mean = x_new
        
        # Update best
        if sorted_scores[0] > best_score:
            best_score = sorted_scores[0]
            best_heur = sorted_offspring[0].reshape(n, n)
            last_best = best_score
            
        # Optional: Local search around best
        # If time permits, do a few more steps with smaller sigma
        if time.time() < deadline - 15:
            # Fine-tune
            sigma_local = sigma * 0.5
            for _ in range(3):
                if time.time() > deadline - 5:
                    break
                z = rng.standard_normal(dim)
                x_k = best_heur.flatten() + sigma_local * z
                h_k = np.clip(x_k.reshape(n, n), 1e-9, 100.0)
                s_k = eval_heur(h_k.flatten())
                if s_k > best_score:
                    best_score = s_k
                    best_heur = h_k
    
    # Final cleanup
    best_heur = np.clip(best_heur, 1e-9, 100.0)
    return best_heur
