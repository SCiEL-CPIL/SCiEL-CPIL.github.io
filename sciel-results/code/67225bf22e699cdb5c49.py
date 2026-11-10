# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _precompute_pooling_matrices(n=10):
    """Precompute pooling matrices for slopes -1, 0, 1, 2."""
    y_idx, x_idx = np.indices((n, n))
    flat_idx = np.arange(n * n)
    
    matrices = {}
    for r in [-1, 0, 1, 2]:
        vals = (x_idx + r * y_idx).flatten()
        unique_vals, inverse = np.unique(vals, return_inverse=True)
        k = len(unique_vals)
        A = np.zeros((k, n * n), dtype=np.float64)
        A[inverse, flat_idx] = 1.0
        matrices[r] = (A, inverse)
    
    return matrices

def _compute_entropies_from_theta(theta, matrices, n):
    """Compute entropies H_r given theta (log-probabilities)."""
    # Softmax to get probabilities
    max_theta = np.max(theta)
    exp_theta = np.exp(theta - max_theta)
    P_flat = exp_theta / np.sum(exp_theta)
    
    entropies = {}
    for r in [-1, 0, 1, 2]:
        A, _ = matrices[r]
        probs = A @ P_flat
        # Ignore zeros
        mask = probs > 1e-15
        if np.sum(mask) == 0:
            entropies[r] = 0.0
        else:
            p = probs[mask]
            entropies[r] = -np.sum(p * np.log2(p))
    return entropies

def _objective_and_grad(theta, matrices, n, beta):
    """
    Objective: -[log(H_-1) - soft_max(log(H_0), log(H_1), log(H_2))]
    Gradient: computed analytically.
    """
    # 1. Compute P and Entropies
    max_theta = np.max(theta)
    exp_theta = np.exp(theta - max_theta)
    P_flat = exp_theta / np.sum(exp_theta)
    
    # Entropies and their gradients w.r.t P
    # H_r = - sum p_v log2 p_v
    # dH_r / dP_i = - (log2 p_v + 1/ln 2) where v is the bin of i
    
    A_num, inv_num = matrices[-1]
    A_0, inv_0 = matrices[0]
    A_1, inv_1 = matrices[1]
    A_2, inv_2 = matrices[2]
    
    q_num = A_num @ P_flat
    q_0 = A_0 @ P_flat
    q_1 = A_1 @ P_flat
    q_2 = A_2 @ P_flat
    
    # Clip for log safety
    eps = 1e-15
    q_num_c = np.maximum(q_num, eps)
    q_0_c = np.maximum(q_0, eps)
    q_1_c = np.maximum(q_1, eps)
    q_2_c = np.maximum(q_2, eps)
    
    H_num = -np.sum(q_num_c * np.log2(q_num_c))
    H_0 = -np.sum(q_0_c * np.log2(q_0_c))
    H_1 = -np.sum(q_1_c * np.log2(q_1_c))
    H_2 = -np.sum(q_2_c * np.log2(q_2_c))
    
    # Gradients of H w.r.t P
    dH_num_dP = -(np.log2(q_num_c)[inv_num] + 1.0 / np.log(2))
    dH_0_dP = -(np.log2(q_0_c)[inv_0] + 1.0 / np.log(2))
    dH_1_dP = -(np.log2(q_1_c)[inv_1] + 1.0 / np.log(2))
    dH_2_dP = -(np.log2(q_2_c)[inv_2] + 1.0 / np.log(2))
    
    # Gradient of log(H) w.r.t P
    # d log H / dP = (1/H) * dH/dP
    if H_num < eps:
        # Fallback if entropy is 0 (should not happen with valid P)
        obj = 1e10
        grad = np.zeros_like(theta)
        return obj, grad
        
    dlogH_num_dP = dH_num_dP / H_num
    
    # SoftMax of log(H_0), log(H_1), log(H_2)
    # Let L_r = log(H_r)
    # We want S = (1/beta) log(sum exp(beta L_r))
    # Note: H_r > 0 for valid P.
    
    L_0 = np.log(H_0) if H_0 > eps else -1e10
    L_1 = np.log(H_1) if H_1 > eps else -1e10
    L_2 = np.log(H_2) if H_2 > eps else -1e10
    
    L_vals = np.array([L_0, L_1, L_2])
    max_L = np.max(L_vals)
    exp_L = np.exp(beta * (L_vals - max_L))
    sum_exp_L = np.sum(exp_L)
    S = max_L + (1.0 / beta) * np.log(sum_exp_L)
    
    # Weights for SoftMax
    w_0 = exp_L[0] / sum_exp_L
    w_1 = exp_L[1] / sum_exp_L
    w_2 = exp_L[2] / sum_exp_L
    
    # Gradient of S w.r.t P
    # dS/dP = sum w_r * (dL_r/dP)
    # dL_r/dP = (1/H_r) * dH_r/dP
    dL_0_dP = dH_0_dP / H_0 if H_0 > eps else np.zeros_like(P_flat)
    dL_1_dP = dH_1_dP / H_1 if H_1 > eps else np.zeros_like(P_flat)
    dL_2_dP = dH_2_dP / H_2 if H_2 > eps else np.zeros_like(P_flat)
    
    dS_dP = w_0 * dL_0_dP + w_1 * dL_1_dP + w_2 * dL_2_dP
    
    # Gradient of objective f = log(H_num) - S w.r.t P
    df_dP = dlogH_num_dP - dS_dP
    
    # Gradient w.r.t theta
    # P_i = exp(theta_i) / Z
    # dP_i / dtheta_j = P_i (delta_ij - P_j)
    # df/dtheta_j = sum_i (df/dP_i) * P_i (delta_ij - P_j)
    #             = P_j * (df/dP_j - sum_i df/dP_i P_i)
    # Let V = df/dP.
    # grad_theta_j = P_j * (V_j - V @ P)
    
    V = df_dP
    V_dot_P = np.dot(V, P_flat)
    grad_theta = P_flat * (V - V_dot_P)
    
    obj = -(np.log(H_num) - S)
    
    return obj, grad_theta

def _optimize_from_start(theta0, matrices, n, time_limit, beta_schedule):
    """Run SLSQP optimization from a starting point."""
    start_time = time.time()
    best_theta = theta0.copy()
    best_score = -np.inf
    
    # We will run SLSQP with decreasing beta (increasing hardness)
    # Actually, SLSQP handles the gradient. We can just use a fixed high beta
    # or anneal it. Let's anneal beta from 10 to 1000.
    
    # Split time into 3 phases
    phases = [
        {"beta": 10.0, "time_frac": 0.4},
        {"beta": 100.0, "time_frac": 0.4},
        {"beta": 1000.0, "time_frac": 0.2}
    ]
    
    current_theta = theta0.copy()
    
    for phase in phases:
        if time.time() - start_time > time_limit:
            break
            
        beta = phase["beta"]
        phase_time_limit = time_limit * phase["time_frac"]
        phase_deadline = time.time() + phase_time_limit
        
        # Define objective for this phase
        def obj_fun(th):
            if time.time() > phase_deadline:
                return 1e10, np.zeros_like(th)
            return _objective_and_grad(th, matrices, n, beta)
        
        try:
            res = minimize(
                obj_fun,
                current_theta,
                method='L-BFGS-B', # L-BFGS-B is often faster and robust for this size
                jac=True, # We provide gradient
                options={'maxiter': 5000, 'maxfun': 5000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.success or res.fun < best_score:
                current_theta = res.x
                # Evaluate true score (hard max) to update best
                ent = _compute_entropies_from_theta(current_theta, matrices, n)
                D = ent[-1]
                M = max(ent[0], ent[1], ent[2])
                if M > 1e-9:
                    score = D / M
                    if score > best_score:
                        best_score = score
                        best_theta = current_theta.copy()
        except Exception:
            pass

    return best_theta, best_score

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    deadline = start_time + 620 # Safety margin
    
    matrices = _precompute_pooling_matrices(n)
    rng = np.random.default_rng(42)
    
    best_P = np.ones((n, n)) / (n * n)
    best_score = -np.inf
    
    # Generate diverse initial points in theta space
    # P_i = exp(theta_i)/Z. Uniform P corresponds to theta_i = 0.
    # We can perturb theta.
    
    inits = []
    # 1. Uniform
    inits.append(np.zeros(n*n))
    
    # 2. Random perturbations of uniform
    for _ in range(20):
        inits.append(rng.normal(0, 0.5, n*n))
        
    # 3. Structured inits (converted to theta)
    # Diagonal emphasis
    for w in [0.5, 1.0, 2.0]:
        P = np.ones((n, n)) * 0.1
        for i in range(n):
            P[i, i] = w
        P = P / P.sum()
        # theta = log(P) + log(Z) ~ log(P)
        # We can just use log(P) as a valid theta (up to constant)
        inits.append(np.log(np.maximum(P, 1e-10)).flatten())
        
    # Anti-diagonal
    for w in [0.5, 1.0, 2.0]:
        P = np.ones((n, n)) * 0.1
        for i in range(n):
            P[i, n-1-i] = w
        P = P / P.sum()
        inits.append(np.log(np.maximum(P, 1e-10)).flatten())

    # Sort inits by initial score (using hard max) to prioritize good starts
    scored_inits = []
    for th in inits:
        ent = _compute_entropies_from_theta(th, matrices, n)
        D = ent[-1]
        M = max(ent[0], ent[1], ent[2])
        score = D / M if M > 1e-9 else 0
        scored_inits.append((score, th))
        
    scored_inits.sort(key=lambda x: x[0], reverse=True)
    
    # Run optimization on top candidates
    # We have ~600s. Let's allocate time to top 10 candidates.
    top_k = 10
    for i, (init_score, th0) in enumerate(scored_inits[:top_k]):
        if time.time() > deadline - 30:
            break
            
        # Dynamic time allocation
        remaining = deadline - time.time()
        if i < 3:
            t_lim = min(remaining * 0.3, 100)
        elif i < 7:
            t_lim = min(remaining * 0.2, 60)
        else:
            t_lim = min(remaining * 0.1, 30)
            
        if t_lim < 5:
            continue
            
        try:
            best_th, score = _optimize_from_start(th0, matrices, n, t_lim, None)
            if score > best_score:
                best_score = score
                # Convert back to P
                max_th = np.max(best_th)
                exp_th = np.exp(best_th - max_th)
                best_P = (exp_th / np.sum(exp_th)).reshape(n, n)
        except Exception:
            continue
            
    # Final polish on the best found
    if time.time() < deadline - 10:
        # Convert best_P back to theta for polishing
        # theta = log(P)
        th_best = np.log(np.maximum(best_P, 1e-15)).flatten()
        
        # Run one more high-beta optimization
        remaining = deadline - time.time() - 5
        if remaining > 5:
            try:
                # Use a very high beta to sharpen the max
                def final_obj(th):
                    return _objective_and_grad(th, matrices, n, 5000.0)
                
                res = minimize(
                    final_obj,
                    th_best,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 1000, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.fun < 1e10: # Check if it didn't timeout/fail
                    max_th = np.max(res.x)
                    exp_th = np.exp(res.x - max_th)
                    P_candidate = (exp_th / np.sum(exp_th)).reshape(n, n)
                    
                    # Validate
                    ent = _compute_entropies_from_theta(res.x, matrices, n)
                    D = ent[-1]
                    M = max(ent[0], ent[1], ent[2])
                    if M > 1e-9:
                        score = D / M
                        if score > best_score:
                            best_score = score
                            best_P = P_candidate
            except Exception:
                pass

    # Ensure validity
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s > 1e-15:
        best_P = best_P / s
    else:
        best_P = np.ones((n, n)) / (n * n)
        
    return best_P

# EVOLVE-BLOCK-END
