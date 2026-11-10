# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximizes the ratio D/M where:
    D = H(x - y)
    M = max(H(x), H(x + y), H(x + 2y))
    
    Strategy:
    1. Parameterize the problem by the marginal distribution of D = X - Y (19 variables).
    2. For a fixed marginal P_D, the entropy D = H(P_D) is fixed.
    3. To maximize D/M, we need to minimize M subject to the constraint that the 
       marginal of X-Y is P_D.
    4. M = max(H_X, H_XY, H_X2Y). Minimizing the maximum of convex functions (entropy 
       is concave, but we are minimizing it? No, we want to MINIMIZE M. 
       Wait. We want to MAXIMIZE D/M. 
       If we fix D, we want to MINIMIZE M.
       M is the max of entropies. Entropy is concave. Minimizing a concave function 
       is non-convex. However, the set of joint distributions with a fixed marginal 
       is a convex polytope.
       
       Actually, let's look at the structure. 
       The entropy H(X) is a function of the marginal P_X. 
       H(X+Y) is a function of the marginal P_XY.
       H(X+2Y) is a function of the marginal P_X2Y.
       
       If we fix P_D, we can search over the space of joint distributions that 
       respect P_D. But that space is still high dimensional.
       
       Alternative Insight:
       The term M is the max of three entropies. 
       Usually, the "worst" case for M (lowest M) given a D distribution 
       might be achieved by a distribution that is "as concentrated as possible" 
       in the other projections, or perhaps a specific structured distribution.
       
       Let's try a different angle: 
       We can optimize over the 100 joint probabilities, but use the marginal 
       structure to guide the search or reduce dimensionality.
       
       Actually, the prompt suggests: "Model the problem as a convex optimization 
       over the 19 marginal probabilities of X-Y ... minimizing the denominator 
       for a fixed high-entropy numerator structure".
       
       This implies:
       1. Iterate over candidate P_D (19-dim simplex).
       2. For each P_D, solve: min_{P_joint} M(P_joint) s.t. marginal(P_joint)_D = P_D, P_joint >= 0, sum=1.
       
       Is minimizing M convex? 
       M = max(H_X, H_XY, H_X2Y). 
       H is concave. -H is convex. 
       Minimizing a max of concave functions is generally non-convex.
       However, if we introduce an auxiliary variable m such that m >= H_X, m >= H_XY, m >= H_X2Y, and minimize m, 
       the constraints m >= H_X are non-convex (since H_X is concave, the set {p | H_X(p) <= m} is not convex).
       
       Wait, the entropy function is concave. The set {p | H(p) <= m} is non-convex. 
       So this subproblem is non-convex.
       
       However, there is a known result that for fixed marginals, the joint distribution 
       that minimizes the entropy of a function of the variables often has a specific structure (e.g., uniform on the support defined by the marginals, or concentrated on minimal support).
       
       Let's try a heuristic approach for the subproblem:
       Given P_D, we want a joint P that minimizes M.
       We can use local search (L-BFGS-B) for the 100-dim joint P, but initialized 
       smartly based on P_D, and only run for a few iterations to find a local min of M.
       Then we evaluate D/M.
       
       This is essentially "nested optimization". 
       Outer loop: Optimize P_D (19 params).
       Inner loop: Optimize P_joint (100 params) to minimize M subject to marginal D = P_D.
       
       This might be too slow if done naively.
       
       Alternative: 
       Observe that the optimal P_joint for minimizing M given P_D might be "uniform" over the set of cells consistent with the dominant mass of P_D?
       
       Let's try a simpler different strategy first: 
       The current program optimizes 100 params. 
       The inspiration suggests convex optimization over 19 params.
       
       Let's look at the constraints.
       X in 0..9, Y in 0..9.
       D = X - Y in -9..9.
       
       If we assume the optimal joint distribution is "banded" or has low rank, we can reduce parameters.
       
       Let's try this specific strategy:
       1. Parameterize P_joint using a low-rank factorization or a "mixture of lines" model.
       2. Or, simply use the 19-dim P_D as the primary search space, and for each P_D, 
          construct a candidate P_joint that is the "maximum entropy" distribution given P_D? 
          No, max entropy would maximize M. We want to MINIMIZE M.
          
       Minimum entropy given marginals?
       The distribution with minimum entropy given a marginal constraint is often concentrated on the smallest possible support that satisfies the marginal.
       
       Let's try to construct the "minimum M" joint for a given P_D:
       For each d, we have mass p_d. We need to distribute this mass among cells (x, y) with x-y=d.
       To minimize H(X), H(X+Y), H(X+2Y), we should concentrate the mass in these projections.
       
       This is a complex combinatorial problem.
       
       Let's stick to a robust optimization but change the parameterization.
       Instead of 100 free params, we can use a "mixture of product distributions" or similar.
       
       Actually, let's look at the "Convex Optimization" hint again.
       Maybe the hint implies that we can bound M?
       
       Let's try a different tactic: **Direct Optimization of the Ratio with a Different Initializer and Line Search**.
       The current program uses L-BFGS-B on log-probs.
       I will use **Nelder-Mead** on the 100 probabilities (normalized) directly, with a very high number of restarts, but using a **structured initialization** that places mass on the "anti-diagonals" (constant X-Y) to start with high D, and then let the optimizer adjust to reduce M.
       
       Wait, Nelder-Mead on 100 dims is slow.
       
       Let's try **Particle Swarm Optimization (PSO)** or **Simulated Annealing** on the 100-dim space.
       
       Actually, the most promising "different" strategy from the hint is:
       **Optimize the 19 marginal probabilities P_D, and for each P_D, use a fast heuristic to generate a P_joint that minimizes M.**
       
       Heuristic for Min-M given P_D:
       For each d, distribute p_d among the available cells (x, y) such that x-y=d.
       To minimize H(X), we want X to be as concentrated as possible.
       To minimize H(X+Y), we want X+Y to be as concentrated as possible.
       
       This suggests putting mass on cells that are "central" in X, X+Y, X+2Y.
       
       Let's implement:
       1. Outer loop: Optimize P_D (19 params) using L-BFGS-B or Nelder-Mead.
       2. Inner step: Given P_D, construct P_joint:
          - For each d, identify cells C_d = {(x,y) | x-y=d}.
          - Assign weights to C_d proportional to a "centrality" score based on how "central" x, x+y, x+2y are.
          - Normalize within each d.
          - This creates a specific P_joint for each P_D.
       3. Evaluate D/M for this constructed P_joint.
       4. Optimize P_D to maximize this score.
       
       This reduces the problem to 19 dimensions for the main optimization, and the inner step is deterministic and fast.
       
       Let's refine the "Centrality" heuristic.
       We want to minimize the max of entropies.
       Entropy is maximized by uniform distribution.
       So we want to avoid uniformity in X, X+Y, X+2Y.
       We want to concentrate mass.
       
       For a given d, the cells are (d+y, y) for y in max(0, -d)..min(9, 9-d).
       Let's pick the cell(s) in C_d that are closest to the "center" of the grid in terms of X, X+Y, X+2Y.
       Center of X is 4.5. Center of X+Y is 9. Center of X+2Y is 13.5.
       
       Score for cell (x,y) = -(|x - 4.5| + |x+y - 9| + |x+2y - 13.5|).
       Assign probability proportional to exp(Score * beta).
       
       This creates a "focused" distribution.
       
       Then we optimize P_D to maximize H(P_D) / M(constructed_P).
       
       This seems computationally feasible and structurally different.
    """
    n = 10
    deadline = time.time() + 590.0
    
    # Precompute cells for each d
    cells_by_d = {d: [] for d in range(-9, 10)}
    for y in range(n):
        for x in range(n):
            d = x - y
            cells_by_d[d].append((x, y))
    
    # Centers
    c_x = 4.5
    c_xy = 9.0
    c_x2y = 13.5
    
    def construct_joint_from_marginal(p_d, beta=1.0):
        """
        p_d: array of 19 probabilities for d=-9..9
        Returns: 10x10 array P
        """
        P = np.zeros((n, n))
        for i, d in enumerate(range(-9, 10)):
            mass = p_d[i]
            if mass < 1e-15:
                continue
            cells = cells_by_d[d]
            if not cells:
                continue
            
            # Calculate centrality scores
            scores = []
            for x, y in cells:
                s = -np.abs(x - c_x) - np.abs(x + y - c_xy) - np.abs(x + 2*y - c_x2y)
                scores.append(s)
            scores = np.array(scores)
            
            # Softmax with temperature 1/beta
            # Higher beta -> more concentrated
            scores_scaled = scores * beta
            scores_scaled -= np.max(scores_scaled)
            weights = np.exp(scores_scaled)
            weights /= np.sum(weights)
            
            # Assign mass
            for (x, y), w in zip(cells, weights):
                P[y, x] += mass * w
                
        return P

    def compute_score(P):
        """
        Compute D/M ratio for a given 10x10 distribution P.
        """
        P = P.flatten()
        # Marginals
        ys, xs = np.mgrid[0:n, 0:n]
        d_val = (xs - ys).flatten()
        m0_val = xs.flatten()
        m1_val = (xs + ys).flatten()
        m2_val = (xs + 2 * ys).flatten()
        
        d_vals, d_inv = np.unique(d_val, return_inverse=True)
        m0_vals, m0_inv = np.unique(m0_val, return_inverse=True)
        m1_vals, m1_inv = np.unique(m1_val, return_inverse=True)
        m2_vals, m2_inv = np.unique(m2_val, return_inverse=True)
        
        num_d = len(d_vals)
        num_m0 = len(m0_vals)
        num_m1 = len(m1_vals)
        num_m2 = len(m2_vals)
        
        p_d = np.bincount(d_inv, weights=P, minlength=num_d)
        p_m0 = np.bincount(m0_inv, weights=P, minlength=num_m0)
        p_m1 = np.bincount(m1_inv, weights=P, minlength=num_m1)
        p_m2 = np.bincount(m2_inv, weights=P, minlength=num_m2)
        
        def shannon(p):
            p = p[p > 1e-15]
            if len(p) == 0: return 0.0
            return -np.sum(p * np.log2(p))
            
        H_d = shannon(p_d)
        H_m0 = shannon(p_m0)
        H_m1 = shannon(p_m1)
        H_m2 = shannon(p_m2)
        
        M = max(H_m0, H_m1, H_m2)
        if M < 1e-9:
            return 0.0
        return H_d / M

    # Outer Optimization: Optimize p_d (19 params)
    # We use log-probabilities for p_d to ensure positivity and sum-to-1 via softmax
    def objective_p_d(log_p_d_params):
        # Softmax to get p_d
        log_p_d = log_p_d_params - np.max(log_p_d_params)
        p_d = np.exp(log_p_d)
        p_d /= np.sum(p_d)
        
        # Try different beta values to find the best construction
        # We can optimize beta as well, or just pick a good one.
        # Let's try a few betas and take the best? No, that makes objective non-smooth.
        # Let's fix beta to a value that creates moderate concentration.
        # Or, we can add beta as a 20th parameter (log_beta).
        
        # Let's optimize log_beta as well.
        # log_p_d_params[0:19] are log probs for d
        # log_p_d_params[19] is log_beta
        
        log_beta = log_p_d_params[19]
        beta = np.exp(log_beta)
        
        P = construct_joint_from_marginal(p_d, beta)
        score = compute_score(P)
        return -score

    # Initial guess
    # Uniform p_d
    init_params = np.zeros(20)
    # Beta around 1.0 (log(1)=0)
    
    rng = np.random.default_rng(42)
    
    best_score = 0.0
    best_params = init_params.copy()
    
    # Run optimization
    # Use L-BFGS-B with multiple restarts
    for restart in range(50):
        if time.time() > deadline - 30:
            break
            
        if restart == 0:
            start_params = init_params.copy()
        else:
            # Random perturbation
            start_params = best_params + rng.normal(0, 0.5, 20)
            
        try:
            res = minimize(
                objective_p_d,
                start_params,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            if res.success or res.fun < -best_score:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
        except Exception:
            pass

    # Get best P
    log_p_d = best_params[:19]
    log_beta = best_params[19]
    beta = np.exp(log_beta)
    
    log_p_d_norm = log_p_d - np.max(log_p_d)
    p_d = np.exp(log_p_d_norm)
    p_d /= np.sum(p_d)
    
    P = construct_joint_from_marginal(p_d, beta)
    
    # Final Polish: Now that we have a good P, do a few steps of direct 100-dim optimization
    # to escape the "constructed" structure if needed.
    if time.time() < deadline - 10:
        # Convert P to log-probs
        P_flat = P.flatten()
        P_flat[P_flat < 1e-15] = 1e-15
        P_flat /= np.sum(P_flat)
        log_p_flat = np.log(P_flat)
        
        def objective_joint(log_p_flat_params):
            if time.time() > deadline - 5:
                return 1e9
            log_p = log_p_flat_params - np.max(log_p_flat_params)
            p = np.exp(log_p)
            p /= np.sum(p)
            P_test = p.reshape((n, n))
            score = compute_score(P_test)
            return -score
        
        try:
            res_joint = minimize(
                objective_joint,
                log_p_flat,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res_joint.fun < -best_score:
                best_score = -res_joint.fun
                P = np.exp(res_joint.x - np.max(res_joint.x)).reshape((n, n))
                P /= np.sum(P)
        except Exception:
            pass

    # Final Validation and Return
    P = np.clip(P, 0.0, None)
    s = np.sum(P)
    if s > 1e-9:
        P = P / s
    else:
        P = np.ones((n, n)) / (n * n)
        
    return P

# EVOLVE-BLOCK-END
