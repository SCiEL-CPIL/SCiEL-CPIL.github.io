import numpy as np
import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # Precompute all arithmetic progressions as index lists (0-based)
    # An AP: start a (0-based), step s >= 1, length k >= 1
    # indices: a, a+s, a+2s, ..., a+(k-1)s where a+(k-1)s < n
    progs = []
    for a in range(n):
        for s in range(1, n):
            idx = list(range(a, n, s))
            if len(idx) >= 1:
                progs.append(idx)
    
    Np = len(progs)
    
    # Build dense incidence matrix M of shape (n, Np)
    # M[i, pid] = 1 if position i is in progression pid, else 0
    # Using int16 to save memory and speed up matmul slightly, or float32 for GD
    M = np.zeros((n, Np), dtype=np.float32)
    for pid, idx in enumerate(progs):
        for i in idx:
            M[i, pid] = 1.0
            
    # M_T is Np x n, useful for gradients
    M_T = M.T
    
    def compute_sums(signs):
        # signs is (n,) vector
        # sums is (Np,) vector of signed sums
        return signs @ M

    def max_abs_sum(sums):
        return int(np.max(np.abs(sums)))

    def soft_max_abs(sums, beta=10.0):
        # Differentiable approximation of max(|sums|)
        # Using log-sum-exp of exp(beta * |sums|) / beta
        # To avoid overflow, subtract max
        abs_sums = np.abs(sums)
        max_val = np.max(abs_sums)
        # If max_val is very large, beta*max_val might overflow in exp
        # But we are in float32. exp(88) is max. 
        # If beta=10 and max_sum ~ 50, 10*50=500 -> overflow.
        # So we must be careful.
        # Alternative: just use sum of squares for GD, then switch to max for SA.
        # The task asks to minimize max. Sum of squares is a good surrogate.
        pass

    # Strategy:
    # Phase 1: Continuous Gradient Descent on x in R^n to minimize ||M x||^2 subject to x_i ~ +/- 1.
    # We don't enforce constraint strictly during GD, but we project at the end or use a soft constraint.
    # Actually, minimizing ||M x||^2 with x free leads to x=0. 
    # We want x in {-1, 1}^n.
    # We can minimize L(x) = 0.5 * ||M x||^2 + lambda * sum(x_i^2 - 1)^2
    # This is a non-convex optimization.
    # Let's use Gradient Descent on the "energy" E(x) = sum_p (sum_i M[i,p] x_i)^2
    # Gradient wrt x_i is 2 * M[i,:] @ (M x)
    
    rng = np.random.RandomState(42)
    
    best_signs = None
    best_score = float('inf')
    
    # We will run multiple restarts.
    # In each restart:
    # 1. Initialize x randomly in [-1, 1] or {-1, 1}
    # 2. Run Gradient Descent steps on the continuous relaxation, projecting to hypercube or just letting it flow.
    #    Actually, a common trick for Ising-like models is to use the "continuous spin" model.
    #    Let x be continuous. Minimize E(x) = ||M x||^2.
    #    But we need x_i = +/- 1.
    #    We can minimize E(x) + mu * sum((x_i^2 - 1)^2).
    #    Or simply: Start with random +/- 1. 
    #    Compute gradient of the "penalty" for being far from +/-1? No.
    
    # Let's try a different angle:
    # The objective is max_p |sum_i M[i,p] s_i|.
    # This is equivalent to finding s in {-1,1}^n minimizing the L-infinity norm of M s.
    
    # Approximation 1: Minimize L2 norm (sum of squares).
    # Approximation 2: Minimize "Soft-L1" or "Soft-Linf".
    
    # Let's use Gradient Descent on the L2 loss, but with a "hard" projection step?
    # No, projection to {-1,1} is just sign(x).
    # If we do: x_{t+1} = x_t - lr * grad(L2(x_t)), then s = sign(x_{t+1}).
    # This is basically a form of iterative improvement.
    
    # Let's implement a hybrid:
    # 1. Random init s.
    # 2. Calculate sums = M s.
    # 3. "Gradient" of max|sums| is not differentiable, but gradient of sum(sum^2) is 2 * M^T (M s).
    # 4. Update continuous x = x - lr * 2 * M^T (M s).
    # 5. Project s = sign(x).
    # 6. Repeat.
    
    # This is essentially "Gradient Descent with Projection".
    
    def run_gd_projection(x_init, steps=1000, lr=0.01):
        x = x_init.copy()
        s = np.sign(x)
        # Ensure no zeros in s
        s[s == 0] = 1
        
        for step in range(steps):
            if time.time() > deadline - 5:
                break
            
            sums = s @ M  # Np
            # Gradient of 0.5 * ||M s||^2 wrt s is M^T (M s) = M_T @ sums
            grad = M_T @ sums  # n
            
            # Update x
            x = x - lr * grad
            
            # Project to signs
            s_new = np.sign(x)
            s_new[s_new == 0] = 1
            
            # Check if s changed
            # If s_new == s, we might be stuck or converged.
            # To escape, we can add noise to x or s.
            
            # Evaluate new s
            if not np.all(s_new == s):
                sums_new = s_new @ M
                # We track max abs sum for the discrete solution
                # But GD minimizes L2. We need to check if L2 reduction helps Max.
                # Usually minimizing L2 helps Max, but not always.
                # We will accept if it improves the actual Max Abs Sum, or probabilistically if it improves L2.
                
                # For speed, we might just take the sign(x) as the candidate.
                # Let's compute the score for s_new
                # This is expensive if done every step. 
                # M is 256 x ~10k. Matmul is fast.
                
                s = s_new
            
        return s

    # Multi-restart loop
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Initialize
        if r == 0:
            # Start with a somewhat structured init? Or random.
            x = rng.randn(n).astype(np.float32)
        else:
            x = rng.randn(n).astype(np.float32)
        
        # Run GD
        # We need to tune lr. 
        # The eigenvalues of M^T M determine stability.
        # M is sparse-ish but dense in storage.
        # Let's use a small lr.
        s = run_gd_projection(x, steps=500, lr=0.05)
        
        # Evaluate
        sums = s @ M
        score = max_abs_sum(sums)
        
        if score < best_score:
            best_score = score
            best_signs = s.copy()
        
        # Phase 2: Local Search (1-flip) to polish the GD result
        # This is crucial because GD minimizes L2, not L-inf.
        # The L-inf minimizer is likely a local minimum of L2, but not necessarily the global L-inf min.
        # 1-flip search on the L-inf objective.
        
        if time.time() > deadline - 5:
            break
            
        current_signs = s.copy()
        current_sums = current_signs @ M
        current_max = max_abs_sum(current_sums)
        
        # Greedy 1-flip
        improved = True
        while improved:
            improved = False
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                
                # Flip i
                old_sign = current_signs[i]
                new_sign = -old_sign
                current_signs[i] = new_sign
                
                # Update sums efficiently
                # delta = 2 * new_sign * M[i, :]
                delta_vec = 2.0 * new_sign * M[i]
                new_sums = current_sums + delta_vec
                new_max = max_abs_sum(new_sums)
                
                if new_max < current_max:
                    current_sums = new_sums
                    current_max = new_max
                    improved = True
                    if current_max < best_score:
                        best_score = current_max
                        best_signs = current_signs.copy()
                else:
                    # Revert
                    current_signs[i] = old_sign
                    # current_sums is unchanged in memory since we didn't assign it back?
                    # Wait, new_sums was a new array. current_sums is untouched.
                    # So no need to revert sums, just signs.
            
            if time.time() > deadline - 5:
                break
        
        # Phase 3: Simulated Annealing on the polished result to escape local minima
        if time.time() > deadline - 5:
            break
            
        T = 2.0
        cooling = 0.99
        sa_signs = current_signs.copy()
        sa_sums = current_sums.copy()
        sa_max = current_max
        
        # Run SA for a few steps
        steps_sa = 1000
        for _ in range(steps_sa):
            if time.time() > deadline - 5:
                break
            if T < 0.1:
                break
                
            i = rng.randint(n)
            old_sign = sa_signs[i]
            new_sign = -old_sign
            
            delta_vec = 2.0 * new_sign * M[i]
            new_sums = sa_sums + delta_vec
            new_max = max_abs_sum(new_sums)
            
            delta = new_max - sa_max
            
            if delta < 0 or (rng.random_sample() < np.exp(-delta / T)):
                sa_signs[i] = new_sign
                sa_sums = new_sums
                sa_max = new_max
                if sa_max < best_score:
                    best_score = sa_max
                    best_signs = sa_signs.copy()
            
            T *= cooling

    if best_signs is None:
        # Fallback
        return [1] * n
        
    return [int(x) for x in best_signs]
