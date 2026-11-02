import numpy as np
from scipy.optimize import minimize

N = 600

def construct():
    # Initial guess: Uniform
    h0 = np.ones(N)
    
    def objective_and_grad(y):
        # y is the raw vector, we normalize it to L2=1
        h = np.abs(y) # Ensure non-negative? Or use bounds.
        # Using bounds is better for L-BFGS-B.
        # But L-BFGS-B doesn't support equality constraints.
        # We will minimize f(h) = smooth_max(conv(h,h)) / ||h||^2
        # and rely on the scale invariance.
        
        h = np.maximum(y, 0)
        norm_sq = np.sum(h**2)
        if norm_sq < 1e-12:
            return 1e6, np.zeros_like(y)
        
        ac = np.convolve(h, h, mode='full')
        # Smooth max
        T = 0.05
        # Shift to avoid overflow
        ac_shifted = ac - np.max(ac)
        log_sum_exp = np.log(np.sum(np.exp(ac_shifted / T))) + T * np.max(ac)
        obj = log_sum_exp / norm_sq
        
        # Gradient
        # d obj / d h
        # Let S = sum exp((ac_k - M)/T)
        # d log_sum_exp / d h_j = sum_k (1/T) exp((ac_k - M)/T) * d ac_k / d h_j
        # d ac_k / d h_j = h_{k-j} + h_{j-k} (with bounds)
        
        # This gradient calculation is expensive if done naively.
        # We can approximate or use a simpler method.
        # For the sake of the "new approach", let's just use the objective and let L-BFGS-B approximate the gradient?
        # No, providing gradient is better.
        
        # Let's use a finite difference or a simplified gradient.
        # Actually, for the "new program" requirement, using a black-box optimizer with a smoothed objective is a valid distinct approach.
        # We can return just the objective and let L-BFGS-B use numerical gradients?
        # That might be slow (N=600, 2N function evals per step).
        # 120 seconds limit.
        
        # Let's stick to the analytic gradient but implement it efficiently.
        # ac_k = sum_i h_i h_{k-i}
        # grad_j = sum_k w_k (h_{k-j} + h_{j-k})
        # where w_k = (1/T) exp((ac_k - M)/T) / S * (1/norm_sq - 2 obj * h_j / norm_sq^2 ? No, obj is scale invariant)
        
        # Simplified: Minimize smooth_max(ac). The scale is handled by normalization.
        # Let's minimize smooth_max(ac) subject to ||h||=1 is hard.
        # Let's minimize smooth_max(ac) / ||h||^2.
        
        # Due to complexity of exact gradient with normalization, let's use a simpler "new" approach:
        # **Alternating Minimization**
        # 1. Fix h, find best shift? No.
        # 2. **Simulated Annealing**
        
        pass

    # Given the complexity and time limit, a robust "new" approach is 
    # **Genetic Algorithm** with NumPy vectorization.
    
    # Pop size 100, Generations 50.
    # 100 * 50 * 600 operations = 3M. Fast.
    
    pop_size = 100
    generations = 100
    
    # Initialize population
    # Mix of uniform, gaussian, random
    pop = np.zeros((pop_size, N))
    pop[0] = np.ones(N)
    x = np.linspace(-0.5, 0.5, N)
    pop[1] = np.exp(-x**2)
    pop[2:] = np.random.rand(pop_size-2, N)
    
    # Normalize
    norms = np.linalg.norm(pop, axis=1, keepdims=True)
    pop = pop / norms
    
    def score(h):
        ac = np.convolve(h, h, mode='full')
        return np.max(ac)
    
    best_h = pop[0].copy()
    best_score = score(best_h)
    
    for g in range(generations):
        scores = np.array([score(h) for h in pop])
        best_idx = np.argmin(scores)
        if scores[best_idx] < best_score:
            best_score = scores[best_idx]
            best_h = pop[best_idx].copy()
        
        # Selection: Keep top 20%
        k = pop_size // 5
        elite = pop[np.argsort(scores)[:k]]
        
        # Crossover and Mutation for the rest
        new_pop = np.zeros_like(pop)
        new_pop[:k] = elite
        
        for i in range(k, pop_size):
            # Pick two parents
            p1_idx = np.random.randint(0, k)
            p2_idx = np.random.randint(0, k)
            p1 = elite[p1_idx]
            p2 = elite[p2_idx]
            
            # Uniform crossover
            mask = np.random.rand(N) > 0.5
            child = np.where(mask, p1, p2)
            
            # Mutation: Add Gaussian noise
            noise = np.random.randn(N) * 0.1
            child = child + noise
            child = np.maximum(child, 0)
            
            norm = np.linalg.norm(child)
            if norm > 1e-12:
                child = child / norm
            else:
                child = np.random.rand(N)
                child = child / np.linalg.norm(child)
                
            new_pop[i] = child
            
        pop = new_pop
        
    return list(best_h)
