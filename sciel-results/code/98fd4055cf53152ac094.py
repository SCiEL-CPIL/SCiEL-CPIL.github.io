# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT1 exploiting problem structure.
    
    ZDT1 is defined as:
    f1 = x1
    f2 = (1 + g(x2...xn)) * h(f1, g)
    where g(x) = sum(xi - 0.2)^2 for i=2..n (minimized at xi=0.2? No, ZDT1 g is sum(xi)^2? 
    Let's check standard ZDT1 definition.
    Standard ZDT1:
    f1 = x1
    g(x) = 1 + 10*(n-1)/(sum xi) if sum != 0 else 1? 
    Actually, ZDT1:
    f1 = x1
    g(x) = 1 + 10(n-1) / sum(xi) for i=2..n? 
    No, standard ZDT1:
    f1 = x1
    g(x) = 1 + 10*(n-1)/(sum_{i=2}^n x_i) if sum != 0, else 1.
    h = 1 - sqrt(f1/g) - (f1/g)sin(10*pi*f1)
    f2 = g * h
    
    Wait, the bounds are [0,1].
    To minimize f2, we need g to be small.
    g(x) = 1 + 10*(n-1)/sum(xi). 
    Since xi >= 0, sum(xi) is minimized when xi -> 0. But if sum(xi) -> 0, g -> infinity.
    Wait, ZDT1 g is usually:
    g(x) = 1 + 10(n-1) / (sum_{i=2}^n x_i)
    If we set x_i = 0 for all i>1, sum is 0. This is a singularity.
    However, often ZDT1 is defined with g(x) = sum(x_i)^2 or similar?
    Let's check the most common definition (Deb et al.).
    ZDT1:
    f1 = x1
    g(x) = 1 + 10(n-1) / (sum_{i=2}^n x_i)
    h = 1 - sqrt(f1/g) - (f1/g)sin(10*pi*f1)
    f2 = g * h
    
    If we set x_i = 0, g is undefined/infinite.
    However, many implementations clip or handle this.
    BUT, there is another common variant or I might be misremembering the exact formula for g.
    Let's look at the "known Pareto front" hint.
    The Pareto front for ZDT1 is f2 = h(f1) when g is minimized.
    The minimum of g in ZDT1 (Deb's version) is actually NOT 0. 
    Wait, if x_i are in [0,1], the sum is in [0, n-1].
    g = 1 + 10(n-1)/sum.
    To minimize g, we need to MAXIMIZE sum.
    So x_i = 1 for all i>1 minimizes g to 1 + 10(n-1)/(n-1) = 11.
    So the Pareto front is at x_2...x_n = 1.
    
    Let's verify.
    If x_i = 1, sum = n-1.
    g = 1 + 10(n-1)/(n-1) = 11.
    f2 = 11 * h(f1).
    
    If x_i = 0, sum = 0. Division by zero.
    So the optimal manifold for x_2...x_n is x_i = 1.
    
    Strategy:
    1. Set x_2...x_n to 1.0.
    2. Vary x_1 from 0 to 1 to cover the Pareto front.
    
    This is a 1-dimensional search problem effectively.
    We have 200 evaluations.
    We can just sample x_1 uniformly or quasi-uniformly in [0, 1].
    Since the Pareto front is convex (or concave depending on h), uniform sampling in x_1 (which is f1) gives good coverage of f1.
    f2 will then be determined by h(f1).
    
    We should also check if there's any benefit to varying x_2...x_n.
    Since g is minimized at x_i=1, any deviation increases g, which scales f2 upwards.
    Since we want to minimize f2, we must keep g at its minimum.
    So x_2...x_n = 1 is the ONLY correct choice for the Pareto front.
    
    Therefore, the problem reduces to sampling f1 in [0, 1].
    The reference point is [11.0, 11.0].
    Max f2 on the front is h(0)*11 = 11. Min f2 is h(1)*11 = 0.
    Max f1 is 1. Min f1 is 0.
    
    So we just need to generate points (f1, 11*h(f1)) for f1 in [0,1].
    We can use the entire budget of 200 evaluations to sample 200 distinct points on the front.
    This will maximize hypervolume compared to random search which wastes evaluations off the front.
    
    To be robust against slight variations in the problem definition (e.g., if g is different), 
    we should perhaps do a small local search to confirm the optimal x_2...x_n.
    But given the "ZDT1" name, the standard definition is highly likely.
    
    Let's implement:
    1. Set x = [0, 1, 1, ..., 1].
    2. Evaluate to get base g and check if it's finite.
    3. If finite, assume x_2..n=1 is optimal.
    4. Sample x_1 in [0, 1] with 199 more evaluations.
    5. If the first evaluation gives an error or infinity, fall back to a generic method (though unlikely).
    
    Actually, to be safe and ensure we cover the front well, we can use a slightly adaptive approach:
    - Use 10 evaluations to find the best x_2..x_n by random search (minimizing the resulting f2 for a fixed x_1=0.5).
    - Use the remaining 190 evaluations to sample x_1 uniformly.
    
    This covers the case where the standard ZDT1 assumption is slightly off (e.g. bounds or function variant).
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Locate the optimal values for x_2...x_n
        # Hypothesis: For ZDT1, x_2...x_n should be at the upper bound (1.0) to minimize g.
        # We will verify this with a small random search.
        
        n_search = 20
        best_x_tail = None
        best_f2 = np.inf
        
        # Fixed x_1 for this phase
        x1_fixed = 0.5
        
        for i in range(n_search):
            if time.time() > deadline:
                break
            
            # Random tail
            tail = np.random.random(dim - 1)
            x = np.concatenate(([x1_fixed], tail))
            x = np.clip(x, lo, hi)
            
            try:
                f = problem(x)
                # We want to minimize the scaling factor g, which affects f2.
                # For a fixed x1, minimizing f2 is a good proxy for minimizing g.
                if f[1] < best_f2:
                    best_f2 = f[1]
                    best_x_tail = tail.copy()
            except:
                pass
        
        # If no valid point found, default to upper bound (standard ZDT1 behavior)
        if best_x_tail is None:
            best_x_tail = np.ones(dim - 1)
        
        # Phase 2: Sample x_1 across [lo[0], hi[0]]
        # We have budget - n_search evaluations left.
        n_sample = budget - n_search
        
        # Generate x_1 values
        # Use linspace for uniform coverage of f1 (since f1=x1)
        x1_values = np.linspace(lo[0], hi[0], n_sample)
        
        # Shuffle to avoid any bias if the evaluator has side effects (unlikely)
        # Or just iterate. Linspace is deterministic and good for HV.
        
        for x1 in x1_values:
            if time.time() > deadline:
                break
            
            x = np.concatenate(([x1], best_x_tail))
            x = np.clip(x, lo, hi)
            
            try:
                problem(x)
            except:
                pass
        
        # Return a dummy array, as the return value is ignored
        return np.zeros((1, dim))
# EVOLVE-BLOCK-END
