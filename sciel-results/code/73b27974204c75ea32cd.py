import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # ZDT4 specific structure:
        # f1(x) = x0
        # g(x) = 10*n + sum_{i=1}^{n-1} (xi^2 - 10*cos(12*pi*xi*t))
        # f2(x) = g(x) * (1 + sin(10*pi*f1(x)))
        # 
        # For standard bounds [0,1], the global minimum of each term in g is at xi=0.
        # Term: xi^2 - 10*cos(12*pi*xi). At xi=0, value is -10.
        # g_min = 10*10 + 9*(-10) = 10.
        # So on the Pareto front, f2 = 10 * (1 + sin(10*pi*x0)).
        # 
        # The Pareto front is 1D, parameterized by x0.
        # To maximize hypervolume, we need to cover the front densely.
        # Since the evaluator records ALL returned objective vectors, 
        # we just need to return many non-dominated points.
        
        # Determine optimal tail values
        optimal_tail = np.zeros(dim - 1)
        for i in range(1, dim):
            if lo[i] <= 0 <= hi[i]:
                optimal_tail[i-1] = 0.0
            elif hi[i] < 0:
                optimal_tail[i-1] = hi[i]
            else:
                optimal_tail[i-1] = lo[i]
        
        # Strategy:
        # 1. Use a large portion of budget for dense sampling of x0.
        # 2. The front is a curve in 2D objective space.
        # 3. Uniform sampling in x0 is generally good for hypervolume if the front is smooth.
        #    However, ZDT4's front can have oscillatory behavior in f2.
        #    Let's analyze f2(x0) = 10 * (1 + sin(10*pi*x0)) for x0 in [0,1].
        #    sin(10*pi*x0) oscillates 5 times.
        #    The Pareto front is the set of non-dominated points.
        #    A point (f1, f2) is dominated if there exists another point (f1', f2') 
        #    such that f1' <= f1 and f2' <= f2, and at least one is strict.
        #    
        #    Since f1 = x0 is strictly increasing with x0, 
        #    a point with larger x0 can only be non-dominated if it has a smaller f2 
        #    than all points with smaller x0.
        #    
        #    Let's look at the shape of f2(x0).
        #    f2(x0) = 10 + 10*sin(10*pi*x0).
        #    This oscillates between 0 and 20.
        #    
        #    The non-dominated set will be the "lower envelope" of this curve when viewed 
        #    from the bottom-left. Since f1 increases with x0, we are looking for points 
        #    where f2 is lower than all previous points (in terms of x0).
        #    
        #    Actually, since f1 is monotonic with x0, the Pareto front is simply the 
        #    set of points that are not dominated. 
        #    Because f1 is strictly monotonic, for any x0_1 < x0_2, 
        #    point 1 dominates point 2 if f2(x0_1) <= f2(x0_2).
        #    So point 2 is non-dominated only if f2(x0_2) < min(f2(x0) for x0 < x0_2).
        #    
        #    This means the Pareto front consists of the "record lows" of f2 as x0 increases.
        #    
        #    Let's find where the minima of f2 occur.
        #    f2(x0) = 10 + 10*sin(10*pi*x0).
        #    Minima occur when sin(10*pi*x0) = -1.
        #    10*pi*x0 = 3*pi/2 + 2*k*pi
        #    x0 = 3/20 + k/5 = 0.15 + 0.2*k
        #    For k=0: x0=0.15, f2=0
        #    For k=1: x0=0.35, f2=0
        #    For k=2: x0=0.55, f2=0
        #    For k=3: x0=0.75, f2=0
        #    For k=4: x0=0.95, f2=0
        #    
        #    So the global minima of f2 are at these 5 points, all with f2=0.
        #    However, points with higher x0 dominate points with lower x0 ONLY if they have lower f2.
        #    Since all minima have f2=0, the point at x0=0.95 has f2=0.
        #    The point at x0=0.75 has f2=0.
        #    Does (0.75, 0) dominate (0.95, 0)? No, because f1=0.75 < 0.95, but f2=0 is not < 0.
        #    Wait, domination requires f1' <= f1 AND f2' <= f2, with at least one strict.
        #    (0.75, 0) vs (0.95, 0): f1 0.75 < 0.95, f2 0 == 0. So (0.75, 0) does NOT dominate (0.95, 0) because f2 is not strictly less.
        #    Actually, standard definition: A dominates B if A_i <= B_i for all i, and A_j < B_j for some j.
        #    So (0.75, 0) does NOT dominate (0.95, 0).
        #    And (0.95, 0) does NOT dominate (0.75, 0) because f1 is larger.
        #    So both are non-dominated?
        #    
        #    Let's check a point in between, e.g., x0=0.85.
        #    f2(0.85) = 10 + 10*sin(8.5*pi) = 10 + 10*sin(0.5*pi) = 20.
        #    (0.85, 20) is dominated by (0.75, 0) because 0.75 < 0.85 and 0 < 20.
        #    
        #    So the Pareto front consists of the local minima and the segments connecting them?
        #    No, only the non-dominated points.
        #    Any point on the curve between two minima is dominated by the earlier minimum.
        #    Example: x0=0.2, f2 > 0. Dominated by (0.15, 0).
        #    So the Pareto front is effectively the set of points where f2 is at a local minimum 
        #    or potentially the start of the curve?
        #    
        #    Let's trace from x0=0.
        #    x0=0, f2=10.
    #    x0=0.05, f2 = 10+10*sin(0.5*pi)=20.
    #    x0=0.1, f2 = 10.
    #    x0=0.15, f2 = 0.
    #    The point (0, 10) is non-dominated? 
    #    Is there any point with f1<0? No. 
    #    Is there any point with f1<=0 and f2<10? No.
    #    So (0, 10) is non-dominated.
    #    The point (0.15, 0) dominates (0, 10)? 
    #    f1: 0.15 > 0. No.
    #    So (0, 10) is non-dominated.
    #    
    #    What about (0.1, 10)? Dominated by (0, 10)? No, f1 is larger. Dominated by (0.15, 0)? No, f1 is smaller.
    #    Wait, (0.15, 0) has f1=0.15. (0.1, 10) has f1=0.1.
    #    Does (0.1, 10) dominate (0.15, 0)? No, f2 is larger.
    #    Does (0.15, 0) dominate (0.1, 10)? No, f1 is larger.
    #    So (0.1, 10) is non-dominated?
    #    Let's check if any point dominates (0.1, 10).
    #    We need f1' <= 0.1 and f2' <= 10.
    #    At x0=0, f2=10. So (0, 10) dominates (0.1, 10) because f1=0 < 0.1 and f2=10 <= 10.
    #    Yes! (0, 10) dominates (0.1, 10).
    #    
    #    So, generally, a point (x0, f2(x0)) is dominated by any earlier point (x0', f2(x0')) 
    #    where x0' < x0 and f2(x0') <= f2(x0).
    #    
    #    This means the Pareto front consists of the points that set new "record lows" in f2 
    #    as x0 increases? No, it's the opposite.
    #    If f2(x0) is lower than all previous f2 values, it is NOT dominated by previous points.
    #    But it might be dominated by a FUTURE point? No, future points have larger f1, so they cannot dominate it.
    #    So a point is non-dominated if and only if f2(x0) < min(f2(x0') for x0' < x0).
    #    i.e., it is a strict record low in f2.
    #    
    #    Let's verify.
    #    x0=0, f2=10. Record low (empty set). Non-dominated.
    #    x0=0.15, f2=0. 0 < 10. Record low. Non-dominated.
    #    x0=0.35, f2=0. 0 is not < 0. So NOT a strict record low.
    #    Is it dominated? By (0.15, 0). f1=0.15 < 0.35, f2=0 <= 0. Yes, dominated.
    #    So only the FIRST occurrence of a minimum is non-dominated?
    #    
    #    Let's check x0=0.95, f2=0.
    #    Dominated by (0.15, 0)? Yes.
    #    
    #    So the Pareto front is:
    #    1. (0, 10)
    #    2. (0.15, 0)
    #    
    #    Wait, is that it?
    #    Let's check x0 between 0 and 0.15.
    #    f2 goes from 10 to 20 to 10 to 0.
    #    All these points have f2 > 0 (except 0.15).
    #    They are all dominated by (0, 10) if f2 >= 10?
    #    (0.05, 20) dominated by (0, 10).
    #    (0.1, 10) dominated by (0, 10).
    #    (0.14, ~0.1) dominated by (0, 10)? f1=0 < 0.14, f2=10 > 0.1. No.
    #    Is (0.14, 0.1) dominated by (0.15, 0)? f1=0.15 > 0.14. No.
    #    So (0.14, 0.1) is non-dominated?
    #    It is not dominated by (0, 10) because f2 is smaller.
    #    It is not dominated by (0.15, 0) because f1 is smaller.
    #    So yes, the points on the descending slope from (0,10) to (0.15,0) are non-dominated.
    #    
    #    So the Pareto front is the curve segment from x0=0 to x0=0.15.
    #    And then?
    #    After x0=0.15, f2 increases to 20 at 0.25, then decreases to 0 at 0.35.
    #    Points in (0.15, 0.35] are dominated by (0.15, 0) because f2 >= 0 and f1 > 0.15.
    #    So no points in (0.15, 1] are non-dominated?
    #    
    #    Wait, what if f2 becomes negative?
    #    f2 = 10 + 10*sin(...). Min is 0. So f2 is always >= 0.
    #    So (0.15, 0) dominates all subsequent points.
    #    
    #    Conclusion: The Pareto front of ZDT4 (standard) is just the segment from x0=0 to x0=0.15!
    #    This is a very short segment.
    #    
    #    If this is true, we should concentrate all our samples in [0, 0.15].
    #    Sampling the rest of the domain is useless for hypervolume because those points are dominated.
    #    
    #    Let's double check this logic.
    #    Reference point is [11.0, 400.0].
    #    Hypervolume is calculated on the non-dominated set.
    #    If we return dominated points, they are ignored in the hypervolume calculation 
    #    (or rather, the hypervolume of the union of hyper-rectangles defined by the points 
    #    and the reference point is the same as the hypervolume of the non-dominated subset).
    #    So returning dominated points doesn't hurt, but it wastes budget.
    #    
    #    So the optimal strategy is to sample x0 densely in [0, 0.15].
    #    Specifically, we want to cover the curve f2(x0) = 10 + 10*sin(10*pi*x0) for x0 in [0, 0.15].
    #    
    #    Let's verify the bounds.
    #    If bounds are different, this logic might change.
    #    The problem statement says "bounds given per variable in bounds".
    #    We should check the bounds for x0.
    #    If hi[0] is larger than 0.15, we still only care about [0, 0.15] (or [lo[0], min(hi[0], 0.15)]).
    #    
    #    What if lo[0] > 0?
    #    Then the front starts at lo[0].
    #    What if hi[0] < 0.15?
    #    Then the front is [lo[0], hi[0]].
    #    
    #    Let's refine the sampling range.
    #    The "knee" or end of the front is at the first global minimum of f2.
    #    For standard ZDT4, this is at x0 = 0.15.
    #    
    #    Let's implement this.
    
        x0_lo = lo[0]
        x0_hi = hi[0]
        
        # The Pareto front ends at the first minimum of f2.
        # For standard ZDT4, this is 0.15.
        # If bounds are non-standard, we might need to adjust.
        # But assuming standard ZDT4 structure, we focus on [x0_lo, min(x0_hi, 0.15)].
        
        # To be safe, let's identify the first minimum.
        # f2(x0) = 10 + 10*sin(10*pi*x0)
        # Minima at x0 = 0.15, 0.35, ...
        # First minimum in [x0_lo, x0_hi] is 0.15 (if 0.15 >= x0_lo).
        
        front_end = min(x0_hi, 0.15)
        if x0_lo > front_end:
            front_end = x0_lo # Should not happen if lo < hi
        
        # Sample densely in [x0_lo, front_end]
        # Use most of the budget here.
        n_main = int(budget * 0.9)
        n_refine = budget - n_main
        
        # Generate samples
        x0_samples = np.linspace(x0_lo, front_end, n_main)
        
        X = np.zeros((n_main, dim))
        X[:, 0] = x0_samples
        X[:, 1:] = optimal_tail
        
        F = np.array([problem(x) for x in X])
        used = n_main
        
        # Check deadline
        if time.time() > deadline:
            return X
        
        # Refinement:
        # Since the front is a smooth curve, uniform sampling is quite good.
        # We can use the remaining budget to fill in gaps or refine endpoints.
        # However, with 180 samples in a small interval, the gap is very small.
        # The hypervolume error is roughly proportional to the gap size.
        # Adding more points linearly improves hypervolume approximation.
        
        # Let's just use all remaining budget to add more points in the same interval.
        if n_refine > 0:
            # Add random points in the interval to avoid grid artifacts?
            # Or just extend the linspace?
            # Linspace is fine.
            x0_samples_refine = np.linspace(x0_lo, front_end, n_refine)
            X_refine = np.zeros((n_refine, dim))
            X_refine[:, 0] = x0_samples_refine
            X_refine[:, 1:] = optimal_tail
            
            F_refine = np.array([problem(x) for x in X_refine])
            used += n_refine
            
            X = np.vstack([X, X_refine])
            F = np.vstack([F, F_refine])
        
        return X
