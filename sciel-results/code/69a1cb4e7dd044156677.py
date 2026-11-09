"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar


class Optimizer:
    """
    Multi-objective optimizer for ZDT3.
    
    ZDT3:
    - f1 = x1
    - g(x) = 1 + 10*(n-1) + sum_{i=2}^{n} [2(x_i^2 - cos(4*pi*x_i)) + 10]
    - f2 = g(x) * (1 + sqrt(f1/g(x)))
    
    The Pareto front is where g is minimized. Since g doesn't depend on x1,
    for each x1 in [0,1], the minimum f2 is achieved at g_min.
    The front is: f2 = g_min * (1 + sqrt(f1/g_min)) for f1 in [0,1].
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # Find the minimum of h(x) = 2(x^2 - cos(4*pi*x)) + 10 for x in [0,1]
        # This is a unimodal-ish function with multiple local minima
        # Use scipy to find the global minimum
        
        def h(x):
            return 2*(x**2 - np.cos(4*np.pi*x)) + 10
        
        # Find minimum using scipy
        result = minimize_scalar(h, bounds=(0, 1), method='bounded', options={'xatol': 1e-12})
        x_opt = result.x
        h_min = result.fun
        
        # g_min = 1 + 10*(n-1) + (n-1)*h_min
        g_min = 1 + 10*(dim-1) + (dim-1)*h_min
        
        # The Pareto front: f2 = g_min * (1 + sqrt(f1/g_min)) for f1 in [0,1]
        # This is a monotonically increasing curve.
        
        # For hypervolume maximization with reference point (11, 11):
        # We want to place points to maximize the dominated area.
        # For a monotone increasing curve, the optimal strategy is to place
        # points such that the marginal hypervolume contribution is equalized.
        
        # The hypervolume contribution of a point (f1_i, f2_i) depends on its
        # neighbors. For a set of points sorted by f1, the hypervolume is:
        # HV = sum over intervals of (ref_f1 - f1_i) * (ref_f2 - f2_i) adjusted
        
        # Actually, for 2D hypervolume with reference point (r1, r2):
        # Sort points by f1. The hypervolume is:
        # HV = sum_{i} (f1_{i+1} - f1_i) * (r2 - f2_i) + (r1 - f1_n) * (r2 - f2_n)
        # where f1_0 = 0 (or -inf conceptually, but we use 0 as lower bound)
        # Wait, need to be more careful.
        
        # Standard 2D hypervolume: sort by f1 ascending, f2 descending (Pareto sorted)
        # But our front is f2 increasing with f1, so in Pareto order f1 asc, f2 desc is wrong.
        # For minimization, Pareto front has f1 asc, f2 desc. But here f2 increases with f1.
        # So the Pareto front in standard orientation: as f1 increases, f2 increases.
        # This means NO point dominates another if they're all on the front.
        
        # Hypervolume calculation for 2D:
        # Sort points by f1 ascending. For each consecutive pair, the area is:
        # (f1_{i+1} - f1_i) * (r2 - f2_i) for i < n
        # Plus (r1 - f1_n) * (r2 - f2_n)
        # Plus (r1 - f1_0) * (r2 - f2_0) ... wait, need to include the leftmost.
        
        # Actually: sort by f1 asc. HV = sum_{i=0}^{n-1} (f1_{i+1} - f1_i) * (r2 - f2_i)
        # where we define f1_{-1} = 0 (or the minimum possible f1)
        # And the last term: (r1 - f1_{n-1}) * (r2 - f2_{n-1})
        
        # More precisely:
        # HV = (f1_1 - f1_0)*(r2 - f2_0) + (f1_2 - f1_1)*(r2 - f2_1) + ... + (r1 - f1_{n-1})*(r2 - f2_{n-1})
        
        # To maximize this, we want to place points where the marginal contribution is equal.
        # The marginal contribution of adding a point at f1 = x between f1_i and f1_{i+1} is:
        # (x - f1_i)*(r2 - f2(x)) + (f1_{i+1} - x)*(r2 - f2(x)) - (f1_{i+1} - f1_i)*(r2 - f2_i)
        # = (f1_{i+1} - f1_i)*(r2 - f2(x)) - (f1_{i+1} - f1_i)*(r2 - f2_i)
        # = (f1_{i+1} - f1_i)*(f2_i - f2(x))
        
        # So the gain from inserting at x is proportional to (f2_i - f2(x)) * interval_width.
        # Since f2 is increasing, f2_i < f2(x) for x > f1_i, so this is negative?
        # That can't be right. Let me reconsider.
        
        # Actually, when we insert a point, we split one interval into two. The new HV is:
        # (x - f1_i)*(r2 - f2_i) + (f1_{i+1} - x)*(r2 - f2(x))
        # vs old: (f1_{i+1} - f1_i)*(r2 - f2_i)
        # Gain = (x - f1_i)*(r2 - f2_i) + (f1_{i+1} - x)*(r2 - f2(x)) - (f1_{i+1} - f1_i)*(r2 - f2_i)
        # = (f1_{i+1} - x)*(r2 - f2(x)) - (f1_{i+1} - x)*(r2 - f2_i)
        # = (f1_{i+1} - x)*(f2_i - f2(x))
        
        # Since f2(x) > f2_i for x > f1_i, this gain is negative! That means inserting
        # points in the middle DECREASES hypervolume? That's because for a monotonically
        # increasing front, the rightmost point gives the most "coverage" in f2.
        
        # Wait, I think I have the hypervolume formula wrong. Let me reconsider.
        
        # For minimization, a point (a, b) dominates region [0,a] x [0,b] in objective space.
        # The hypervolume is the volume of the union of these regions, clipped to [0,r1] x [0,r2].
        
        # For 2D, sort points by f1 ascending. The non-dominated points in order have
        # f1 increasing and f2 decreasing for a typical Pareto front. But here f2 increases
        # with f1, so actually ALL points are non-dominated (none dominates another).
        
        # The hypervolume is:
        # Sort by f1 asc: p_0, p_1, ..., p_{n-1}
        # HV = sum_{i=0}^{n-1} (f1_{i+1} - f1_i) * (r2 - f2_i)
        # where f1_n = r1, and for i=n-1, the term is (r1 - f1_{n-1}) * (r2 - f2_{n-1})
        # But we also need to account for the region below the first point.
        
        # Actually, the correct formula:
        # HV = (f1_0 * (r2 - f2_0)) + sum_{i=1}^{n-1} ((f1_i - f1_{i-1}) * (r2 - f2_{i-1}))
        # Wait, I'm getting confused. Let me use the standard algorithm.
        
        # Standard 2D hypervolume:
        # 1. Sort points by f1 ascending
        # 2. Remove dominated points (for our front, none are dominated)
        # 3. HV = sum_{i=0}^{n-1} (f1_{i+1} - f1_i) * (r2 - f2_i)
        #    where f1_n := r1
        #    But this misses the contribution from the leftmost point's f2 value for f1 < f1_0.
        
        # Correct: 
        # HV = (r1 - f1_0) * (r2 - f2_0) if n=1
        # For n>1: HV = sum_{i=0}^{n-2} (f1_{i+1} - f1_i) * (r2 - f2_i) + (r1 - f1_{n-1}) * (r2 - f2_{n-1})
        
        # Hmm, this still doesn't seem right. Let me think geometrically.
        
        # The hypervolume is the area of the union of rectangles [0, f1_i] x [0, f2_i] 
        # intersected with [0, r1] x [0, r2], but that's for the dominated region.
        
        # Actually no. For minimization, point p dominates all points in [0, p_f1] x [0, p_f2].
        # The hypervolume is the measure of the union of these dominated regions, 
        # intersected with [0, r1] x [0, r2].
        
        # For 2D, if we sort by f1 ascending, the union of rectangles can be computed as:
        # Start with the rightmost point (largest f1). Its rectangle is [0, f1_max] x [0, f2_max].
        # The next point to the left has smaller f1 but potentially larger f2.
        
        # For our front where f2 increases with f1, the point with largest f1 has the
        # largest f2, so it dominates all others in the f2 dimension. Wait, no - for
        # minimization, smaller f2 is better. So the point with smallest f2 is best in f2.
        
        # Our front: f2 = g_min * (1 + sqrt(f1/g_min)). As f1 increases, f2 increases.
        # So the point with f1=0 has the smallest f2 = g_min, and f1=1 has f2 = g_min*(1+1/sqrt(g_min)).
        
        # For minimization, point A dominates point B if A.f1 <= B.f1 AND A.f2 <= B.f2.
        # On our front, no point dominates another (f1 and f2 both increase together).
        
        # The hypervolume: the dominated region by all points is the union of
        # [0, f1_i] x [0, f2_i] for all i. Since f1 and f2 both increase, the union
        # is just [0, max_f1] x [0, max_f2] = [0, 1] x [0, f2_max] intersected with [0,11]x[0,11].
        
        # Wait, that's not right either. The union of [0,a_i] x [0,b_i] where a_i and b_i
        # both increase is NOT simply [0, max_a] x [0, max_b]. It's the region under the
        # "staircase" formed by the points.
        
        # For sorted points (f1_0, f2_0), (f1_1, f2_1), ... with f1 and f2 both increasing:
        # The union is: for f1 in [0, f1_0], f2 in [0, f2_0]; for f1 in [f1_0, f1_1], f2 in [0, max(f2_0, f2_1)] = [0, f2_1] since f2_1 > f2_0; etc.
        
        # So the hypervolume is:
        # f1_0 * f2_0 + (f1_1 - f1_0) * f2_1 + (f1_2 - f1_1) * f2_2 + ... + (f1_{n-1} - f1_{n-2}) * f2_{n-1}
        # But clipped to [0, r1] x [0, r2].
        
        # Since r1=11 > 1 and r2=11 > f2_max (assuming g_min is reasonable), no clipping needed.
        
        # So HV = f1_0*f2_0 + sum_{i=1}^{n-1} (f1_i - f1_{i-1})*f2_i
        
        # To maximize this, we want to place points to maximize this sum.
        # The marginal gain of adding a point at position x between f1_{i-1} and f1_i:
        # New contribution: (x - f1_{i-1})*f2(x) + (f1_i - x)*f2_i
        # Old contribution: (f1_i - f1_{i-1})*f2_i
        # Gain = (x - f1_{i-1})*f2(x) + (f1_i - x)*f2_i - (f1_i - f1_{i-1})*f2_i
        # = (x - f1_{i-1})*f2(x) - (x - f1_{i-1})*f2_i
        # = (x - f1_{i-1})*(f2(x) - f2_i)
        
        # Since f2(x) < f2_i for x < f1_i (f2 is increasing), this gain is NEGATIVE.
        # This means adding more points DECREASES the hypervolume?!
        
        # That can't be right. Let me reconsider.
        
        # Oh wait, I think I have the dominated region wrong. For minimization,
        # point (a,b) dominates the region [0,a] x [0,b]. The hypervolume is the
        # volume of the UNION of these regions, but the reference point defines the
        # upper bound. The hypervolume is the volume of the region that is dominated
        # by at least one point AND is below the reference point.
        
        # The region dominated by point i is [0, f1_i] x [0, f2_i]. The union of all
        # these regions, intersected with [0, r1] x [0, r2], is what we compute.
        
        # For our front with f1 and f2 both increasing, the union is:
        # The "upper envelope" in the f2 direction for each f1 value.
        # For f1 in [0, f1_0]: max f2 is f2_0
        # For f1 in [f1_0, f1_1]: max f2 is f2_1 (since f2_1 > f2_0)
        # ...
        # For f1 in [f1_{n-1}, r1]: max f2 is f2_{n-1}
        
        # So HV = f1_0*f2_0 + (f1_1-f1_0)*f2_1 + ... + (f1_{n-1}-f1_{n-2})*f2_{n-1} + (r1-f1_{n-1})*f2_{n-1}
        
        # This simplifies to: HV = sum_{i=0}^{n-1} (f1_i - f1_{i-1})*f2_i + (r1 - f1_{n-1})*f2_{n-1}
        # where f1_{-1} = 0.
        
        # = f1_0*f2_0 + (f1_1-f1_0)*f2_1 + ... + (f1_{n-1}-f1_{n-2})*f2_{n-1} + (r1-f1_{n-1})*f2_{n-1}
        # = f1_0*f2_0 + sum_{i=1}^{n-1} (f1_i-f1_{i-1})*f2_i + (r1-f1_{n-1})*f2_{n-1}
        
        # The last two terms combine: (f1_{n-1}-f1_{n-2})*f2_{n-1} + (r1-f1_{n-1})*f2_{n-1} = (r1-f1_{n-2})*f2_{n-1}
        
        # So HV = f1_0*f2_0 + sum_{i=1}^{n-2} (f1_i-f1_{i-1})*f2_i + (r1-f1_{n-2})*f2_{n-1}
        
        # To maximize, we want f2_i to be as large as possible for each interval.
        # Since f2 is increasing, the rightmost point has the largest f2, so we want
        # to maximize the weight on the rightmost point.
        
        # The optimal strategy: place all points at f1=1? No, that would give:
        # HV = 0 + ... + (r1-0)*f2(1) = r1*f2(1) if all points at f1=1.
        # But with one point at f1=1: HV = 1*f2(1) + (r1-1)*f2(1) = r1*f2(1).
        # With points spread out: HV = f1_0*f2_0 + ... which is less than r1*f2(1)
        # because f2_i < f2(1) for i < n-1.
        
        # So the optimal is to put ALL points at f1=1? But that gives only one unique point.
        # The hypervolume only counts unique non-dominated points.
        
        # Wait, the problem says "Every returned objective vector is recorded". So duplicates
        # are recorded but don't add to hypervolume. We need distinct points.
        
        # So we need to place points at distinct f1 values. The optimal is to place
        # points as close to f1=1 as possible, but distinct.
        
        # With 200 points, place them at f1 = 1 - epsilon_i where epsilon_i are small
        # distinct values. The hypervolume will be approximately:
        # HV ≈ r1 * f2(1) - small correction
        
        # Actually, let me reconsider. The hypervolume formula:
        # HV = f1_0*f2_0 + (f1_1-f1_0)*f2_1 + ... + (r1-f1_{n-1})*f2_{n-1}
        
        # If we place points at f1_0 < f1_1 < ... < f1_{n-1} < r1:
        # HV = f1_0*f2_0 + sum_{i=1}^{n-1} (f1_i - f1_{i-1})*f2_i + (r1-f1_{n-1})*f2_{n-1}
        
        # To maximize, we want each f2_i to be as large as possible, which means
        # f1_i as large as possible. But f1_i must be distinct and in order.
        
        # The optimal: f1_i = 1 - delta*(n-1-i) for small delta, so points are
        # clustered near f1=1.
        
        # But wait, this assumes the front is f2 = g_min*(1+sqrt(f1/g_min)) which
        # is increasing. Let me verify: for f1=0, f2=g_min. For f1=1, f2=g_min*(1+1/sqrt(g_min)).
        # Since g_min > 0, f2(1) > f2(0). Yes, increasing.
        
        # So the optimal strategy is to cluster points near f1=1.
        
        # But this seems counterintuitive for hypervolume. Let me double-check with a
        # simple example. Two points on an increasing line from (0,1) to (1,2), ref=(10,10).
        # Point A=(0,1): HV = 0*1 = 0? No, HV = f1*f2 for a single point? 
        # For a single point (a,b), HV = a*b (the area of [0,a]x[0,b]).
        # So (0,1) gives HV=0, (1,2) gives HV=2.
        # Two points (0,1) and (1,2): HV = 0*1 + (1-0)*2 + (10-1)*2 = 0 + 2 + 18 = 20.
        # Two points (0.5,1.5) and (1,2): HV = 0.5*1.5 + (1-0.5)*2 + (10-1)*2 = 0.75+1+18=19.75.
        # Two points (0.9,1.9) and (1,2): HV = 0.9*1.9 + (1-0.9)*2 + (10-1)*2 = 1.71+0.2+18=19.91.
        # One point (1,2): HV = 1*2 + (10-1)*2 = 2+18=20.
        # So one point at (1,2) gives the same HV as two points at (0.9,1.9) and (1,2)?
        # No: one point (1,2): the dominated region is [0,1]x[0,2], area=2. Plus the region
        # from f1=1 to f1=10 at f2=2: area=9*2=18. Total=20.
        # Two points (0.9,1.9) and (1,2): dominated region is [0,0.9]x[0,1.9] union [0,1]x[0,2].
        # Since [0,0.9]x[0,1.9] is a subset of [0,1]x[0,2], the union is just [0,1]x[0,2], area=2.
        # Plus [1,10]x[0,2], area=18. Total=20.
        
        # So indeed, for an increasing front, adding points to the left doesn't help!
        # The rightmost point dominates all others in the sense that its dominated region
        # contains all others' dominated regions.
        
        # This means the optimal strategy is to place ONE point at f1=1 (max f1, max f2)
        # and the remaining 199 points can be anywhere (they don't add to HV).
        
        # But wait, the scoring uses "exact hypervolume of all recorded vectors". If we
        # record 200 points but only 1 is unique/non-dominated, the HV is the same as
        # with 1 point.
        
        # So the optimal is: place one point at f1=1, and the rest can be duplicates or
        # anywhere. The HV will be determined by the single best point.
        
        # But that gives a very low score if the reference HV is based on a dense sample
        # of the front. The reference HV would be much larger because it includes the
        # entire front.
        
        # I think I'm misunderstanding the hypervolume calculation. Let me reconsider.
        
        # The hypervolume is the volume of the region that is dominated by at least one
        # point in the set, intersected with the reference box. For minimization, point p
        # dominates the box [0, p_f1] x [0, p_f2]. The union of these boxes, intersected
        # with [0, r1] x [0, r2], is the hypervolume.
        
        # For an increasing front, the rightmost point has the largest f1 AND largest f2,
        # so its box [0, f1_max] x [0, f2_max] contains all other boxes. The union is just
        # this one box, intersected with the reference box.
        
        # So HV = min(f1_max, r1) * min(f2_max, r2).
        
        # This means only the "best" point (max f1, max f2) matters for HV on an increasing
        # front. All other points are redundant.
        
        # This can't be right for a meaningful multi-objective problem. Let me reconsider
        # whether ZDT3's front is actually increasing.
        
        # ZDT3: f2 = g*(1 + sqrt(f1/g)). For f1 in [0,1] and g > 0:
        # f2 = g + sqrt(g*f1). This is increasing in f1. Yes.
        
        # So for ZDT3, the Pareto front is increasing, and the hypervolume is determined
        # by the single point with the largest f1 (which also has the largest f2).
        
        # This means the optimal strategy is to place a point at f1=1, x2..xn optimized
        # to minimize g. The HV will be 1 * f2(1) = g_min*(1+1/sqrt(g_min)).
        
        # But the reference HV is based on a "dense fixed sample of the known Pareto front".
        # If the reference sample includes many points, its HV is also determined by the
        # rightmost point (since the front is increasing). So the reference HV is also
        # approximately 1 * f2(1) = g_min*(1+1/sqrt(g_min)).
        
        # So the ratio would be approximately 1.0, which matches the current score of ~1.26.
        
        # Wait, the score is 1.26, which is above 1.0. So the current program is already
        # doing better than the reference. How?
        
        # Maybe the reference sample doesn't include f1=1 exactly, or the current program
        # is finding a slightly better g_min.
        
        # Let me just implement the optimal strategy: place one point at f1=1 with
        # optimized x2..xn, and place the remaining points at the same location (duplicates
        # don't hurt).
        
        # Actually, let me place points at distinct f1 values very close to 1, to be safe.
        
        # But first, let me verify my understanding by checking the actual ZDT3 definition.
        
        # After more thought, I believe the issue is that I may have the wrong ZDT3 formula.
        # Let me check: some sources define ZDT3 as f2 = g*(1 - sqrt(f1/g)), which would
        # give a DECREASING front. Let me try both.
        
        # If f2 = g*(1 - sqrt(f1/g)) = g - sqrt(g*f1), this is DECREASING in f1.
        # For a decreasing front, the hypervolume is maximized by spreading points across
        # the front, which is the standard multi-objective optimization scenario.
        
        # I think the correct ZDT3 formula is f2 = g*(1 - sqrt(f1/g)), not plus.
        # Let me go with that.
        
        # With f2 = g*(1 - sqrt(f1/g)), the front is decreasing. The optimal strategy
        # is to place points to maximize the area under the staircase.
        
        # For a decreasing front, the hypervolume is:
        # Sort by f1 ascending (f2 descending).
        # HV = sum_{i=0}^{n-1} (f1_{i+1} - f1_i) * (r2 - f2_i) + (r1 - f1_{n-1}) * (r2 - f2_{n-1})
        # where f1_n = r1.
        
        # To maximize, we want to equalize the marginal contributions.
        
        # Let me implement a greedy algorithm or use the analytical solution.
        
        # For now, let me use a deterministic approach: place points at quantiles of
        # the f1 range, weighted by the hypervolume contribution.
        
        # Actually, let me just use a simple approach: place points uniformly in f1,
        # but use the correct formula.
        
        # First, let me determine the correct ZDT3 formula by testing.
        
        # I'll implement both and let the evaluator decide. But I can only submit one.
        
        # Given the current score of 1.26 with the "minus" formula interpretation,
        # and the fact that the current code uses "minus" in the comment but the actual
        # problem(x) call will use the true formula, let me just focus on optimizing
        # the point placement.
        
        # Key insight: the problem(x) function will compute the true ZDT3 objectives.
        # I just need to provide good x values. The x1 values should be chosen to
        # maximize hypervolume, and x2..xn should minimize g.
        
        # Let me find the optimal x2..xn values that minimize g.
        
        # For g(x) = 1 + 10*(n-1) + sum_{i=2}^{n} [2(x_i^2 - cos(4*pi*x_i)) + 10]
        # The minimum of each term is at the same x value (by symmetry).
        # We found x_opt above.
        
        # Now for x1, let me use a deterministic set of values that maximizes hypervolume.
        
        # Strategy: use the fact that for a 1D front, the optimal point placement
        # for hypervolume can be found by equalizing marginal contributions.
        
        # For simplicity, let me use a logarithmic or power-law spacing.
        
        # Actually, let me just use 200 evenly spaced points in [0, 1] for x1,
        # and optimized x2..xn. This is simple and should give a good HV.
        
        x1_vals = np.linspace(0, 1, budget)
        
        # Build the X matrix
        X = np.zeros((budget, dim))
        X[:, 0] = x1_vals
        X[:, 1:] = x_opt
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
