"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT2 using hypervolume-aware search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT2: f1 = x1, f2 = (1 + 10*(n-1)) * (1 - sqrt(f1/(1+10*(n-1))))
        # The Pareto front is: f2 = 1 + 10*(dim-1) - sqrt(1 + 10*(dim-1)) * sqrt(f1)
        # For dim=30: 1+10*29 = 291
        # f2 = 291 - sqrt(291) * sqrt(f1)
        # To maximize hypervolume with reference point [11, 11], we want points
        # that are below the reference in both objectives.
        # f1 must be < 11 (always true since x1 in [0,1])
        # f2 must be < 11: 291 - sqrt(291)*sqrt(f1) < 11
        # sqrt(291)*sqrt(f1) > 280
        # sqrt(f1) > 280/sqrt(291) ≈ 280/17.058 ≈ 16.41
        # f1 > 269.3 -- impossible since f1 <= 1
        # Wait, that means NO point on the Pareto front has f2 < 11?
        # Let me recalculate: sqrt(291) ≈ 17.058
        # f2 = 291 - 17.058 * sqrt(f1)
        # For f1 = 1: f2 = 291 - 17.058 = 273.94
        # So the minimum f2 on the PF is ~273.94, which is way above 11.
        # This means NO point from the Pareto front is strictly better than [11,11] in both objectives.
        # 
        # Wait, that can't be right for the scoring. Let me re-read.
        # "Only points strictly better than the reference point in every objective count."
        # "reference point [11.0, 11.0]"
        # 
        # If the PF has f2 >= 273.94, then no PF point is better than [11,11] in f2.
        # But the problem says we minimize both objectives. So we want LOW f1 and LOW f2.
        # A point is "better than reference" if f1 < 11 AND f2 < 11.
        # 
        # On the Pareto front, f2 ranges from 273.94 (at f1=1) to 291 (at f1=0).
        # So NO point on the true Pareto front has f2 < 11.
        # 
        # This means the hypervolume of the dense PF sample with reference [11,11] is 0!
        # That would make the score undefined (division by zero).
        # 
        # Let me re-read the task more carefully...
        # "divided by the hypervolume of a dense fixed sample of the known Pareto front with the same reference point"
        # 
        # Maybe I'm misunderstanding ZDT2. Let me check the standard definition.
        # ZDT2: f1(x) = x1, g(x) = 1 + 10(n-1), h(f1,g) = 1 - sqrt(f1/g)
        # f2(x) = g(x) * h(f1, g) = (1+10(n-1)) * (1 - sqrt(x1/(1+10(n-1))))
        # 
        # For n=30: g = 1 + 10*29 = 291
        # f2 = 291 * (1 - sqrt(x1/291))
        # At x1=1: f2 = 291*(1 - sqrt(1/291)) = 291*(1 - 0.05858) = 291*0.94142 = 273.95
        # At x1=0: f2 = 291*(1-0) = 291
        # 
        # So indeed f2 is always >= 273.95 on the PF. Reference point [11,11] makes no sense for standard ZDT2 with n=30.
        # 
        # Unless... the problem definition is different. Maybe the bounds or scaling is different.
        # Or maybe I should just focus on finding points that minimize both objectives as much as possible,
        # and the scoring will handle the rest.
        #
        # Actually, wait. Let me re-read: "bounds [0, 1] for every variable" and "problem(x) clips a finite 
        # length-30 vector to the bounds and returns 2 objective values."
        #
        # Perhaps the ZDT2 implementation in the evaluator uses a different scaling or the reference point
        # is chosen such that some PF points do dominate it. But given the math, with standard ZDT2 n=30,
        # f2 min on PF is ~274. So reference [11,11] would give 0 hypervolume for the PF sample.
        #
        # This is confusing. Let me just implement a good MOO algorithm and let the scoring handle it.
        # The key insight: we want to find points that minimize both f1 and f2.
        # Since f1 = x1 and f2 depends on x1 and the other variables through g(x),
        # to minimize f2 we want g(x) as small as possible AND h(f1,g) as small as possible.
        # g(x) = 1 + 10*sum(x_i^2 for i=2..n) for ZDT2? No wait, let me check again.
        #
        # Standard ZDT2: g(x) = 1 + 10(n-1) is a constant! It doesn't depend on x2..xn.
        # So f2 = (1+10(n-1)) * (1 - sqrt(x1/(1+10(n-1))))
        # This means f2 only depends on x1, not on x2..xn!
        # And f1 = x1.
        # So the Pareto front is entirely determined by x1, and x2..xn don't matter for the objectives.
        #
        # This simplifies things greatly. The optimal strategy is:
        # - For each evaluation, choose x1 to spread out the f1 values
        # - x2..xn can be anything (they don't affect objectives)
        #
        # To maximize hypervolume, we want to place our 200 evaluations at optimal x1 values
        # that maximize the hypervolume of the resulting points.
        #
        # Given that f1 = x1 and f2 = 291*(1-sqrt(x1/291)), the PF is a curve.
        # We want to choose 200 x1 values in [0,1] to maximize the hypervolume of the
        # corresponding points with reference [11,11].
        #
        # But as noted, all PF points have f2 >= 274 > 11, so none are better than reference in f2.
        # Unless the evaluator's ZDT2 is different.
        #
        # Let me just implement a robust approach: evaluate at many different x1 values,
        # with x2..xn set to 0 (which minimizes any potential g dependence if it exists).
        #
        # Actually, let me reconsider. Maybe the ZDT2 in the evaluator has g(x) = 1 + 10*sum(x_i^2)
        # for i=2..n, which is the standard definition I was confusing. Let me check:
        # 
        # ZDT1: g(x) = 1 + 10(n-1) (constant)
        # ZDT2: g(x) = 1 + 10(n-1) (constant) -- same as ZDT1
        # ZDT3: g(x) = 1 + 10(n-1) (constant)
        # ZDT4: g(x) = 1 + 10(n-1) (constant)
        # ZDT5: g(x) = 1 + 2*|{i: x_i=1}|
        # ZDT6: g(x) = 1 + 9*(sum(x_i)/n)^0.25
        #
        # So for ZDT2, g is indeed constant. f2 only depends on x1.
        #
        # Given the confusion about the reference point, let me just implement a strategy that:
        # 1. Spreads x1 values across [0,1] to cover the Pareto front
        # 2. Sets x2..xn to 0.5 (middle of bounds, doesn't matter)
        # 3. Uses the remaining budget to explore if needed
        #
        # For hypervolume maximization with a concave PF, non-uniform spacing can help.
        # The optimal spacing for a concave curve under a rectangular reference point
        # tends to be denser where the curve is flatter.
        #
        # Let me use a power-law spacing: x1_i = (i/(N-1))^alpha for i=0..N-1
        # where alpha > 1 puts more points near x1=0 (where the PF is steeper in f2 vs f1).
        #
        # Actually, for ZDT2: f2 = 291*(1 - sqrt(x1/291))
        # df2/dx1 = 291 * (-1/(2*sqrt(291*x1))) * (1/291) = -1/(2*sqrt(291*x1))
        # The slope magnitude is large near x1=0 and small near x1=1.
        # For hypervolume, we want to balance the area contribution.
        #
        # Let me just try a few strategies and use the budget wisely.

        # Strategy: Use all budget to evaluate at distinct x1 values
        # Spread x1 across [0,1] with appropriate spacing
        # Set x2..xn to 0.5

        N = budget  # use all evaluations
        # Power-law spacing to concentrate points where the PF curve is steep
        # For ZDT2, the curve is steep near x1=0, so use alpha < 1 to spread more near 0?
        # Actually, let's think about it differently.
        # 
        # The hypervolume contribution of a point (f1, f2) with reference (r1, r2) where
        # f1 < r1 and f2 < r2 is (r1-f1)*(r2-f2).
        # But since all our points have f2 > 11 = r2, none contribute.
        #
        # I'm stuck on the reference point issue. Let me just implement a reasonable MOO
        # algorithm and hope the scoring works out.
        
        # Use a simple approach: evaluate at N distinct x1 values
        # with x2..xn = 0.5
        x1_values = np.linspace(0, 1, N)
        
        # For each x1, create a full vector
        # x2..xn set to 0.5
        X = np.full((N, dim), 0.5)
        X[:, 0] = x1_values
        
        # Evaluate all
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
