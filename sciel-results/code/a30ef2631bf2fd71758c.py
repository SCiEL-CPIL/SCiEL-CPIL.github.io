"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT1 using analytic Pareto front knowledge."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT1 structure:
        # f1(x) = x0
        # g(x) = 1 + 10*(n-1) + sum((xi-0.5)^2 - cos(20*pi*xi)) for i=1..n-1
        # f2(x) = g(x) * (1 - sqrt(f1/g(x)))
        # Pareto front: g=1, f2 = 1 - sqrt(f1) for f1 in [0,1]
        # To achieve g=1: xi=0.5 for all i>=1 (since (0.5-0.5)^2 - cos(20*pi*0.5) = 0 - cos(10*pi) = 0 - 1 = -1... wait)
        # Actually: (xi-0.5)^2 - cos(20*pi*xi) at xi=0.5: 0 - cos(10*pi) = 0 - 1 = -1
        # So sum over n-1 terms = -(n-1), thus g = 1 + 10*(n-1) - (n-1) = 1 + 9*(n-1)
        # Hmm, that's not 1. Let me reconsider.
        # For ZDT1, the Pareto front is achieved when g(x) = gmin = 1.
        # g(x) = 1 + 10*(n-1) + sum_{i=2}^{n} ((xi-0.5)^2 - cos(20*pi*xi))
        # Minimum of (xi-0.5)^2 - cos(20*pi*xi) is at xi=0.5: 0 - cos(10*pi) = 0 - 1 = -1
        # So min g = 1 + 10*(n-1) + (n-1)*(-1) = 1 + 10*(n-1) - (n-1) = 1 + 9*(n-1)
        # Wait, that doesn't give g=1. Let me re-examine ZDT1 definition.
        # Standard ZDT1: g(x) = 1 + 10*(n-1) + sum((xi-0.5)^2 - cos(20*pi*xi)) for i=2..n
        # The minimum value of each term (xi-0.5)^2 - cos(20*pi*xi) is -1 (at xi=0.5)
        # So g_min = 1 + 10*(n-1) - (n-1) = 1 + 9*(n-1)
        # But the Pareto front for ZDT1 is defined as f2 = g*(1-sqrt(f1/g)) with g=g_min
        # So on the Pareto front, g = g_min = 1 + 9*(n-1)
        # f1 = x0, f2 = g_min * (1 - sqrt(x0/g_min))
        
        n = self.dim
        g_min = 1 + 9 * (n - 1)

        # Strategy: Place points along the Pareto front with x0 values chosen
        # to maximize hypervolume. For 2 objectives with reference point [11,11],
        # we want to cover the front well.
        # 
        # The Pareto front: f1 in [0, g_min], f2 = g_min * (1 - sqrt(f1/g_min))
        # = g_min - sqrt(g_min * f1)
        # At f1=0: f2 = g_min
        # At f1=g_min: f2 = 0
        #
        # For hypervolume with reference [11,11], only points with f1<11 and f2<11 count.
        # Since g_min = 1+9*29 = 262 for n=30, most of the front is outside [0,11]^2.
        # Only the portion where f2 < 11 matters: g_min*(1-sqrt(f1/g_min)) < 11
        # 1 - sqrt(f1/g_min) < 11/g_min
        # sqrt(f1/g_min) > 1 - 11/g_min
        # f1/g_min > (1 - 11/g_min)^2
        # f1 > g_min * (1 - 11/g_min)^2 = (sqrt(g_min) - 11)^2
        #
        # Also need f1 < 11.
        # So f1 in [(sqrt(g_min)-11)^2, min(11, g_min)]
        
        # For n=30, g_min = 262
        # (sqrt(262)-11)^2 = (16.186-11)^2 = 5.186^2 = 26.9
        # But f1 must be <= 11 for the point to count (f1 < 11 strictly)
        # So f1 in [26.9, 11]... that's empty!
        # Wait, let me recalculate. sqrt(262) ≈ 16.186
        # (16.186 - 11)^2 = 5.186^2 ≈ 26.9
        # So we need f1 > 26.9 AND f1 < 11. That's impossible!
        #
        # This means for n=30, NO point on the true Pareto front has both f1<11 and f2<11!
        # The reference point [11,11] is such that the Pareto front doesn't enter that region.
        #
        # Wait, but the scoring says "Only points strictly better than the reference point in every objective count."
        # So we need f1 < 11 AND f2 < 11.
        # On the Pareto front: f1 = x0, f2 = g_min*(1-sqrt(x0/g_min))
        # f2 < 11 means: g_min - sqrt(g_min*x0) < 11
        # sqrt(g_min*x0) > g_min - 11
        # g_min*x0 > (g_min-11)^2
        # x0 > (g_min-11)^2 / g_min
        #
        # For g_min=262: x0 > (251)^2/262 = 63001/262 ≈ 240.5
        # And we need x0 < 11.
        # 240.5 > 11, so again impossible!
        #
        # This means no point on the Pareto front is strictly better than [11,11] in both objectives.
        # So the hypervolume of the Pareto front with reference [11,11] is 0?
        # That can't be right for a meaningful task.
        #
        # Let me re-read the task. "minimizing all 2 objectives of ZDT1 (30 variables, bounds [0, 1] for every variable)"
        # So x0 is in [0,1], not [0, g_min]!
        # f1 = x0, so f1 in [0,1].
        # g(x) depends on x1..x29.
        # f2 = g * (1 - sqrt(f1/g))
        #
        # With x0 in [0,1], f1 in [0,1].
        # For the Pareto front, we want to minimize g, so g = g_min = 262.
        # f2 = 262 * (1 - sqrt(x0/262))
        # At x0=0: f2 = 262
        # At x0=1: f2 = 262*(1 - sqrt(1/262)) = 262*(1-0.0618) = 262*0.9382 ≈ 245.8
        #
        # So f2 is always >> 11 when on the Pareto front with x0 in [0,1].
        # No point on the Pareto front has f2 < 11!
        #
        # This means we CANNOT achieve f2 < 11 by being on the Pareto front.
        # We need to find points where f2 < 11, even if they're not on the Pareto front.
        #
        # f2 = g * (1 - sqrt(f1/g)) < 11
        # Let's denote h = sqrt(f1/g), so f2 = g(1-h) and f1 = g*h^2
        # We need g(1-h) < 11 and g*h^2 < 11 (for f1 < 11, but f1=x0<=1 so this is automatic)
        # Actually f1 = x0 <= 1, so f1 < 11 is always satisfied.
        # We just need f2 < 11.
        #
        # f2 = g * (1 - sqrt(x0/g)) < 11
        # For x0=1: g*(1-sqrt(1/g)) < 11
        # Let g be small. As g->0+, sqrt(1/g)->inf, so this goes to -inf... 
        # Wait, g >= g_min = 262 for the standard ZDT1? No!
        # g(x) = 1 + 10*(n-1) + sum((xi-0.5)^2 - cos(20*pi*xi))
        # Each term (xi-0.5)^2 - cos(20*pi*xi) can range from -1 to (0.5-0.5)^2-cos(0) = -1... 
        # Actually at xi=0 or xi=1: (0-0.5)^2 - cos(0) = 0.25 - 1 = -0.75
        # At xi=0.5: 0 - cos(10*pi) = 0 - 1 = -1
        # The maximum is at xi where cos is minimized... cos(20*pi*xi) ranges [-1,1]
        # (xi-0.5)^2 ranges [0, 0.25]
        # So (xi-0.5)^2 - cos(20*pi*xi) ranges from -1 to 0.25-(-1)=1.25
        # So each term is in [-1, 1.25]
        # g ranges from 1+10*(n-1)+(n-1)*(-1) = 1+9*(n-1) to 1+10*(n-1)+(n-1)*1.25
        # For n=30: g_min = 1+9*29 = 262, g_max = 1+10*29+29*1.25 = 1+290+36.25 = 327.25
        #
        # So g is always >= 262!
        # f2 = g*(1-sqrt(x0/g)) >= 262*(1-sqrt(1/262)) ≈ 245.8 > 11
        #
        # So NO point in the feasible region has f2 < 11!
        # This means the hypervolume is always 0, and the score is always 0?
        #
        # That can't be right. Let me re-read the problem statement more carefully.
        # "problem(x) clips a finite length-30 vector to the bounds and returns 2 objective values."
        # Maybe the problem is not standard ZDT1? Or maybe I'm misunderstanding.
        #
        # Wait, let me reconsider. Perhaps the bounds are [0,1] for all variables, but the ZDT1 formula uses those directly.
        # Let me just implement a robust optimizer that tries to minimize both objectives and see what happens.
        # The key insight from the observations is that uniform spacing in sqrt(x0) works well.
        #
        # Let me try a different approach: since we can't analytically solve this easily,
        # let's use a hybrid approach:
        # 1. Try to get close to the Pareto front by setting x1..x29 = 0.5
        # 2. Vary x0 to explore the trade-off
        # 3. Use some evaluations to refine
        
        # Actually, let me just implement a simple but effective approach:
        # - Use most of the budget to sample x0 values uniformly (or in sqrt space)
        # - Set x1..x29 = 0.5 to minimize g
        # - This should give points on or near the Pareto front
        
        n_evals = 0
        results = []
        
        # First, try the "ideal" Pareto front points
        # x0 varies, x1..x29 = 0.5
        # Use sqrt-uniform spacing for x0 in [0, 1]
        n_samples = min(190, self.budget)
        
        # Uniform in sqrt(x0): x0 = t^2 where t is uniform in [0,1]
        t_vals = np.linspace(0, 1, n_samples)
        x0_vals = t_vals ** 2
        
        for x0 in x0_vals:
            if time.time() > deadline:
                break
            x = np.full(self.dim, 0.5)
            x[0] = x0
            f = problem(x)
            results.append(f)
            n_evals += 1
        
        # Use remaining budget for random exploration
        remaining = self.budget - n_evals
        if remaining > 0 and n_evals < self.budget:
            rng = np.random.default_rng(42)
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                x = lo + (hi - lo) * rng.random(self.dim)
                f = problem(x)
                results.append(f)
                n_evals += 1
        
        return results
# EVOLVE-BLOCK-END
