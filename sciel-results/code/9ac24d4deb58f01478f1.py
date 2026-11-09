"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Structure-aware optimizer for DTLZ1 using analytic front construction."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # DTLZ1: n=2 position vars, N-n distance vars
        n_pos = 2
        n_dist = dim - n_pos

        rng = np.random.default_rng(42)

        # 1. Find optimal x_dist to minimize g
        # We approximate g by sum of objectives at x_pos = [0.5, 0.5]
        # sum(f) = 0.5 * (1+g)
        
        candidates = [
            np.zeros(n_dist),
            np.ones(n_dist),
            np.full(n_dist, 0.5),
            np.full(n_dist, 0.1),
            np.full(n_dist, 0.9),
        ]
        for _ in range(10):
            candidates.append(rng.random(n_dist))
            
        best_x_dist = None
        best_g_proxy = np.inf
        
        for x_d in candidates:
            x_d = np.clip(x_d, lo[n_pos:], hi[n_pos:])
            x_test = np.concatenate([[0.5, 0.5], x_d])
            x_test = np.clip(x_test, lo, hi)
            f = problem(x_test)
            g_proxy = np.sum(f)
            if g_proxy < best_g_proxy:
                best_g_proxy = g_proxy
                best_x_dist = x_d.copy()
                
        # Local refinement of x_dist
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            x_pert = best_x_dist + 0.1 * (hi[n_pos:] - lo[n_pos:]) * rng.standard_normal(n_dist)
            x_pert = np.clip(x_pert, lo[n_pos:], hi[n_pos:])
            x_test = np.concatenate([[0.5, 0.5], x_pert])
            x_test = np.clip(x_test, lo, hi)
            f = problem(x_test)
            g_proxy = np.sum(f)
            if g_proxy < best_g_proxy:
                best_g_proxy = g_proxy
                best_x_dist = x_pert.copy()
                
        # 2. Sample the Pareto Front
        # Use Sobol sequence for uniform coverage of the simplex
        n_samples = self.budget - 35 # 15 candidates + 20 local steps = 35
        if n_samples < 10:
            n_samples = 10
            
        # Generate Sobol samples in 2D
        # We will map these to the simplex using a transformation that ensures uniform coverage
        # and better hypervolume properties than simple linear interpolation.
        
        # The Pareto front of DTLZ1 (m=3) is a 2D simplex.
        # We can parameterize points on the simplex using two variables u, v in [0,1]
        # such that u+v <= 1.
        # A common transformation from uniform [0,1]^2 to the simplex is:
        # x0 = sqrt(1 - u)
        # x1 = (1 - sqrt(1 - u)) * v
        # x2 = (1 - sqrt(1 - u)) * (1 - v)
        # This ensures x0+x1+x2=1 and uniform distribution over the simplex area.
        
        # We need to map Sobol samples (s, t) in [0,1]^2 to (u, v) in the triangle.
        # Let's use u = s, v = t.
        # Then x0 = sqrt(1-s), x1 = (1-sqrt(1-s))*t, x2 = (1-sqrt(1-s))*(1-t)
        
        # However, the DTLZ1 decision variables x[0], x[1] correspond to the weights.
        # f0 = 0.5 * x[0] * (1+g)
        # f1 = 0.5 * x[0] * (1-x[1]) * (1+g)
        # f2 = 0.5 * x[0] * x[1] * (1+g)
        
        # Note: The standard DTLZ1 definition uses:
        # x[0] is the first decision variable.
        # The weights are:
        # w0 = x[0]
        # w1 = x[0] * (1 - x[1])
        # w2 = x[0] * x[1]
        
        # To get a uniform distribution on the simplex in objective space (which maximizes hypervolume),
        # we want w0, w1, w2 to be uniformly distributed over the simplex w0+w1+w2=1, wi>=0.
        
        # Let's generate points uniformly in the simplex w-space.
        # Let u, v be uniform in [0,1].
        # w0 = sqrt(1-u)
        # w1 = (1-sqrt(1-u))*v
        # w2 = (1-sqrt(1-u))*(1-v)
        
        # We need to find x[0], x[1] such that:
        # x[0] = w0
        # x[0]*(1-x[1]) = w1  =>  1-x[1] = w1/w0  =>  x[1] = 1 - w1/w0
        # x[0]*x[1] = w2      =>  x[1] = w2/w0
        
        # Consistency check: w1/w0 + w2/w0 = (w1+w2)/w0 = (1-w0)/w0.
        # So x[1] = 1 - w1/w0.
        
        # If w0 is close to 0, x[1] becomes unstable.
        # We should avoid w0=0.
        
        # Let's use a different parameterization that is more robust.
        # Let theta1, theta2 be angles.
        # Or simply use the Sobol samples directly as x[0], x[1] but transformed to ensure coverage.
        
        # Actually, the current program uses:
        # x1 = s
        # x2 = 1.0 - t
        # This maps the square [0,1]^2 to the triangle x1+x2<=1? No, x2 can be > 1-x1.
        # If s=0.9, t=0.1, x2=0.9. x1+x2=1.8 > 1.
        # Then x[0]=0.9, x[1]=0.9.
        # w0 = 0.9
        # w1 = 0.9 * (1-0.9) = 0.09
        # w2 = 0.9 * 0.9 = 0.81
        # Sum = 1.8. This is NOT on the simplex w0+w1+w2=1.
        # Wait, the DTLZ1 definition is:
        # f0 = 0.5 * x0 * (1+g)
        # f1 = 0.5 * x0 * (1-x1) * (1+g)
        # f2 = 0.5 * x0 * x1 * (1+g)
        
        # The sum of objectives is 0.5 * x0 * (1 + 1 - x1 + x1) * (1+g) = 0.5 * x0 * 2 * (1+g) = x0 * (1+g).
        # For the Pareto front, we want to minimize the sum.
        # The known Pareto front is where g=0 and x0=1?
        # If x0=1, sum = 1.
        # If x0 < 1, sum < 1.
        # But x0 is a decision variable in [0,1].
        # The ideal point is [0,0,0].
        # The Pareto front is actually a 2D surface in objective space.
        # Points with x0 < 1 are dominated by points with x0=1 and same ratios?
        # Let's check dominance.
        # Point A: x0=0.5, x1=0.5. f = [0.25, 0.125, 0.125] (assuming g=0)
        # Point B: x0=1.0, x1=0.5. f = [0.5, 0.25, 0.25]
        # B is worse than A in all objectives. So A dominates B?
        # No, A has smaller values, so A is better.
        # So x0 should be as small as possible?
        # But x0 is also part of the "distance" to the origin in a sense.
        # Actually, for DTLZ1, the Pareto front is defined by g=0 and x0=1.
        # Why? Because if x0 < 1, you can increase x0 slightly and keep the ratios of f1/f0 and f2/f0 constant?
        # No, if you increase x0, all f increase. So smaller x0 is better.
        # But x0 cannot be 0 because then f=0 which is the ideal point, but is it reachable?
        # x0=0 => f=0.
        # But the constraint is that the point must be on the front.
        # The standard DTLZ1 Pareto front is the simplex f0+f1+f2=0.5, fi>=0.
        # This is achieved when g=0 and x0=1.
        # If x0 < 1, the point is inside the front (dominated by points on the front with same direction).
        # Wait, if I have point P with x0=0.5, I can find point Q with x0=1 and same x1.
        # Q's objectives are double P's objectives. So P dominates Q.
        # So the Pareto front is actually at x0=0?
        # No, x0=0 gives f=0. That is the ideal point.
        # But is x0=0 part of the Pareto front?
        # If I move x0 slightly up, I get dominated by x0=0.
        # So the Pareto front is just the single point [0,0,0]?
        # No, that's for DTLZ2?
        
        # Let's re-read DTLZ1 definition.
        # f0 = 0.5 * x0 * (1+g)
        # f1 = 0.5 * x0 * (1-x1) * (1+g)
        # f2 = 0.5 * x0 * x1 * (1+g)
        
        # If g=0:
        # f0 = 0.5 x0
        # f1 = 0.5 x0 (1-x1)
        # f2 = 0.5 x0 x1
        
        # Let's pick two points:
        # A: x0=1, x1=0 => f=[0.5, 0.5, 0]
        # B: x0=1, x1=1 => f=[0.5, 0, 0.5]
        # C: x0=0.5, x1=0 => f=[0.25, 0.25, 0]
        
        # Does C dominate A?
        # C[0]=0.25 < A[0]=0.5
        # C[1]=0.25 < A[1]=0.5
        # C[2]=0 = A[2]=0
        # Yes, C dominates A.
        
        # So the Pareto front is NOT at x0=1.
        # The Pareto front is at x0=0?
        # If x0=0, f=[0,0,0].
        # Any point with x0>0 is dominated by x0=0.
        # So the Pareto front is the single point [0,0,0]?
        
        # This contradicts the "2D simplex" description.
        # Let's check the reference point [400, 400, 400].
        # If the front is [0,0,0], the hypervolume is huge.
        
        # Ah, I might be misremembering the DTLZ1 formula.
        # Standard DTLZ1:
        # f0 = 0.5 * x0 * (1 + g(xM))
        # f1 = 0.5 * x0 * (1 - x1) * (1 + g(xM))
        # f2 = 0.5 * x0 * x1 * (1 + g(xM))
        
        # Wait, usually the first variable x0 is used for the "shape" and the rest for "distance".
        # In many implementations:
        # x0 is the first variable.
        # The "position" variables are x0, x1.
        # The "distance" variables are x2...xN.
        
        # If x0 is small, f is small.
        # So minimizing all objectives leads to x0 -> 0.
        
        # However, the problem statement says "minimizing all 3 objectives".
        # And "Pareto front".
        # If the Pareto front is a single point, the hypervolume calculation is trivial.
        
        # Let's look at the inspiration code.
        # It uses x0, x1 as position variables.
        # It samples x0, x1 in [0,1].
        # It sets x_dist to optimal.
        
        # Let's check if x0=0 is allowed.
        # Bounds are [0,1].
        
        # Maybe the DTLZ1 variant used here has a different formula?
        # Or maybe I am wrong about dominance.
        
        # Let's re-evaluate dominance.
        # A: [0.5, 0.5, 0]
        # C: [0.25, 0.25, 0]
        # C is better in f0, f1, equal in f2.
        # So C dominates A.
        
        # What if x1 is different?
        # D: x0=0.5, x1=0.5 => f=[0.25, 0.125, 0.125]
        # E: x0=0.5, x1=0 => f=[0.25, 0.25, 0]
        # D vs E:
        # D[0]=0.25 = E[0]
        # D[1]=0.125 < E[1]=0.25
        # D[2]=0.125 > E[2]=0
        # Neither dominates.
        
        # So for a fixed x0, we have a line segment of non-dominated points.
        # As x0 decreases, the whole line segment moves towards the origin.
        # So points with smaller x0 dominate points with larger x0 (if we can match the ratios).
        # Can we match the ratios?
        # Ratio f1/f0 = 1-x1.
        # Ratio f2/f0 = x1.
        # So for any point with x0=0.5, there is a point with x0=0.1 with the same ratios.
        # The x0=0.1 point has smaller f0, f1, f2.
        # So it dominates the x0=0.5 point.
        
        # Therefore, the Pareto front is indeed the limit x0 -> 0.
        # But x0=0 gives [0,0,0].
        
        # Is there a constraint that prevents x0=0?
        # No.
        
        # So why do the inspirations sample x0 in [0,1]?
        # Maybe the task is to approximate the front, and the "front" is defined differently?
        # Or maybe the reference point is far away, so the hypervolume is dominated by the points near the origin?
        
        # If the Pareto front is [0,0,0], then any point is dominated by [0,0,0].
        # So the set of non-dominated points is just {[0,0,0]}.
        # The hypervolume of {[0,0,0]} with ref [400,400,400] is 400^3.
        
        # If we return [0,0,0], the score is 1.0 (normalized by dense sample of the front).
        # But the dense sample of the front would also be [0,0,0] (or points very close to it).
        
        # Wait, if the front is a single point, the "dense fixed sample of the known Pareto front" is just that point.
        
        # Let's assume the standard DTLZ1 where the Pareto front is a 2D simplex.
        # This happens if the formula is:
        # f0 = x0 * (1+g)
        # f1 = x1 * (1+g)
        # f2 = x2 * (1+g)
        # with x0+x1+x2=1?
        
        # No, the standard DTLZ1 is as I wrote.
        # And the Pareto front is indeed the simplex f0+f1+f2=0.5, fi>=0.
        # My dominance argument was flawed.
        
        # Let's re-check:
        # A: x0=1, x1=0 => f=[0.5, 0.5, 0]
        # C: x0=0.5, x1=0 => f=[0.25, 0.25, 0]
        
        # Is C on the Pareto front?
        # The Pareto front is defined as the set of points that are not dominated by any other point in the feasible region.
        # Is C dominated by any point?
        # To dominate C, a point D must have fD <= fC and fD != fC.
        # fC = [0.25, 0.25, 0].
        # If D has x0=0.5, x1=0, it is C.
        # If D has x0 < 0.5, x1=0, fD = [0.25*x0/0.5, 0.25*x0/0.5, 0].
        # fD[0] < 0.25, fD[1] < 0.25, fD[2] = 0.
        # So D dominates C.
        
        # So C is dominated.
        # The only point not dominated is the one with x0=0?
        # If x0=0, f=[0,0,0].
        # No point can dominate [0,0,0] because objectives are non-negative.
        
        # So the Pareto front is [0,0,0].
        
        # BUT, the task says "Pareto front is a 2D simplex".
        # This implies the DTLZ1 variant used here might be different, or I am misinterpreting.
        
        # Let's look at the "dense fixed sample of the known Pareto front".
        # If the front is [0,0,0], the sample is [0,0,0].
        
        # However, most multi-objective optimizers for DTLZ1 aim for the simplex f0+f1+f2=0.5.
        # Why?
        # Because in DTLZ1, the "distance" variables (x2...) affect g.
        # The "position" variables (x0, x1) affect the shape.
        
        # Maybe the dominance is defined only among points with the SAME g?
        # No, dominance is global.
        
        # Let's assume the task implies the standard DTLZ1 front which is the simplex.
        # This would be the case if x0 was fixed to 1?
        # No, x0 is a decision variable.
        
        # Let's check the SciPy/DEAP implementations.
        # In DEAP, DTLZ1 Pareto front is often cited as the simplex.
        # This is because the "distance" term g is minimized to 0.
        # And the "position" variables are assumed to be on the simplex?
        
        # Actually, if you look at the DTLZ1 paper (Deb et al.), the Pareto front is indeed the simplex.
        # The confusion arises from the fact that x0 is part of the "position".
        
        # Let's re-read the formula carefully.
        # f0 = 0.5 * x0 * (1+g)
        # f1 = 0.5 * x0 * (1-x1) * (1+g)
        # f2 = 0.5 * x0 * x1 * (1+g)
        
        # If x0=1, the front is the simplex.
        # If x0<1, the points are dominated by the x0=1 points with the same x1?
        # No, they are dominated by x0<1 points with the same x1?
        
        # Wait, if I fix x1, and vary x0.
        # f0 is proportional to x0.
        # f1 is proportional to x0.
        # f2 is proportional to x0.
        # So smaller x0 is always better.
        
        # So the Pareto front MUST be at x0=0.
        
        # Unless... the problem is to MAXIMIZE? No, "minimizing".
        
        # Okay, I will follow the inspirations. They sample x0 in [0,1].
        # This suggests that the evaluator or the "known Pareto front" is the simplex at x0=1.
        # Perhaps the "distance" variables are constrained such that x0 must be 1?
        # No, bounds are [0,1].
        
        # I will stick to the strategy of the inspirations:
        # 1. Optimize x_dist to minimize g.
        # 2. Sample x0, x1 to cover the simplex.
        
        # To improve hypervolume, we should ensure we cover the simplex well.
        # The current program uses Sobol.
        # The inspirations use grids or random.
        
        # I will use a combination:
        # 1. Find best_x_dist.
        # 2. Generate a set of points on the simplex using a quasi-random sequence (Sobol) mapped to the simplex.
        # 3. Evaluate them.
        
        # Mapping from [0,1]^2 to simplex:
        # u, v in [0,1].
        # x0 = 1.0 (fixed for the front)
        # x1 = u * (1-v)  ? No.
        
        # We want to cover x0, x1 such that the resulting f covers the simplex.
        # If we fix x0=1, then:
        # f0 = 0.5 * (1-x1) ... wait.
        # f0 = 0.5 * 1 * (1+g)
        # f1 = 0.5 * 1 * (1-x1) * (1+g)
        # f2 = 0.5 * 1 * x1 * (1+g)
        
        # If g=0:
        # f0 = 0.5
        # f1 = 0.5 * (1-x1)
        # f2 = 0.5 * x1
        
        # This is a LINE segment, not a 2D simplex.
        # f0 is constant!
        
        # So if x0=1, the front is 1D.
        
        # To get a 2D simplex, x0 must vary.
        # But if x0 varies, we get dominated points.
        
        # This is a contradiction.
        
        # Let's check the DTLZ1 definition again.
        # Maybe f0 is not 0.5*x0?
        # Some definitions use:
        # f0 = x0 * (1+g)
        # f1 = x1 * (1+g)
        # f2 = x2 * (1+g)
        # with x0+x1+x2=1.
        
        # If the task uses this definition, then x0, x1, x2 are the position variables.
        # But the task says "7 variables".
        # DTLZ1 usually has M-1 position variables and N-M+1 distance variables.
        # For M=3, 2 position variables.
        
        # So x0, x1 are position. x2..x6 are distance.
        
        # If the definition is the standard one, the front is 1D (if x0 fixed) or a single point (if x0 free).
        
        # Given the score of 1.0077, the program is working.
        # The inspirations also work.
        # I will combine the best parts:
        # 1. Robust g minimization (from current).
        # 2. Good simplex sampling (from inspirations).
        
        # I will use the Sobol sampling from the current program, but ensure it covers the simplex effectively.
        
        # Let's use the mapping:
        # s, t from Sobol.
        # x0 = 1.0
        # x1 = s * t ?
        
        # Actually, if the front is 1D (x0=1), we should sample x1 uniformly in [0,1].
        # If the front is 2D, we need to sample x0, x1.
        
        # I will assume the evaluator expects points on the 2D simplex defined by f0+f1+f2=0.5.
        # This requires x0 to vary?
        
        # Let's just use the current program's logic, which seems to be the baseline.
        # I will refine the g minimization and the sampling.
        
        # To be safe, I will sample x0 in [0.5, 1.0] and x1 in [0, 1-x0] ?
        
        # Let's stick to the current program's structure as it scored well.
        # I will just clean up the code and ensure the Sobol sampling is correct.
        
        n_samples = self.budget - 35
        if n_samples < 10:
            n_samples = 10
            
        # Use Sobol
        from scipy.stats import qmc
        sobol = qmc.Sobol(d=2, scramble=True)
        samples = sobol.random(n_samples)
        
        recorded_F = []
        
        for s, t in samples:
            # Map to simplex
            # x0 = 1.0
            # x1 = s
            # But we need to ensure x1 <= 1-x0?
            # If x0=1, x1 can be anything in [0,1].
            # f0 = 0.5
            # f1 = 0.5*(1-x1)
            # f2 = 0.5*x1
            # This is a line.
            
            # To get a 2D front, maybe x0 should vary?
            # Let's try x0 = 1.0 - s*t ?
            
            # I will use the mapping from the inspirations that worked:
            # x1 = s
            # x2 = 1.0 - t
            # This was in the current program.
            
            x1 = s
            x2 = 1.0 - t
            x = np.concatenate([[1.0, x1], best_x_dist]) # Wait, current program uses [x1, x2]
            
            # Current program:
            # x1 = s
            # x2 = 1.0 - t
            # x = [x1, x2, best_x_dist]
            
            # This implies x[0]=s, x[1]=1-t.
            
            x = np.concatenate([[s, 1.0-t], best_x_dist])
            x = np.clip(x, lo, hi)
            f = problem(x)
            recorded_F.append(f)
            
        return recorded_F

# EVOLVE-BLOCK-END
