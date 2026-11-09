import numpy as np
import time

class Optimizer:
        def __init__(self, budget, dim, n_obj):
            self.budget = budget
            self.dim = dim
            self.n_obj = n_obj

        def __call__(self, problem, bounds):
            start_time = time.time()
            deadline = start_time + 290  # 300s limit, 10s safety margin
            
            # 1. Prepare bounds
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            mid = (lo + hi) / 2.0
            
            # 2. Generate points on the DTLZ2 Pareto front
            # DTLZ2 front is a quarter-sphere in 3D: f1^2 + f2^2 + f3^2 = 1, fi >= 0
            # Parameterization: 
            # theta1 in [0, pi/2], theta2 in [0, pi/2]
            # f1 = cos(theta1)
            # f2 = sin(theta1) * cos(theta2)
            # f3 = sin(theta1) * sin(theta2)
            
            # To sample uniformly on the surface of the quarter-sphere:
            # The surface area element is proportional to sin(theta1) dtheta1 dtheta2
            # So we need to sample theta1 with density proportional to sin(theta1)
            # and theta2 uniformly.
            
            # For theta1: let u ~ U(0,1), then theta1 = arccos(1 - u) 
            # Wait, let's check: if theta1 = arccos(1-u), then u = 1 - cos(theta1)
            # du = sin(theta1) dtheta1. So yes, this gives uniform area sampling.
            # But theta1 must be in [0, pi/2].
            # If u in [0,1], then 1-u in [0,1], arccos(1-u) in [0, pi/2]. Correct.
            
            # For theta2: uniform in [0, pi/2]. Let v ~ U(0,1), theta2 = v * pi/2.
            
            # We have budget evaluations. We want to use all of them.
            # We will generate budget points.
            
            n_points = self.budget
            
            # Use a low-discrepancy sequence for better coverage.
            # Halton or Sobol. Let's use Halton for simplicity, or random if time is tight.
            # But the previous program used Halton. Let's stick to a robust, fast method.
            # Since the mapping is non-linear, uniform in (u,v) space maps to uniform area on sphere.
            # So we just need a good 2D low-discrepancy sequence.
            
            # Let's use a simple grid or Halton.
            # With 200 points, a 14x14 grid is 196 points.
            # Halton is generally better for arbitrary N.
            
            def halton(index, base):
                f = 1.0
                r = 0.0
                while index > 0:
                    f = f / base
                    r = r + f * (index % base)
                    index //= base
                return r

            u = np.array([halton(i+1, 2) for i in range(n_points)])
            v = np.array([halton(i+1, 3) for i in range(n_points)])
            
            # Map to angles
            theta1 = np.arccos(1.0 - u)
            theta2 = v * (np.pi / 2.0)
            
            # Calculate objectives (which are the coordinates on the front)
            # f1 = cos(theta1)
            # f2 = sin(theta1) * cos(theta2)
            # f3 = sin(theta1) * sin(theta2)
            
            # We need to construct x such that problem(x) returns these f values.
            # DTLZ2:
            # f1 = g(x) * (1 + sum(x_i^2 for i in 3..n)) * cos(x1) * cos(x2) ... wait, standard DTLZ2:
            # f1 = g * (1 + sum_{i=2}^{n} x_i^2) * cos(x_1)
            # f2 = g * (1 + sum_{i=2}^{n} x_i^2) * sin(x_1) * cos(x_2)
            # f3 = g * (1 + sum_{i=2}^{n} x_i^2) * sin(x_1) * sin(x_2)
            # where g(x) = 1 + 10*(n-1) + sum (x_i^2 - 2cos(20*pi*x_i)) for i=2..n
            # For Pareto front, g=1 and x_i=0.5 for i=2..n.
            # Then sum x_i^2 = (n-2)*0.25.
            # Wait, the standard definition often has x_1, x_2 as the first two variables.
            # Let's check the variable indices in the prompt.
            # "12 variables, bounds [0, 1] for every variable."
            # "minimizing all 3 objectives of DTLZ2"
            # Standard DTLZ2 with M=3 objectives uses 2 decision variables for the front (x1, x2) and the rest for g.
            # If dim=12, then x1, x2 are the front variables, x3..x12 are for g.
            
            # Let's verify the mapping from x to f.
            # If we set x3..x12 = 0.5, then g(x) = 1 + 10*(10) + sum_{i=3}^{12} (0.25 - 2cos(10*pi))
            # cos(10*pi) = 1.
            # g = 1 + 100 + 10*(0.25 - 2) = 101 + 10*(-1.75) = 101 - 17.5 = 83.5.
            # This is not 1. So the points will NOT be on the Pareto front if we just set x_i=0.5.
            
            # Ah, the definition of g in DTLZ2 is:
            # g(x) = 1 + 10*(n-M) + sum_{i=M+1}^n (x_i^2 - 2*cos(20*pi*x_i))
            # For the Pareto front, we need g(x)=1 and the sum term to be 0.
            # The minimum of x_i^2 - 2cos(20*pi*x_i) is -1 at x_i=0.5.
            # So sum is (n-M)*(-1).
            # g = 1 + 10*(n-M) - (n-M) = 1 + 9*(n-M).
            # This is 1 only if n=M, which is not the case here (n=12, M=3).
            
            # Wait, I might be misremembering the DTLZ2 definition or the "known Pareto front" implies we should target the ideal front where g=1?
            # Usually, the Pareto front of DTLZ2 is defined as the set of points where g=1.
            # But g=1 is only achievable if the constant term allows it?
            # No, g is a function of x. We can choose x to make g=1?
            # g(x) >= 1 + 10(n-M) - (n-M) = 1 + 9(n-M).
            # So the minimum possible value of g is 1 + 9(n-M).
            # Therefore, the Pareto front is scaled by this factor?
            # Or is the "Pareto front" defined as the set of non-dominated points, which occurs when g is minimized?
            # Yes, the Pareto front is the set of points where g is at its minimum.
            # So the objectives are scaled by g_min = 1 + 9*(n-M).
            # For n=12, M=3, g_min = 1 + 9*9 = 82.
            
            # However, the problem statement says:
            # "divided by the hypervolume of a dense fixed sample of the known Pareto front"
            # This implies the reference front is the true Pareto front of the problem.
            # If the problem instance uses standard DTLZ2, the front is scaled.
            # But wait, looking at the previous program:
            # It set x3..x12 to 0.5.
            # And it mapped x1, x2 to the unit sphere.
            # It scored 1.1187.
            # If the front was scaled by 82, the hypervolume would be huge.
            # But the reference point is [2.5, 2.5, 2.5].
            # If the objectives are ~82, they are all > 2.5.
            # "Only points strictly better than the reference point in every objective count."
            # If f > 2.5, they don't count.
            # So if the front was scaled by 82, the score would be 0.
            # Since the score is > 1, the objectives MUST be <= 2.5.
            # This implies g=1.
            
            # How can g=1?
            # Maybe the definition of DTLZ2 used here is different?
            # Or maybe the "bounds" and "problem" function handle it differently?
            # The prompt says: "problem(x) clips a finite length-12 vector to the bounds and returns 3 objective values."
            # It doesn't explicitly give the formula.
            # But standard DTLZ2 is the only reasonable assumption.
            
            # Let's re-read carefully.
            # "minimizing all 3 objectives of DTLZ2"
            # If the implementation is standard, g_min = 1 + 9(n-M).
            # Is it possible that n-M is small? No, n=12, M=3.
            
            # Perhaps the "known Pareto front" in the scoring is NOT the actual Pareto front of the function, but a normalized one?
            # "divided by the hypervolume of a dense fixed sample of the known Pareto front with the same reference point"
            # This suggests the denominator is a fixed constant.
            # If the numerator is 0 (because points are > 2.5), the score is 0.
            # Since score > 0, the points MUST be < 2.5.
            
            # This implies that for the specific `problem` function provided, the objectives are small.
            # This could happen if:
            # 1. The DTLZ2 implementation is non-standard (e.g. g is always 1).
            # 2. The variables x3..x12 are set such that g=1.
            #    But g(x) = 1 + 10(n-M) + sum...
            #    Can we make the sum negative enough?
            #    Min sum is -(n-M).
            #    Min g is 1 + 9(n-M).
            #    It is impossible to get g=1 if n>M.
            
            # UNLESS... the problem definition is:
            # f1 = (1 + g) * cos...
            # No, that's DTLZ1.
            
            # Let's look at the previous program's success.
            # It used x1 = (2/pi) * arccos(1-u).
            # This maps u in [0,1] to x1 in [0,1].
            # Wait, arccos(1-u) is in [0, pi/2].
            # So x1 is in [0, 1].
            # Then f1 = g * ... * cos(x1)?
            # If x1 is in [0,1] radians, cos(x1) is in [0, 1].
            # If g is large, f1 is large.
            
            # Is it possible that the `problem` function provided in the evaluation environment is a MOCK or a SPECIFIC implementation where g=1?
            # The prompt says "problem(x) ... returns 3 objective values". It doesn't show the code.
            # But it says "known Pareto front".
            
            # Let's assume the previous program worked because it generated points on the unit sphere (or a scaled version) and they were within the reference point.
            # If the previous program scored 1.11, it means the points generated were valid and good.
            # The previous program set x3..x12 = 0.5.
            # If that resulted in points < 2.5, then g must be small.
            # This strongly suggests that the `problem` function implements a DTLZ2 variant where the scaling factor is 1, or the "g" part is handled such that the front is the unit sphere.
            
            # Given the constraint to "KEEP this design", I should not change the fundamental assumption that setting x3..x12 to mid (0.5) places the points on the relevant front.
            # The previous program did this and worked.
            
            # So, the improvement should focus on:
            # 1. Better sampling of the 2D parameter space (x1, x2) to maximize hypervolume.
            # 2. Ensuring the points are well-distributed.
            
            # The current mapping:
            # u -> x1 = (2/pi) * arccos(1-u)
            # v -> x2 = v
            # This assumes x1, x2 are angles?
            # If f1 = cos(x1), f2 = sin(x1)cos(x2), f3 = sin(x1)sin(x2)
            # Then x1, x2 are the angles.
            # The range of x1 should be [0, pi/2] for the first octant.
            # But the bounds are [0,1].
            # So x1 is in [0,1] radians.
            # This covers only a part of the sphere!
            # pi/2 is ~1.57.
            # So x1 in [0,1] covers [0, 1 rad].
            # This is NOT the full quarter-sphere.
            
            # If the problem expects x1, x2 to be in [0, pi/2], but the bounds are [0,1], then we are missing the part where x1 > 1.
            # However, the previous program scored well.
            # Maybe the `problem` function scales x1, x2 internally?
            # Or maybe the "known Pareto front" is also restricted to this domain?
            
            # Let's look at Inspiration 1: "DTLZ2 front is a quarter-sphere; sample it surface-area-uniform."
            # Inspiration 2: "uniform grid in the transformed angle space ... ensures surface-area uniformity"
            
            # If the previous program used x1 = (2/pi)*arccos(1-u), it maps [0,1] to [0,1].
            # If the true angle is theta, and x1 = theta, then we are only sampling theta in [0,1].
            # To sample the full [0, pi/2], we should map u in [0,1] to theta in [0, pi/2].
            # theta = u * pi/2.
            
            # Let's check the mapping in the previous program again.
            # x1 = (2.0 / np.pi) * np.arccos(1.0 - u)
            # If u=0, x1=0.
            # If u=1, x1=2/pi * pi/2 = 1.
            # So x1 is in [0,1].
            
            # If the DTLZ2 implementation uses x1 directly as the angle, we are missing [1, pi/2].
            # If the DTLZ2 implementation scales x1 to [0, pi/2] internally (e.g. x1 * pi/2), then we are fine.
            
            # Given the uncertainty, and the instruction to "KEEP this design", I should probably stick to the variable ranges that worked before.
            # However, "investigate improvements within its family".
            
            # One improvement: Use Sobol sequence instead of Halton for better uniformity.
            # Another: Check if we can use the full range [0, pi/2] for the angles.
            # If the bounds are [0,1], and the problem clips, maybe we should map [0,1] to [0, pi/2] to cover the full front?
            # If the previous program only covered [0,1] radians and still scored 1.11, maybe the rest of the front is less important or the problem scales it.
            
            # Let's try to improve the sampling quality.
            # Sobol is generally better than Halton for low dimensions.
            
            # Also, ensure we don't exceed the budget.
            
            # Let's switch to Sobol if available, else Halton.
            
            n_points = self.budget
            
            try:
                from scipy.stats import qmc
                sampler = qmc.Sobol(d=2, scramble=True)
                uv = sampler.random(n_points)
                u = uv[:, 0]
                v = uv[:, 1]
            except ImportError:
                def halton(index, base):
                    f = 1.0
                    r = 0.0
                    while index > 0:
                        f = f / base
                        r = r + f * (index % base)
                        index //= base
                    return r
                u = np.array([halton(i+1, 2) for i in range(n_points)])
                v = np.array([halton(i+1, 3) for i in range(n_points)])
            
            # Mapping:
            # Previous: x1 = (2/pi) * arccos(1-u), x2 = v
            # This implies x1, x2 are in [0,1].
            
            # Let's stick to this mapping to be safe, but ensure the distribution is good.
            
            x1 = (2.0 / np.pi) * np.arccos(1.0 - u)
            x2 = v
            
            X = np.zeros((n_points, self.dim))
            X[:, 0] = x1
            X[:, 1] = x2
            for j in range(2, self.dim):
                X[:, j] = mid[j]
                
            # Evaluate
            F = np.array([problem(x) for x in X])
            
            if time.time() > deadline:
                pass
                
            return X
