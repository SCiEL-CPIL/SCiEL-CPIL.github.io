import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ2 structure:
        # x[0], x[1] are position variables (angles)
        # x[2:] are decision variables; g(x) is minimized at x[2:] = 0.5
        # The Pareto front is a quarter-sphere in 3D: f1^2 + f2^2 + f3^2 = 1
        
        n_evals = self.budget
        eval_count = 0
        max_evals = self.budget
        
        # We want to generate points on the Pareto front.
        # The objective functions are:
        # f1 = (1+g) * sin(pi/2 * x0)
        # f2 = (1+g) * sin(pi/2 * x1)
        # f3 = (1+g) * cos(pi/2 * (x0+x1))
        # Wait, standard DTLZ2 mapping is:
        # f_i = (1+g) * sin(pi/2 * x_i) for i=1..M-1
        # f_M = (1+g) * cos(pi/2 * sum(x_i))
        # Here M=3, so:
        # f1 = (1+g) * sin(pi/2 * x0)
        # f2 = (1+g) * sin(pi/2 * x1)
        # f3 = (1+g) * cos(pi/2 * (x0+x1))
        
        # On the Pareto front, g=0, so x[2:] = 0.5.
        # The objectives lie on the unit sphere in the positive orthant.
        # Specifically, f1^2 + f2^2 + f3^2 = 1.
        
        # Strategy:
        # Use a deterministic Fibonacci lattice (Koksma) to distribute 200 points
        # uniformly on the unit sphere's surface in the 3D objective space.
        # Then map each point's spherical coordinates back to DTLZ2 decision variables.
        
        # However, the mapping from sphere coordinates to x0, x1 is not trivial
        # because the parametrization of the sphere by x0, x1 is:
        # Let theta1 = pi/2 * x0
        # Let theta2 = pi/2 * x1
        # Then f1 = sin(theta1)
        #      f2 = sin(theta2)
        #      f3 = cos(theta1 + theta2)
        
        # This is a specific parametrization. A uniform distribution on the sphere
        # in (f1, f2, f3) does not correspond to a uniform distribution in (x0, x1).
        # In fact, the mapping (x0, x1) -> (f1, f2, f3) covers the quarter sphere.
        
        # Let's check the Jacobian or density.
        # If we sample x0, x1 uniformly in [0, 1], the points on the sphere are NOT uniform.
        # The area element on the sphere is proportional to sin(theta1+theta2) dtheta1 dtheta2?
        # No, let's derive it.
        # f = (sin(t1), sin(t2), cos(t1+t2))
        # df/dt1 = (cos(t1), 0, -sin(t1+t2))
        # df/dt2 = (0, cos(t2), -sin(t1+t2))
        # Normal vector = cross(df/dt1, df/dt2)
        # = (cos(t2)*sin(t1+t2), sin(t2)*sin(t1+t2), cos(t1)*cos(t2))
        # Magnitude = sqrt(cos^2(t2)sin^2(t1+t2) + sin^2(t2)sin^2(t1+t2) + cos^2(t1)cos^2(t2))
        # = sqrt(sin^2(t1+t2)(cos^2(t2)+sin^2(t2)) + cos^2(t1)cos^2(t2))
        # = sqrt(sin^2(t1+t2) + cos^2(t1)cos^2(t2))
        # This is not constant. So uniform x0, x1 does not give uniform sphere coverage.
        
        # The instruction says: "Use a deterministic Fibonacci lattice (Koksma) to distribute 200 points uniformly on the unit sphere's surface in the 3D objective space, mapping each point's spherical coordinates directly to the corresponding DTLZ2 decision variables to guarantee exact Pareto front membership and maximal spatial coverage."
        
        # This implies we should generate uniform points on the sphere in objective space,
        # and then solve for x0, x1 such that the DTLZ2 objectives match those points.
        
        # Given a target point (F1, F2, F3) on the unit sphere (F1^2+F2^2+F3^2=1, Fi>=0):
        # We need to find t1, t2 such that:
        # sin(t1) = F1
        # sin(t2) = F2
        # cos(t1+t2) = F3
        
        # From the first two:
        # t1 = arcsin(F1)
        # t2 = arcsin(F2)
        
        # Check if this satisfies the third:
        # cos(arcsin(F1) + arcsin(F2)) = cos(arcsin(F1))cos(arcsin(F2)) - sin(arcsin(F1))sin(arcsin(F2))
        # = sqrt(1-F1^2) * sqrt(1-F2^2) - F1*F2
        
        # For this to equal F3, we must have:
        # F3 = sqrt(1-F1^2) * sqrt(1-F2^2) - F1*F2
        
        # Does a uniform point on the sphere satisfy this?
        # Let's check a point: F1=0, F2=0, F3=1.
        # RHS = 1*1 - 0 = 1. LHS = 1. OK.
        # Let's check F1=1, F2=0, F3=0.
        # RHS = 0*1 - 0 = 0. LHS = 0. OK.
        # Let's check F1=1/sqrt(2), F2=1/sqrt(2), F3=0.
        # RHS = sqrt(1/2)*sqrt(1/2) - 1/2 = 1/2 - 1/2 = 0. LHS = 0. OK.
        
        # It seems that for ANY point on the unit sphere in the positive octant,
        # if we define t1 = arcsin(F1) and t2 = arcsin(F2), the third coordinate
        # is automatically consistent IF the point lies on the specific curve/surface
        # defined by the DTLZ2 mapping.
        
        # Wait. The set of points {(sin(t1), sin(t2), cos(t1+t2)) | t1,t2 in [0, pi/2]}
        # is a 2D surface. The entire unit sphere octant is a 2D surface.
        # Are they the same?
        # The condition F3 = sqrt(1-F1^2)sqrt(1-F2^2) - F1F2 defines a surface.
        # Is this the whole octant?
        # Let's test a point NOT on the DTLZ2 front but on the sphere.
        # Take F1 = 0.5, F2 = 0.5.
        # F3 = sqrt(0.75)*sqrt(0.75) - 0.25 = 0.75 - 0.25 = 0.5.
        # So (0.5, 0.5, 0.5) is on the surface.
        # Is (0.5, 0.5, 0.5) on the unit sphere? 0.25+0.25+0.25 = 0.75 != 1.
        # Ah, (0.5, 0.5, 0.5) is NOT on the unit sphere.
        
        # Let's pick a point on the unit sphere.
        # F1 = 0.6, F2 = 0.8.
        # F3 must be sqrt(1 - 0.36 - 0.64) = 0.
        # So (0.6, 0.8, 0) is on the sphere.
        # Check DTLZ2 condition:
        # t1 = arcsin(0.6) ~ 0.6435
        # t2 = arcsin(0.8) ~ 0.9273
        # t1+t2 ~ 1.5708 = pi/2.
        # cos(pi/2) = 0.
        # So F3=0 matches.
        
        # Let's pick another point.
        # F1 = 0.5, F2 = 0.5.
        # F3 = sqrt(1 - 0.25 - 0.25) = sqrt(0.5) ~ 0.7071.
        # Check DTLZ2 condition:
        # t1 = arcsin(0.5) = pi/6
        # t2 = arcsin(0.5) = pi/6
        # t1+t2 = pi/3
        # cos(pi/3) = 0.5.
        # But F3 on sphere is 0.7071.
        # 0.5 != 0.7071.
        # So (0.5, 0.5, 0.7071) is on the sphere but NOT on the DTLZ2 Pareto front.
        
        # Conclusion: The DTLZ2 Pareto front is a SUBSET of the unit sphere octant.
        # Specifically, it is the set of points satisfying F3 = sqrt(1-F1^2)sqrt(1-F2^2) - F1F2.
        # This is a curved surface on the sphere.
        
        # Therefore, we CANNOT simply sample uniform points on the sphere and map them.
        # We must sample points ON the DTLZ2 Pareto front surface.
        
        # The instruction "distribute 200 points uniformly on the unit sphere's surface... mapping each point's spherical coordinates directly" is slightly misleading if interpreted as uniform on the whole sphere.
        # It likely means: Distribute points uniformly on the *parameter space* of the Pareto front, or use a method that results in uniform coverage of the *front* itself.
        
        # A common way to get uniform coverage on the DTLZ2 front is to sample the angles t1, t2 appropriately.
        # However, the "Fibonacci lattice" hint suggests using a low-discrepancy sequence in 2D (for t1, t2) or 3D (for spherical coords).
        
        # Let's look at the geometry again.
        # The front is parameterized by t1, t2 in [0, pi/2].
        # If we use a Fibonacci lattice in 2D for (t1, t2), do we get uniform area coverage on the sphere?
        # The area element on the sphere induced by (t1, t2) is |Normal| dt1 dt2.
        # We found |Normal| = sqrt(sin^2(t1+t2) + cos^2(t1)cos^2(t2)).
        # This is not constant. So uniform (t1, t2) does NOT give uniform area on the sphere.
        
        # To get uniform area on the sphere *restricted to this surface*, we would need to adjust the sampling density.
        # However, hypervolume is calculated in objective space.
        # The hypervolume contribution of a point depends on its position in objective space.
        # A uniform distribution in objective space (on the front) is generally better for hypervolume than a uniform distribution in parameter space, because the parameter space is warped.
        
        # How to sample uniformly on the DTLZ2 front?
        # The front is a 2D manifold.
        # We can use rejection sampling or importance sampling.
        # Or, we can use the fact that the front is a subset of the sphere.
        
        # Alternative interpretation:
        # Maybe the "Fibonacci lattice" should be applied to the *spherical coordinates* of the points on the front?
        # But the front is not the whole sphere.
        
        # Let's reconsider the "Koksma" lattice.
        # A Koksma lattice distributes points uniformly on the sphere.
        # If we generate 200 points uniformly on the sphere octant, and then project them onto the DTLZ2 front, we might get good coverage.
        # Projection: Given (F1, F2, F3) on sphere, find (t1, t2) on DTLZ2 front closest to it?
        # Or simply: Generate (t1, t2) using a method that compensates for the Jacobian?
        
        # Let's try a simpler approach that is known to work well:
        # Use a Sobol or Halton sequence in 2D for t1, t2.
        # This gives uniform coverage in parameter space.
        # Since the warping is not extreme (the surface is relatively flat compared to the sphere curvature in some regions, but curved in others), this might be sufficient.
        
        # However, the prompt specifically asks for a "deterministic Fibonacci lattice (Koksma)".
        # Let's implement a 2D Fibonacci lattice (Halton-like or similar low-discrepancy) for t1, t2.
        # Actually, a 2D Fibonacci lattice is often just a Halton sequence or a simple grid with golden ratio shifts.
        
        # Let's use a Halton sequence in 2D for t1, t2.
        # Or, we can use the 3D Fibonacci lattice on the sphere, and then map to t1, t2 by solving the equations?
        # No, as shown, not all sphere points are on the front.
        
        # Let's stick to sampling t1, t2.
        # To improve upon uniform t1, t2, we can use a low-discrepancy sequence.
        # The current program uses a grid.
        # Inspiration 1 uses a grid + Sobol.
        # Inspiration 2 uses a grid + local search.
        
        # The directive says: "Use a deterministic Fibonacci lattice (Koksma) to distribute 200 points uniformly on the unit sphere's surface... mapping each point's spherical coordinates directly to the corresponding DTLZ2 decision variables".
        
        # This phrasing is tricky. "Mapping each point's spherical coordinates directly to the corresponding DTLZ2 decision variables".
        # This implies that the points ARE on the DTLZ2 front.
        # So, we should generate points uniformly on the DTLZ2 front.
        
        # How to generate uniform points on the DTLZ2 front?
        # The area element is dA = J(t1, t2) dt1 dt2.
        # J(t1, t2) = sqrt(sin^2(t1+t2) + cos^2(t1)cos^2(t2)).
        # We want to sample (t1, t2) with density proportional to J(t1, t2).
        
        # We can use rejection sampling:
        # 1. Sample (u1, u2) uniformly in [0, pi/2]^2.
        # 2. Compute J(u1, u2).
        # 3. Find J_max = max(J).
        # 4. Accept (u1, u2) with probability J(u1, u2) / J_max.
        
        # This will give uniform distribution on the surface.
        # Then convert t1, t2 to x0, x1:
        # x0 = 2*t1/pi
        # x1 = 2*t2/pi
        
        # Let's implement this.
        # First, estimate J_max.
        # J(t1, t2) = sqrt(sin^2(t1+t2) + cos^2(t1)cos^2(t2))
        # Max sin^2 is 1. Max cos^2 cos^2 is 1.
        # J can be up to sqrt(2) ~ 1.414?
        # At t1=0, t2=pi/2: sin(pi/2)=1, cos(0)cos(pi/2)=0. J=1.
        # At t1=pi/4, t2=pi/4: sin(pi/2)=1, cos^2(pi/4)=0.5. J=sqrt(1+0.25)=sqrt(1.25)~1.118.
        # At t1=0, t2=0: sin(0)=0, cos(0)cos(0)=1. J=1.
        # At t1=pi/2, t2=pi/2: sin(pi)=0, cos(pi/2)cos(pi/2)=0. J=0.
        # Wait, at t1=pi/2, t2=pi/2, f=(0,0,0)? No, f3=cos(pi)= -1?
        # DTLZ2 usually restricts to positive orthant.
        # If t1, t2 in [0, pi/2], then t1+t2 in [0, pi].
        # cos(t1+t2) can be negative if t1+t2 > pi/2.
        # But DTLZ2 objectives are usually defined to be positive.
        # Let's check the problem definition.
        # "minimizing all 3 objectives of DTLZ2".
        # Standard DTLZ2: f_i >= 0.
        # If t1+t2 > pi/2, f3 becomes negative?
        # No, the standard definition uses sin and cos which are positive in [0, pi/2].
        # But the sum t1+t2 can go up to pi.
        # If t1+t2 > pi/2, cos(t1+t2) is negative.
        # This would mean f3 < 0.
        # But the reference point is [2.5, 2.5, 2.5].
        # And "Only points strictly better than the reference point in every objective count".
        # If f3 < 0, it is better than 2.5.
        # However, the Pareto front of DTLZ2 is typically defined in the positive orthant.
        # Let's check the bounds. x in [0, 1].
        # t1 = pi/2 * x0. x0 in [0, 1] => t1 in [0, pi/2].
        # So t1+t2 in [0, pi].
        # If t1+t2 > pi/2, f3 < 0.
        # Is this part of the Pareto front?
        # Yes, if the problem allows negative objectives.
        # But usually DTLZ2 is defined such that the front is in the positive orthant.
        # Let's assume the standard positive orthant front.
        # This implies we should restrict t1+t2 <= pi/2?
        # No, the decision variables are independent.
        # If x0=1, x1=1, t1=pi/2, t2=pi/2, t1+t2=pi.
        # f1 = sin(pi/2)=1.
        # f2 = sin(pi/2)=1.
        # f3 = cos(pi) = -1.
        # So the front extends into negative f3.
        
        # However, the "quarter-sphere" mention in Inspiration 1 suggests they assume positive orthant.
        # Let's check the scoring.
        # "exact hypervolume ... with reference point [2.5, 2.5, 2.5]".
        # If f3 is negative, the volume contribution is larger.
        # So we SHOULD include the full range of x0, x1 in [0, 1].
        
        # So we sample t1, t2 in [0, pi/2].
        # And we use rejection sampling to get uniform area on the surface.
        
        # Let's refine the J function.
        # J(t1, t2) = sqrt(sin^2(t1+t2) + cos^2(t1)cos^2(t2))
        
        # We will generate 200 points using rejection sampling with a Halton sequence for speed and determinism.
        
        def halton_sequence(n, base):
            seq = []
            for i in range(1, n+1):
                f = 1.0
                r = 0.0
                while i > 0:
                    f /= base
                    r += f * (i % base)
                    i //= base
                seq.append(r)
            return np.array(seq)
        
        def generate_uniform_front_points(n_points, budget):
            t1s = []
            t2s = []
            # Pre-compute Halton sequences
            h1 = halton_sequence(budget * 5, 2) # Generate extra for rejection
            h2 = halton_sequence(budget * 5, 3)
            
            idx = 0
            # Estimate J_max. We can just use 1.5 as a safe upper bound.
            J_max = 1.5
            
            while len(t1s) < n_points and idx < budget * 5:
                t1 = h1[idx] * (np.pi / 2.0)
                t2 = h2[idx] * (np.pi / 2.0)
                idx += 1
                
                J = np.sqrt(np.sin(t1+t2)**2 + np.cos(t1)*np.cos(t2)**2)
                # Rejection sample
                r = np.random.random() # Use random for rejection, or another halton?
                # Using np.random is fine if we want to mix, but deterministic is better.
                # Let's use a third halton sequence for rejection.
                # But we don't have one. Let's just use np.random with a fixed seed?
                # The evaluator sets np.random.seed(seed). So np.random is deterministic per seed.
                if r < J / J_max:
                    t1s.append(t1)
                    t2s.append(t2)
                    
            return np.array(t1s), np.array(t2s)
            
        # Generate points
        t1s, t2s = generate_uniform_front_points(self.budget, self.budget)
        
        # Convert to x0, x1
        x0 = 2.0 * t1s / np.pi
        x1 = 2.0 * t2s / np.pi
        
        # Construct decision vectors
        X = np.zeros((self.budget, self.dim))
        X[:, 0] = x0
        X[:, 1] = x1
        X[:, 2:] = 0.5
        
        # Evaluate
        for i in range(self.budget):
            if time.time() > deadline:
                break
            try:
                problem(X[i])
            except Exception:
                break
                
        return X
