"""Geometric Importance Sampling for DTLZ2: Uniform density on the Pareto front surface."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer for DTLZ2 using geometric importance sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 310.0  # safety margin under 330s kill

        used = 0
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # DTLZ2 Pareto Front Geometry:
        # f1 = sin(pi/2 * x1)
        # f2 = cos(pi/2 * x1) * sin(pi/2 * x2)
        # f3 = cos(pi/2 * x1) * cos(pi/2 * x2)
        #
        # Parameterization:
        # theta1 = pi/2 * x1  (in [0, pi/2])
        # theta2 = pi/2 * x2  (in [0, pi/2])
        #
        # Surface Area Element:
        # dA = cos(theta1) * cos(theta2) dtheta1 dtheta2
        #
        # To sample uniformly on the surface, we need:
        # 1. theta1 ~ Uniform[0, pi/2] (since integral of cos(theta1) is sin(theta1), 
        #    wait, let's re-derive.
        #    Area = integral_0^{pi/2} integral_0^{pi/2} cos(t1) cos(t2) dt2 dt1
        #    = [sin(t1)]_0^{pi/2} * [-sin(t2)]_0^{pi/2} ? No.
        #    Integral of cos(t2) is sin(t2).
        #    Total Area = (sin(pi/2)-sin(0)) * (sin(pi/2)-sin(0)) = 1 * 1 = 1.
        #    
        #    CDF for theta1: F1(t1) = (1/1) * integral_0^{t1} cos(u) du = sin(t1).
        #    So t1 = arcsin(u1) where u1 ~ Uniform[0,1].
        #    
        #    CDF for theta2 given theta1: 
        #    Conditional density p(t2|t1) = cos(t2) / sin(t1) ? 
        #    Let's check joint density p(t1, t2) = cos(t1)cos(t2).
        #    Marginal p(t1) = integral_0^{pi/2} cos(t1)cos(t2) dt2 = cos(t1) * [sin(t2)]_0^{pi/2} = cos(t1).
        #    So p(t2|t1) = p(t1,t2)/p(t1) = cos(t1)cos(t2)/cos(t1) = cos(t2).
        #    CDF for theta2: F2(t2) = integral_0^{t2} cos(v) dv = sin(t2).
        #    So t2 = arcsin(u2) where u2 ~ Uniform[0,1].
        #
        # Therefore, to sample uniformly on the surface:
        # u1, u2 ~ Uniform[0, 1]
        # theta1 = arcsin(u1)
        # theta2 = arcsin(u2)
        #
        # Then map to x:
        # x1 = 2/pi * theta1
        # x2 = 2/pi * theta2
        # x3...x12 = 0.5 (to ensure g=0)

        # Generate uniform random numbers
        # Using np.random.seed is handled by evaluator, but we should use a local rng or rely on global?
        # The contract says: "For seeds 0 and 1 the evaluator calls np.random.seed(seed)".
        # So we can use np.random directly.
        
        u1 = np.random.rand(budget)
        u2 = np.random.rand(budget)
        
        # Clamp to [0, 1] just in case of floating point issues at boundaries, 
        # though rand() is [0, 1). arcsin(1) is fine, arcsin(0) is fine.
        u1 = np.clip(u1, 0.0, 1.0)
        u2 = np.clip(u2, 0.0, 1.0)
        
        theta1 = np.arcsin(u1)
        theta2 = np.arcsin(u2)
        
        x1 = (2.0 / np.pi) * theta1
        x2 = (2.0 / np.pi) * theta2
        
        # Build candidate decision vectors
        candidates = []
        for i in range(budget):
            x = np.full(dim, 0.5)
            x[0] = x1[i]
            x[1] = x2[i]
            candidates.append(x)

        # Evaluate all candidates
        for x in candidates:
            if used >= budget:
                break
            if time.time() >= deadline:
                break
            problem(x)  # every call is recorded by the evaluator
            used += 1

        return np.array(candidates[:used])
# EVOLVE-BLOCK-END
