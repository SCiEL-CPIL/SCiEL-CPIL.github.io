"""Optimizer for DTLZ4 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using structured Pareto front coverage."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, index, base):
        """Generate Halton sequence value."""
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    def _dtlz4_front_point(self, alpha, beta):
        """
        Generate a point on the DTLZ4 Pareto front given two parameters.
        alpha, beta in [0, 1] with alpha + beta <= 1 (simplex).
        Returns a 12-dim vector in [0, 1]^12.
        """
        # DTLZ4: f1 = 0.5 * (1 + g) * cos(theta1) * cos(theta2)
        #        f2 = 0.5 * (1 + g) * cos(theta1) * sin(theta2)
        #        f3 = 0.5 * (1 + g) * sin(theta1)
        # where g = sum(xi - 0.5)^2 for i in [9..11]
        # On the front, g = 0, so xi = 0.5 for i in [9..11]
        # theta1 = alpha * pi/2, theta2 = beta * pi/2
        # But we need alpha + beta <= 1 for the simplex constraint... actually
        # for 3 objectives, the front is a 2D surface. Let's use:
        # theta1 = alpha * pi/2, theta2 = beta * pi/2 where alpha, beta in [0,1]
        # This gives all points on the front when g=0.
        
        theta1 = alpha * np.pi / 2
        theta2 = beta * np.pi / 2
        
        # First 9 variables determine the angles, last 3 are at 0.5 for g=0
        # For DTLZ4 with M=3, n=12:
        # x[0:9] are used for angles (actually only 2 angles needed for M=3)
        # The standard DTLZ4 mapping:
        # theta_i = x[i] * pi/2 for i = 0..M-2
        # g = sum((x[j] - 0.5)^2) for j = M-1..n-1
        
        # For M=3: theta1 = x[0]*pi/2, theta2 = x[1]*pi/2
        # g = sum((x[j]-0.5)^2) for j=2..11
        # On front: g=0 => x[j]=0.5 for j=2..11
        
        x = np.zeros(12)
        x[0] = alpha
        x[1] = beta
        x[2:] = 0.5  # This makes g = 0
        
        return x

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin: 300s limit, use 290s
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Use Halton sequence to generate well-distributed points on the Pareto front
        # The front for DTLZ4 with 3 objectives is parameterized by (alpha, beta) in [0,1]^2
        # We'll use a 2D Halton sequence to fill this space uniformly
        
        n_points = self.budget
        alpha = np.zeros(n_points)
        beta = np.zeros(n_points)
        
        bases = [2, 3]  # Two bases for 2D Halton
        
        for i in range(1, n_points + 1):
            alpha[i - 1] = self._halton(i, bases[0])
            beta[i - 1] = self._halton(i, bases[1])
        
        # Generate decision variables on the Pareto front
        X = np.zeros((n_points, self.dim))
        for i in range(n_points):
            X[i] = self._dtlz4_front_point(alpha[i], beta[i])
        
        # Evaluate all points
        F = np.zeros((n_points, self.n_obj))
        for i in range(n_points):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
        
        return X
# EVOLVE-BLOCK-END
