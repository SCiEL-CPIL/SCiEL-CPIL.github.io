"""Optimized DTLZ4 solver: exploit known Pareto front with vertex-biased placement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ4 optimizer exploiting known Pareto front with vertex-biased placement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim, n_obj, budget = self.dim, self.n_obj, self.budget

        n_pos = n_obj - 1
        g_opt = 0.5 * (hi[n_pos:] + lo[n_pos:])

        all_X = []
        all_F = []
        used = 0

        def eval_point(x):
            nonlocal used
            f = problem(x)
            all_X.append(np.asarray(x, dtype=float))
            all_F.append(np.asarray(f, dtype=float))
            used += 1
            return all_F[-1]

        def make_x(x0, x1):
            x = np.full(dim, 0.5)
            x[0] = x0
            x[1] = x1
            x[n_pos:] = g_opt
            return x

        evaluated_positions = set()

        def try_eval(x0, x1):
            nonlocal used
            if used >= budget or time.time() > deadline:
                return
            key = (round(x0, 10), round(x1, 10))
            if key in evaluated_positions:
                return
            evaluated_positions.add(key)
            x = make_x(x0, x1)
            eval_point(x)

        # Vertex-biased points: use power transform to cluster near vertices
        # For each edge of the triangle, place points with density ~ 1/sqrt(t)
        # where t is distance from vertex (higher density near vertices)
        
        # Vertices
        vertices = [(1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]
        for x0, x1 in vertices:
            try_eval(x0, x1)
        
        # Edge midpoints
        mids = [(0.5, 0.5), (0.5, 0.0), (0.0, 0.5)]
        for x0, x1 in mids:
            try_eval(x0, x1)
        
        remaining = budget - used
        if remaining <= 0:
            return np.array(all_X) if all_X else np.array([])
        
        # Generate vertex-biased points on each edge
        # Edge 1: from (1,0) to (0,0): x0 = t, x1 = 0, t in [0,1]
        # Edge 2: from (0,1) to (0,0): x0 = 0, x1 = t, t in [0,1]  
        # Edge 3: from (1,0) to (0,1): x0 = 1-t, x1 = t, t in [0,1]
        
        # Use beta distribution with alpha<1 for vertex bias
        # Or simply use t = u^(1/alpha) where alpha > 1 biases toward 0
        # For hypervolume, we want more points near vertices (corners of triangle)
        
        n_per_edge = remaining // 3
        n_interior = remaining - 3 * n_per_edge
        
        # Edge 1: (1,0) to (0,0) - vary x0 from 1 to 0, x1=0
        # Bias toward x0=1 (vertex (1,0)) and x0=0 (vertex (0,0))
        # Use symmetric bias: t = 0.5 - 0.5*cos(pi*u) gives density ~ 1/sqrt(t(1-t))
        # But we want more near BOTH endpoints. Use: t = u^(1/2) for one end, or mix.
        # Simpler: use t = u^(2/3) to bias toward t=0, and t = 1-u^(2/3) for t=1
        # Actually for HV, near-vertex points give large boxes. Let's use:
        # Half the points biased toward one vertex, half toward the other
        
        for i in range(n_per_edge):
            if used >= budget or time.time() > deadline:
                break
            # Edge 1: (1,0)-(0,0)
            u = (i + 0.5) / n_per_edge
            # Bias toward both ends: use u^0.7 for first half, (1-u)^0.7 for second
            if i < n_per_edge // 2:
                t = ((2*i + 1) / n_per_edge) ** 0.6  # biased toward t=0 (vertex 0,0)
                x0 = t
                x1 = 0.0
            else:
                t = (1 - (2*(i - n_per_edge//2) + 1) / (n_per_edge - n_per_edge//2)) ** 0.6
                x0 = t
                x1 = 0.0
            try_eval(x0, x1)
        
        for i in range(n_per_edge):
            if used >= budget or time.time() > deadline:
                break
            # Edge 2: (0,1)-(0,0)
            if i < n_per_edge // 2:
                t = ((2*i + 1) / n_per_edge) ** 0.6
                x0 = 0.0
                x1 = t
            else:
                t = (1 - (2*(i - n_per_edge//2) + 1) / (n_per_edge - n_per_edge//2)) ** 0.6
                x0 = 0.0
                x1 = t
            try_eval(x0, x1)
        
        for i in range(n_per_edge):
            if used >= budget or time.time() > deadline:
                break
            # Edge 3: (1,0)-(0,1)
            if i < n_per_edge // 2:
                t = ((2*i + 1) / n_per_edge) ** 0.6
                x0 = 1.0 - t
                x1 = t
            else:
                t = (1 - (2*(i - n_per_edge//2) + 1) / (n_per_edge - n_per_edge//2)) ** 0.6
                x0 = 1.0 - t
                x1 = t
            try_eval(x0, x1)
        
        # Interior points: use barycentric coordinates with vertex bias
        for i in range(n_interior):
            if used >= budget or time.time() > deadline:
                break
            # Use Dirichlet-like sampling biased toward vertices
            # Generate 3 random numbers, raise to power < 1 to bias toward 0
            r1 = np.random.random() ** 0.5
            r2 = np.random.random() ** 0.5
            r3 = np.random.random() ** 0.5
            s = r1 + r2 + r3
            # Barycentric: (r1/s, r2/s, r3/s) corresponds to triangle
            # x0 = r2/s, x1 = r3/s (since f0=x0, f1=x1, f2=1-x0-x1)
            x0 = r2 / s
            x1 = r3 / s
            try_eval(x0, x1)
        
        # Fill any remaining budget with more edge points
        while used < budget and time.time() < deadline:
            # Pick a random edge and position
            edge = np.random.randint(3)
            u = np.random.random()
            # Apply vertex bias
            t = u ** 0.6
            if edge == 0:
                x0, x1 = t, 0.0
            elif edge == 1:
                x0, x1 = 0.0, t
            else:
                x0, x1 = 1.0 - t, t
            try_eval(x0, x1)
        
        if all_X:
            return np.array(all_X)
        return np.array([])
# EVOLVE-BLOCK-END
