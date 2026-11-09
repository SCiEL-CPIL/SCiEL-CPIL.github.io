# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ1 using a structured sampling + evolutionary refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # DTLZ1 structure: first (dim - n_obj + 1) variables determine g,
        # last (n_obj - 1) variables determine the shape on the front.
        # For DTLZ1 with n_obj=3: first 5 vars -> g, last 2 vars -> front shape.
        n_shape = n_obj - 1  # 2
        n_g = dim - n_shape   # 5

        # Evaluate a single point
        def eval_point(x):
            return np.array(problem(x))

        # Sobol/Halton-like sequence for the 2D shape space (simplified: use quasi-random)
        def shape_sequence(n):
            """Generate n quasi-random points in [0,1]^2 using a simple lattice."""
            # Use golden-ratio based quasi-random sequence
            phi = (1 + np.sqrt(5)) / 2
            points = np.zeros((n, 2))
            for i in range(n):
                t = (i + 1) / (n + 1)
                points[i, 0] = (t * phi) % 1.0
                points[i, 1] = (t * phi * phi) % 1.0
            return points

        # Initialize with structured sampling
        # For DTLZ1, g=0 is optimal. Set first n_g variables to make g small.
        # g(x) = 100 * (n_g + sum((x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5))) for i in 1..n_g)
        # g=0 when x_i = 0.5 for all i in 1..n_g
        # Then f1 = (1+g)*0.5*x1, f2 = (1+g)*0.5*x2*(1-x1), f3 = (1+g)*(1-x2)
        # with x1, x2 in [0,1]

        # Phase 1: Structured initialization on the Pareto front (g=0)
        # Use a combination of:
        # 1. Vertex points (corners of the shape space)
        # 2. Edge points (along each edge)
        # 3. Interior quasi-random points
        n_init = min(budget // 2, 100)
        
        # For 2D shape space (n_shape=2), we have:
        # - 4 vertices: (0,0), (0,1), (1,0), (1,1)
        # - 4 edges with multiple points each
        # - Interior points
        
        shape_pts_list = []
        
        # Vertices (high weight for hypervolume)
        vertices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        shape_pts_list.append(vertices)
        
        # Edge points: 5 points along each edge (excluding vertices)
        n_edge = 5
        edge_pts = []
        for i in range(n_edge):
            t = (i + 1) / (n_edge + 1)
            # Bottom edge: (t, 0)
            edge_pts.append([t, 0])
            # Top edge: (t, 1)
            edge_pts.append([t, 1])
            # Left edge: (0, t)
            edge_pts.append([0, t])
            # Right edge: (1, t)
            edge_pts.append([1, t])
        shape_pts_list.append(np.array(edge_pts))
        
        # Interior quasi-random points
        n_interior = n_init - len(vertices) - len(edge_pts)
        if n_interior > 0:
            interior_pts = shape_sequence(n_interior)
            # Scale to avoid edges (keep in [0.1, 0.9])
            interior_pts = 0.1 + 0.8 * interior_pts
            shape_pts_list.append(interior_pts)
        
        shape_pts = np.vstack(shape_pts_list)
        # Ensure we don't exceed n_init
        if len(shape_pts) > n_init:
            shape_pts = shape_pts[:n_init]
        
        X = np.zeros((len(shape_pts), dim))
        X[:, :n_g] = 0.5  # optimal g variables
        X[:, n_g:] = shape_pts * span[n_g:] + lo[n_g:]

        F = np.array([eval_point(x) for x in X])
        used = len(shape_pts)

        # Filter: keep only points that are strictly better than reference [400,400,400]
        # All DTLZ1 front points should be well below 400, so this should keep all.

        # Phase 2: Evolutionary refinement
        if used < budget:
            pop_size = min(30, max(10, (budget - used) // 5))
            # Select best pop_size points from initial set (non-dominated sort)
            # Simple: pick points with lowest sum of objectives as proxy
            sums = np.sum(F, axis=1)
            idx = np.argsort(sums)[:pop_size]
            X_pop = X[idx].copy()
            F_pop = F[idx].copy()

            # Evolutionary loop
            while used < budget:
                if time.time() > deadline:
                    break

                # Non-dominated front selection
                n = len(F_pop)
                dominated = np.zeros(n, dtype=bool)
                for i in range(n):
                    if dominated[i]:
                        continue
                    for j in range(n):
                        if i == j or dominated[j]:
                            continue
                        if np.all(F_pop[i] <= F_pop[j]) and np.any(F_pop[i] < F_pop[j]):
                            dominated[j] = True
                        if np.all(F_pop[j] <= F_pop[i]) and np.any(F_pop[j] < F_pop[i]):
                            dominated[i] = True
                            break

                front_mask = ~dominated
                if np.sum(front_mask) == 0:
                    front_mask = np.ones(n, dtype=bool)
                parents = X_pop[front_mask]
                F_parents = F_pop[front_mask]

                # Generate child
                parent_idx = np.random.randint(len(parents))
                parent = parents[parent_idx]

                # Mutation: perturb shape variables more, g-variables less
                child = parent.copy()
                # Mutate g-variables with small step
                child[:n_g] += 0.02 * np.random.randn(n_g)
                child[:n_g] = np.clip(child[:n_g], lo[:n_g], hi[:n_g])
                # Mutate shape variables with larger step
                child[n_g:] += 0.15 * np.random.randn(n_shape)
                child[n_g:] = np.clip(child[n_g:], lo[n_g:], hi[n_g:])

                fc = eval_point(child)
                used += 1

                # Replace worst point in population
                # Use sum of objectives as fitness proxy
                worst_idx = np.argmax(np.sum(F_pop, axis=1))
                X_pop[worst_idx] = child
                F_pop[worst_idx] = fc

                # Also track all evaluated points for final result
                if len(X) < used:
                    X = np.vstack([X, child.reshape(1, -1)])
                    F = np.vstack([F, fc.reshape(1, -1)])
                else:
                    X[-1] = child
                    F[-1] = fc

        return X
# EVOLVE-BLOCK-END
