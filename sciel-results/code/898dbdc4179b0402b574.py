"""k-NN Surrogate-Guided Search with Nelder-Mead Polish for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """k-NN Surrogate-Guided Search with Nelder-Mead Polish for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        calls_made = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        evaluated_points = [] # List of (x, f)

        def count_call():
            nonlocal calls_made
            calls_made += 1

        def evaluate(x):
            nonlocal best_x, best_f
            x = np.clip(x, -5, 5)
            f = float(func(x))
            count_call()
            if f < best_f:
                best_f = f
                best_x = x.copy()
            evaluated_points.append((x, f))
            return f

        # Phase 1: Diverse Initialization using Halton sequence
        n_init = 15
        init_results = []

        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        bases = [2, 3, 5, 7, 11]
        for i in range(1, n_init + 1):
            point = []
            for d in range(dim):
                point.append(halton(i, bases[d]))
            point = np.array(point) * 10.0 - 5.0
            point += rng.uniform(-0.1, 0.1, dim)
            point = np.clip(point, -5, 5)

            if time.time() > deadline or calls_made >= budget:
                break
            f = evaluate(point)
            init_results.append((point, f))

        if calls_made >= budget:
            return best_x, best_f

        # Phase 2: k-NN Surrogate-Guided Search
        # We use a surrogate model based on k-Nearest Neighbors
        # to balance exploration (distance to existing points) and 
        # exploitation (predicted low function value).
        
        k_nn = 5
        epsilon = 0.6 # Start with high exploration
        epsilon_decay = 0.99
        min_epsilon = 0.05
        max_surrogate_steps = budget - calls_made - 50 # Reserve 50 for polish
        
        # Pre-allocate arrays for efficiency if needed, but list is fine for N<500
        for step in range(max_surrogate_steps):
            if time.time() > deadline or calls_made >= budget:
                break
            
            # Generate candidate points
            # 1. Random candidates
            candidates = []
            # 50 random points
            rand_cands = rng.uniform(-5, 5, (50, dim))
            candidates.extend([c for c in rand_cands])
            
            # 2. Local candidates around best points
            # Sort evaluated points by fitness
            sorted_points = sorted(evaluated_points, key=lambda x: x[1])
            top_k = min(3, len(sorted_points))
            for i in range(top_k):
                x_best, f_best = sorted_points[i]
                # Generate 20 points around each of the top 3
                local_cands = x_best + rng.normal(0, 0.5, (20, dim))
                local_cands = np.clip(local_cands, -5, 5)
                candidates.extend([c for c in local_cands])
            
            # 3. Candidates far from existing points (for exploration)
            # We can just rely on the random candidates for exploration 
            # if we pick based on distance.
            
            if not candidates:
                break
            
            candidates = np.array(candidates)
            
            # Evaluate candidates using k-NN surrogate
            # For each candidate, find k nearest neighbors in evaluated_points
            # and compute mean f and min distance.
            
            # Optimization: If we have many candidates, this is O(N_cands * N_eval * dim)
            # With N_cands ~ 110 and N_eval < 500, this is ~ 50k * 5 = 250k ops. Very fast.
            
            pred_values = np.zeros(len(candidates))
            min_dists = np.zeros(len(candidates))
            
            for i, cand in enumerate(candidates):
                # Compute distances from cand to all evaluated points
                dists = np.linalg.norm(np.array([p[0] for p in evaluated_points]) - cand, axis=1)
                # Find k nearest
                k_idx = np.argsort(dists)[:k_nn]
                min_dists[i] = dists[k_idx[0]]
                pred_values[i] = np.mean([evaluated_points[j][1] for j in k_idx])
            
            # Selection Strategy: Epsilon-Greedy
            # Exploration: Maximize min_dist (distance to nearest neighbor)
            # Exploitation: Minimize pred_value (predicted f)
            
            # Normalize scores to combine or select based on epsilon
            # We select based on epsilon:
            # With prob epsilon, pick candidate with max min_dist
            # With prob 1-epsilon, pick candidate with min pred_value
            
            if rng.random() < epsilon:
                # Exploration: Pick the one farthest from existing samples
                # But avoid picking points that are extremely far and likely bad?
                # Actually, for multimodal functions, exploring new areas is key.
                # Let's pick the one with max min_dist.
                best_idx = np.argmax(min_dists)
            else:
                # Exploitation: Pick the one with lowest predicted value
                best_idx = np.argmin(pred_values)
            
            x_new = candidates[best_idx]
            f_new = evaluate(x_new)
            
            # Decay epsilon
            epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Phase 3: Final Polish on global best with Nelder-Mead
        if time.time() < deadline and calls_made < budget and best_x is not None:
            x_nm = best_x.copy()
            f_nm = best_f

            delta = 0.2
            simplex = [x_nm.copy()]
            for i in range(dim):
                xi = x_nm.copy()
                xi[i] += delta
                simplex.append(xi)

            simplex_f = [f_nm]
            for i in range(dim):
                if time.time() > deadline or calls_made >= budget:
                    break
                f_new = evaluate(simplex[i + 1])
                simplex_f.append(f_new)

            if len(simplex_f) < dim + 1:
                return best_x, best_f

            alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
            max_iter = min(40, budget - calls_made)
            no_improve_count = 0

            for _ in range(max_iter):
                if time.time() > deadline or calls_made >= budget:
                    break

                indices = np.argsort(simplex_f)
                simplex = [simplex[i] for i in indices]
                simplex_f = [simplex_f[i] for i in indices]

                max_dist = 0.0
                for i in range(len(simplex)):
                    for j in range(i + 1, len(simplex)):
                        d = np.linalg.norm(simplex[i] - simplex[j])
                        if d > max_dist:
                            max_dist = d
                if max_dist < 1e-6:
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                xr = centroid + alpha * (centroid - simplex[-1])
                fr = evaluate(xr)

                if fr < simplex_f[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = evaluate(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                    no_improve_count = 0
                elif fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                    no_improve_count = 0
                else:
                    if fr < simplex_f[-1]:
                        xc = centroid + rho * (centroid - simplex[-1])
                    else:
                        xc = centroid - rho * (centroid - simplex[-1])
                    fc = evaluate(xc)

                    if fc < min(fr, simplex_f[-1]):
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                        no_improve_count = 0
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            if time.time() > deadline or calls_made >= budget:
                                break
                            simplex_f[i] = evaluate(simplex[i])
                        no_improve_count += 1

                if no_improve_count > 5:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
