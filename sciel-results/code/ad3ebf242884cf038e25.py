"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Multi-objective optimizer: Sobol sweep + per-individual adaptive Gaussian refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _sobol_sequence(self, n_points, dim):
        """Generate a Sobol quasi-random sequence in [0, 1]^dim."""
        sampler = qmc.Sobol(d=dim, scramble=True, seed=0)
        samples = sampler.random(n=n_points)
        return samples

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sorting. Returns list of fronts (lists of indices)."""
        n = len(F)
        if n == 0:
            return []
        
        # domination[i] = list of indices dominated by i
        domination = [[] for _ in range(n)]
        # dominance_count[i] = number of points that dominate i
        dominance_count = [0] * n
        fronts = [[]]
        
        for i in range(n):
            for j in range(i + 1, n):
                # Check if i dominates j
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    domination[i].append(j)
                    dominance_count[j] += 1
                # Check if j dominates i
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    domination[j].append(i)
                    dominance_count[i] += 1
        
        # First front: points with no dominators
        for i in range(n):
            if dominance_count[i] == 0:
                fronts[0].append(i)
        
        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in domination[i]:
                    dominance_count[j] -= 1
                    if dominance_count[j] == 0:
                        next_front.append(j)
            k += 1
            if next_front:
                fronts.append(next_front)
            else:
                break
        
        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for a set of indices in F."""
        n = len(indices)
        if n <= 2:
            return [float('inf')] * n
        
        distances = [0.0] * n
        m = self.n_obj
        
        for k in range(m):
            # Sort indices by objective k
            sorted_indices = sorted(range(n), key=lambda i: F[indices[i]][k])
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # Range of objective k
            obj_range = F[indices[sorted_indices[-1]]][k] - F[indices[sorted_indices[0]]][k]
            if obj_range == 0:
                continue
            
            for i in range(1, n - 1):
                distances[sorted_indices[i]] += (
                    F[indices[sorted_indices[i + 1]]][k] - F[indices[sorted_indices[i - 1]]][k]
                ) / obj_range
        
        return distances

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Phase 1: Sobol quasi-random sweep for uniform coverage
        sweep_size = int(self.budget * 0.6)
        sweep_size = min(sweep_size, self.budget)

        if sweep_size >= 2:
            sobol_samples = self._sobol_sequence(sweep_size, self.dim)
        else:
            sobol_samples = np.random.random((sweep_size, self.dim))

        all_X = sobol_samples.copy()
        all_F = np.array([problem(x) for x in all_X])
        used = sweep_size

        # Phase 2: Per-individual adaptive step-size local refinement
        remaining = self.budget - used

        if remaining > 0:
            fronts = self._fast_non_dominated_sort(all_F)

            if fronts and fronts[0]:
                non_dom_indices = list(fronts[0])

                # Per-individual adaptive sigma (index-aligned with non_dom_indices)
                init_sigma = 0.10
                sigmas = [init_sigma] * len(non_dom_indices)
                min_sigma = 0.008
                max_sigma = 0.25
                inc = 1.35
                dec = 0.55

                # Track which original index each current non-dom slot maps to,
                # so we can carry sigma across re-sorts. We instead keep a parallel
                # list of (sigma) tied to a stable "agent id" per non-dom point.
                # Simpler: keep a dict agent_id -> sigma; assign ids to current non-dom set.
                agent_sigma = {}
                next_id = 0
                for idx in non_dom_indices:
                    agent_sigma[idx] = init_sigma
                    next_id += 1

                # Budget per round: we evaluate one child per selected parent per round.
                # Keep rounds small to allow many re-sorts; cap by time.
                while used < self.budget and time.time() < deadline:
                    if not non_dom_indices:
                        break

                    # Choose how many to perturb this round: use up to remaining,
                    # but at least 1, and not more than len(non_dom_indices).
                    batch = min(len(non_dom_indices), max(1, self.budget - used))
                    # Limit batch for speed of re-sort; also respect time.
                    batch = min(batch, max(1, len(non_dom_indices)))

                    # Prioritize by crowding distance for diversity.
                    if len(non_dom_indices) > batch:
                        cd = self._crowding_distance(all_F, non_dom_indices)
                        order = sorted(range(len(non_dom_indices)), key=lambda i: cd[i], reverse=True)
                        selected = [non_dom_indices[i] for i in order[:batch]]
                    else:
                        selected = list(non_dom_indices)

                    new_points = []
                    new_parents = []  # parent index in all_X
                    new_sigmas = []
                    for idx in selected:
                        base = all_X[idx]
                        s = agent_sigma.get(idx, init_sigma)
                        perturbed = base + s * np.random.standard_normal(self.dim)
                        perturbed = np.clip(perturbed, lo, hi)
                        new_points.append(perturbed)
                        new_parents.append(idx)
                        new_sigmas.append(s)

                    if not new_points:
                        break

                    new_points = np.array(new_points)
                    new_F = np.array([problem(x) for x in new_points])
                    all_X = np.vstack([all_X, new_points])
                    all_F = np.vstack([all_F, new_F])
                    used += len(new_points)

                    # Re-sort to get updated non-dominated front and update agent sigmas.
                    fronts = self._fast_non_dominated_sort(all_F)
                    if fronts and fronts[0]:
                        new_non_dom = list(fronts[0])
                    else:
                        new_non_dom = []

                    # Update per-agent sigma: for each parent in this batch,
                    # if its child is non-dominated (or improves), increase; else decrease.
                    # We identify the child indices: they are the last len(new_points) rows.
                    base_row = len(all_F) - len(new_points)
                    child_non_dom_set = set(new_non_dom)

                    for k in range(len(new_points)):
                        parent_idx = new_parents[k]
                        child_idx = base_row + k
                        improved = (child_idx in child_non_dom_set) or (
                            not self._is_dominated(all_F, child_idx, exclude=None)
                        )
                        # Also treat "child not dominated" as success.
                        if improved:
                            new_s = min(max_sigma, agent_sigma.get(parent_idx, init_sigma) * inc)
                        else:
                            new_s = max(min_sigma, agent_sigma.get(parent_idx, init_sigma) * dec)
                        # Keep the parent's agent sigma updated (parent may still be in front).
                        if parent_idx in agent_sigma:
                            agent_sigma[parent_idx] = new_s
                        # If the child is on the front, register it with the improved sigma.
                        if child_idx in child_non_dom_set:
                            agent_sigma[child_idx] = new_s

                    # Prune agent_sigma to only current non-dom set to bound memory.
                    current_set = set(new_non_dom)
                    agent_sigma = {k: v for k, v in agent_sigma.items() if k in current_set}
                    # Ensure every current non-dom has an entry.
                    for idx in new_non_dom:
                        if idx not in agent_sigma:
                            agent_sigma[idx] = init_sigma

                    non_dom_indices = new_non_dom

        return all_X
# EVOLVE-BLOCK-END
