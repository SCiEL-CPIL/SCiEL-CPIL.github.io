"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """MOEA/D style multi-objective optimizer for DTLZ2."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        N = 100
        if budget < N:
            N = max(2, budget)

        # Generate weight vectors for MOEA/D
        # For 3 objectives, we can use a simplex lattice or a grid.
        # Let's use a systematic generation of weight vectors.
        # Number of weight vectors = N
        # We want them to cover the simplex.
        # For M=3, we can iterate over i, j such that i+j <= H, normalize.
        # To get exactly N vectors, we can pick a grid size H such that (H+1)(H+2)/2 is close to N.
        # Or just generate N random points on the simplex? No, systematic is better.
        # Let's find H such that num_vectors is close to N.
        H = int((np.sqrt(8 * N - 7) - 1) / 2)
        while (H + 1) * (H + 2) // 2 < N:
            H += 1
        while (H + 1) * (H + 2) // 2 > N and H > 0:
            H -= 1
        
        weights = []
        for i in range(H + 1):
            for j in range(H - i + 1):
                k = H - i - j
                w = np.array([i, j, k], dtype=float) / H
                weights.append(w)
        
        # If we have more weights than N, we can subsample or just use the first N.
        # If we have fewer, we can duplicate or adjust.
        # For simplicity, let's just use the first N weights if len(weights) >= N, 
        # or if len(weights) < N, we might need to adjust N or weights.
        # Actually, it's safer to set N = len(weights) if we want exact mapping.
        # But budget is fixed. Let's stick to N individuals.
        # If len(weights) != N, we can just take N weights.
        # To ensure good coverage, let's regenerate if needed or just accept.
        # Let's force N to be the number of weights generated to keep 1-to-1 mapping.
        # But we need to respect budget. 
        # Let's just use the generated weights and set N = len(weights).
        # If N is too large for budget, we reduce H.
        # Let's re-calculate H based on budget constraints more carefully.
        # Max evals is 200. 
        # If N=50, we can do ~4 generations.
        # If N=100, we can do ~2 generations.
        # Let's target N around 50-80.
        
        # Let's just generate N weight vectors by sampling the simplex uniformly?
        # No, systematic is better.
        # Let's just use a fixed set of weights if N is not a triangular number.
        # Simpler: Use N weight vectors. 
        # We can generate a grid of size H and if count != N, we just pick N from them randomly? 
        # Or better, adjust H to get closest to N.
        
        # Let's refine H selection to get exactly N or close.
        # Actually, for MOEA/D, it's fine to have N individuals and N weight vectors.
        # Let's just create N weight vectors.
        # A simple way: Create a grid of H=20 (231 points). If N=100, pick 100 randomly? 
        # No, that loses structure.
        # Let's just use the first N points from the grid.
        
        # Recalculate H to get at least N points, then take first N.
        H = 1
        while (H + 1) * (H + 2) // 2 < N:
            H += 1
        
        weights = []
        for i in range(H + 1):
            for j in range(H - i + 1):
                k = H - i - j
                w = np.array([i, j, k], dtype=float) / H
                weights.append(w)
        
        # Take first N weights
        weights = np.array(weights[:N])
        
        # Initial population
        X = lo + span * rng.random((N, dim))
        F = np.array([problem(x) for x in X])
        used = N

        # Compute ideal point (min values) for Tchebycheff normalization
        # We don't know the ideal point a priori, so we use the current min.
        # This is standard for MOEA/D.
        
        # Define neighborhood
        # For each weight vector, find the T closest neighbors.
        T = 20 # Neighborhood size
        if N <= T:
            T = N
        neighbors = []
        for i in range(N):
            dists = np.sum((weights - weights[i])**2, axis=1)
            idx = np.argsort(dists)
            neighbors.append(idx[:T].tolist())

        # Tchebycheff aggregation function
        def tchebycheff(f, w, z):
            # z is the ideal point (min of each objective)
            # We need to normalize f. 
            # Standard: (f_i - z_i) / range_i
            # We don't have range. We can use 1.0 or max-min.
            # Using 1.0 is common if scales are similar. DTLZ2 objectives are in [0, 2].
            # Let's use a dynamic range or just 1.0.
            # To be safe, let's use 1.0 as the range factor, or better, 
            # we can update z and range periodically.
            # For simplicity and speed, let's use z = min(F) and range = 1.0.
            # Actually, DTLZ2 objectives are bounded by 2. So range 2 is safe.
            # Let's use range = 2.0.
            return np.max(w * (f - z) / 2.0)

        # SBX crossover
        def sbx_crossover(x1, x2, eta=20.0):
            if rng.random() < 0.5:
                return x1.copy(), x2.copy()
            child1 = x1.copy()
            child2 = x2.copy()
            for d in range(dim):
                if rng.random() < 0.5:
                    if abs(x1[d] - x2[d]) < 1e-14:
                        continue
                    if rng.random() < 0.5:
                        beta = 1.0 / (2.0 * rng.random() + 1e-12)
                    else:
                        beta = 1.0 / (2.0 * (1.0 - rng.random()) + 1e-12)
                    beta += 1.0
                    if beta > 0:
                        alpha = 2.0 - beta ** (-eta)
                        if rng.random() < 0.5:
                            child1[d] = 0.5 * ((1 + alpha) * x1[d] + (1 - alpha) * x2[d])
                        else:
                            child1[d] = 0.5 * ((1 - alpha) * x1[d] + (1 + alpha) * x2[d])
                    if beta > 0:
                        alpha = 2.0 - beta ** (-eta)
                        if rng.random() < 0.5:
                            child2[d] = 0.5 * ((1 + alpha) * x2[d] + (1 - alpha) * x1[d])
                        else:
                            child2[d] = 0.5 * ((1 - alpha) * x2[d] + (1 + alpha) * x1[d])
            return np.clip(child1, lo, hi), np.clip(child2, lo, hi)

        # Polynomial mutation
        def polynomial_mutate(x, eta=20.0, prob=1.0 / dim):
            child = x.copy()
            for d in range(dim):
                if rng.random() < prob:
                    delta1 = (child[d] - lo[d]) / span[d]
                    delta2 = (hi[d] - child[d]) / span[d]
                    rand = rng.random()
                    if rand < 0.5:
                        xy = 1.0 + (1.0 + 2.0 * delta1) * rand
                        deltaq = (xy ** (1.0 / (eta + 1.0)) - 1.0)
                    else:
                        xy = 1.0 + (1.0 + 2.0 * delta2) * (1.0 - rand)
                        deltaq = 1.0 - (xy ** (1.0 / (eta + 1.0)))
                    child[d] += deltaq * span[d]
                    child[d] = np.clip(child[d], lo[d], hi[d])
            return child

        # Main loop
        # We have budget - N evaluations left.
        # Each generation produces N offspring (2 per parent pair, but we do 1 offspring per individual to save budget? 
        # Standard MOEA/D: For each individual i, produce 1 offspring. 
        # This uses N evaluations per generation.
        
        z = np.min(F, axis=0) # Ideal point estimate
        
        while used < budget:
            # Determine how many offspring we can generate in this generation
            # We can generate up to N offspring.
            # If budget - used < N, we generate only that many.
            n_offspring = min(N, budget - used)
            
            # Shuffle indices to process
            indices = list(range(N))
            rng.shuffle(indices)
            
            # We will update the population in-place
            # To avoid using updated parents for later offspring in the same generation,
            # we can either store new values and apply at end, or just accept the asynchronous update.
            # Asynchronous is common and faster.
            
            for i in indices[:n_offspring]:
                # Select 2 parents from neighborhood of i
                neigh = neighbors[i]
                p1_idx = neigh[rng.integers(len(neigh))]
                p2_idx = neigh[rng.integers(len(neigh))]
                
                # Crossover and Mutation
                c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx])
                # We only need 1 offspring to save budget? 
                # If we use both, we use 2 evals. 
                # If we use 1, we use 1 eval.
                # Let's use 1 offspring per iteration to maximize generations.
                offspring = c1
                offspring = polynomial_mutate(offspring)
                
                f_offspring = problem(offspring)
                used += 1
                
                # Update ideal point
                z = np.minimum(z, f_offspring)
                
                # Check if offspring is better than parent i in Tchebycheff sense
                w_i = weights[i]
                val_parent = tchebycheff(F[i], w_i, z)
                val_offspring = tchebycheff(f_offspring, w_i, z)
                
                if val_offspring < val_parent:
                    X[i] = offspring
                    F[i] = f_offspring
                    
                    # Also check neighbors? 
                    # Optional: Update neighbors if better.
                    # This helps convergence but costs computation.
                    # Given budget is small, let's stick to just updating self to keep it simple and fast.
                    # Or update a few neighbors.
                    # Let's update neighbors if they are worse.
                    for j in neigh:
                        if j == i:
                            continue
                        w_j = weights[j]
                        val_j = tchebycheff(F[j], w_j, z)
                        val_off_j = tchebycheff(f_offspring, w_j, z)
                        if val_off_j < val_j:
                            X[j] = offspring
                            F[j] = f_offspring
                            # Note: This might cause multiple individuals to have the same solution.
                            # This is acceptable in MOEA/D but can reduce diversity.
                            # To mitigate, we might limit neighbor updates.
                            # For now, let's just update self to be safe and fast.
                            # Reverting neighbor update for simplicity and diversity preservation.
                            # The self-update is the core mechanism.
                            pass # Do not update neighbors to preserve diversity and speed

        return X
# EVOLVE-BLOCK-END
