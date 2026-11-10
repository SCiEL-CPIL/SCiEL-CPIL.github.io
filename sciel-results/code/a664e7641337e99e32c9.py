import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 300.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        N = 20  # Population size
        
        # Generate reference vectors
        # For 3 objectives, use a grid on the simplex
        # Number of vectors = N
        # We can use a simple method to generate N points on the simplex
        # Or use a fixed set of reference vectors for 3 objectives
        # Let's use a uniform grid.
        # For N=20, we can't have a perfect grid.
        # We can use `scipy.stats.qmc` or just random points on simplex and project?
        # Simpler: Use a set of well-known reference vectors for 3 objectives.
        # Or generate random points on the simplex and use them.
        
        # Let's generate N random points on the simplex
        # u ~ Uniform(0,1)^2, normalize
        # Actually, for 3D simplex, we can use:
        # v1 = np.random.rand(N, 3)
        # v1 /= np.sum(v1, axis=1, keepdims=True)
        
        # But fixed reference vectors are better for consistency.
        # Let's use a grid.
        # For 3 objectives, a grid of size k x k has ~ k^2/2 points.
        # If N=20, k=6 gives ~21 points.
        
        ref_vectors = []
        k = 6
        for i in range(k):
            for j in range(k - i):
                if i + j < k:
                    ref_vectors.append([i / (k - 1), j / (k - 1), (k - 1 - i - j) / (k - 1)])
        # This gives 21 vectors. We need 20.
        # Let's just take the first 20.
        ref_vectors = np.array(ref_vectors[:N])
        
        # Initialization
        X = lo + (hi - lo) * np.random.rand(N, dim)
        F = np.array([problem(x) for x in X])
        used = N
        
        def sbx_crossover(x1, x2, prob=0.9, eta=15.0):
            if np.random.rand() < prob:
                child1 = x1.copy()
                child2 = x2.copy()
                for i in range(dim):
                    if np.random.rand() < 0.5:
                        u = np.random.rand()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1))
                        else:
                            beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                        c1 = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                        c2 = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                        child1[i] = c1
                        child2[i] = c2
            else:
                child1 = x1.copy()
                child2 = x2.copy()
            return np.clip(child1, lo, hi), np.clip(child2, lo, hi)

        def polynomial_mutation(x, prob, eta=20.0):
            child = x.copy()
            for i in range(dim):
                if np.random.rand() < prob:
                    u = np.random.rand()
                    if u < 0.5:
                        delta1 = (2 * u) ** (1.0 / (eta + 1)) - 1
                    else:
                        delta1 = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                    yl = (x[i] - lo[i]) / (hi[i] - lo[i])
                    yr = (hi[i] - x[i]) / (hi[i] - lo[i])
                    deltaq = min(delta1 * yl, -delta1 * yr)
                    child[i] = x[i] + deltaq * (hi[i] - lo[i])
            return np.clip(child, lo, hi)

        # Helper to assign individuals to reference vectors
        def assign_to_refs(F, refs):
            # Normalize F to [0,1] for angle calculation?
            # RVEA typically uses angle between (F - min) and (ref - min)
            # Or simply distance in normalized space.
            # Let's use angle.
            # f_min = np.min(F, axis=0)
            # F_norm = (F - f_min) / (np.max(F, axis=0) - f_min + 1e-10)
            # ref_norm = refs
            
            # Actually, standard RVEA uses the angle between the vector from the ideal point to the individual and the reference vector.
            # Ideal point: min of each objective.
            # Let's use a simpler metric: Euclidean distance in normalized space.
            f_min = np.min(F, axis=0)
            f_max = np.max(F, axis=0)
            range_f = f_max - f_min
            range_f[range_f < 1e-10] = 1.0
            F_norm = (F - f_min) / range_f
            
            # Reference vectors are already in [0,1] simplex
            # We need to map them to the same normalized space?
            # No, reference vectors define the directions.
            # Let's use the angle between F_norm and ref_vectors.
            # But F_norm is in [0,1]^n_obj, not on the simplex.
            # We can project F_norm to the simplex?
            # Or use a different metric.
            
            # Let's use the "niche" definition from RVEA:
            # An individual belongs to the niche of the reference vector with the smallest angle.
            # Angle calculation:
            # cos(theta) = (F_norm . ref) / (|F_norm| |ref|)
            
            assignments = np.zeros(len(F), dtype=int)
            for i in range(len(F)):
                f_vec = F_norm[i]
                # Normalize f_vec?
                # ref vectors are normalized? No, they sum to 1.
                # Let's normalize both.
                f_norm = f_vec / (np.linalg.norm(f_vec) + 1e-10)
                best_dist = np.inf
                best_idx = 0
                for j in range(len(refs)):
                    r_vec = refs[j]
                    r_norm = r_vec / (np.linalg.norm(r_vec) + 1e-10)
                    dot = np.dot(f_norm, r_norm)
                    # Minimize angle -> Maximize dot
                    if dot > best_dist:
                        best_dist = dot
                        best_idx = j
                assignments[i] = best_idx
            return assignments

        # Main Loop
        while used < budget:
            if time.time() > deadline - 5:
                break
            
            # Assign current population to reference vectors
            assignments = assign_to_refs(F, ref_vectors)
            
            # Selection
            # For each reference vector, pick the best individual assigned to it.
            # If a vector has no individual, pick a random one from the population.
            # We need N parents.
            parents = []
            for j in range(N):
                # Find individuals assigned to ref j
                idxs = np.where(assignments == j)[0]
                if len(idxs) > 0:
                    # Pick the one with best objective value?
                    # Or random?
                    # Let's pick the one with smallest sum of objectives (normalized).
                    # Or just random from the niche.
                    # Let's use random for simplicity, or best.
                    # Best: min sum of F_norm
                    best_idx = idxs[np.argmin(np.sum(F_norm[idxs], axis=1))]
                    parents.append(best_idx)
                else:
                    # Pick random
                    parents.append(np.random.randint(0, N))
            
            # Create offspring
            offspring_X = []
            offspring_F = []
            
            n_pairs = N // 2
            for _ in range(n_pairs):
                p1_idx = parents[np.random.randint(0, N)]
                p2_idx = parents[np.random.randint(0, N)]
                
                c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx])
                c1 = polynomial_mutation(c1, 1.0/dim)
                c2 = polynomial_mutation(c2, 1.0/dim)
                
                offspring_X.append(c1)
                offspring_X.append(c2)
            
            # Evaluate offspring
            for x in offspring_X:
                if used >= budget:
                    break
                f = problem(x)
                offspring_F.append(f)
                used += 1
            
            # Merge and Select
            X_all = np.vstack([X, np.array(offspring_X)])
            F_all = np.vstack([F, np.array(offspring_F)])
            
            # Assign all to reference vectors
            assignments_all = assign_to_refs(F_all, ref_vectors)
            
            # Select next generation
            next_X = []
            next_F = []
            # We want one individual per reference vector
            for j in range(N):
                idxs = np.where(assignments_all == j)[0]
                if len(idxs) > 0:
                    # Pick the best one (smallest angle/distance)
                    # We need to re-calculate distances or store them.
                    # For simplicity, pick the one with best hypervolume contribution?
                    # Or just the one with smallest sum of normalized objectives.
                    f_min = np.min(F_all, axis=0)
                    f_max = np.max(F_all, axis=0)
                    range_f = f_max - f_min
                    range_f[range_f < 1e-10] = 1.0
                    F_norm_all = (F_all - f_min) / range_f
                    
                    # Calculate distance to ref vector j
                    r_vec = ref_vectors[j]
                    r_norm = r_vec / (np.linalg.norm(r_vec) + 1e-10)
                    
                    best_dist = np.inf
                    best_idx = -1
                    for i in idxs:
                        f_vec = F_norm_all[i]
                        f_norm = f_vec / (np.linalg.norm(f_vec) + 1e-10)
                        dot = np.dot(f_norm, r_norm)
                        if dot > best_dist:
                            best_dist = dot
                            best_idx = i
                    if best_idx != -1:
                        next_X.append(X_all[best_idx])
                        next_F.append(F_all[best_idx])
                else:
                    # If no individual, keep a random one from previous generation?
                    # Or just skip? We need N individuals.
                    # Let's pick a random individual from X_all
                    rand_idx = np.random.randint(0, len(X_all))
                    next_X.append(X_all[rand_idx])
                    next_F.append(F_all[rand_idx])
            
            X = np.array(next_X)
            F = np.array(next_F)
            
        return X
