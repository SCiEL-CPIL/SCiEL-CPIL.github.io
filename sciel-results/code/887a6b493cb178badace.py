<<<<<<< SEARCH
        # Initial population: structured + random
        pop_size = 50
        init_points = []

        # Corners
        for i in range(2 ** dim):
            point = np.array([hi[j] if (i >> j) & 1 else lo[j] for j in range(dim)])
            init_points.append(point)

        # Midpoint
        init_points.append((lo + hi) / 2)

        # Face centers
        for j in range(dim):
            for k in range(j + 1, dim):
                for vj in [lo[j], hi[j]]:
                    for vk in [lo[k], hi[k]]:
                        point = np.array([(lo[t] + hi[t]) / 2 if t not in (j, k)
                                          else vj if t == j else vk for t in range(dim)])
                        init_points.append(point)

        # Fill with random
        while len(init_points) < pop_size:
            init_points.append(lo + (hi - lo) * rng.random(dim))

        X = np.array(init_points[:pop_size])
        F = np.array([problem(x) for x in X])
        used = pop_size
=======
        # Initial population: Latin Hypercube Sampling (LHS) for better spread
        pop_size = 50
        init_points = []

        # LHS
        for i in range(pop_size):
            point = np.zeros(dim)
            for j in range(dim):
                # Sample from uniform distribution in [0, 1] and map to [lo, hi]
                # Use stratified sampling: divide [0,1] into pop_size bins, pick one random point in each bin
                # But here we are generating one point at a time, so we need to ensure stratification across the population.
                # Simpler: Generate a permutation of indices 0..pop_size-1 for each dimension, then map.
                pass
        
        # Proper LHS generation
        lhs_indices = np.array([np.random.permutation(pop_size) for _ in range(dim)]).T
        for i in range(pop_size):
            point = np.zeros(dim)
            for j in range(dim):
                # Map index to [0, 1)
                u = (lhs_indices[i, j] + rng.random()) / pop_size
                point[j] = lo[j] + u * (hi[j] - lo[j])
            init_points.append(point)

        X = np.array(init_points)
        F = np.array([problem(x) for x in X])
        used = pop_size
>>>>>>> REPLACE
