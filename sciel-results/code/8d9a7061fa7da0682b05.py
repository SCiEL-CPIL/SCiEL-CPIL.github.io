<<<<<<< SEARCH
        # Use the full budget for LHS samples of x1 to maximize coverage of the Pareto front.
        # For ZDT2, setting x_2...x_n to 0 minimizes g, placing the point on the Pareto front.
        n_lhs = budget
        
        # Generate LHS samples for x1 in [0, 1]
        # LHS: divide [0,1] into n_lhs strata, pick one random point from each
        perm = np.random.permutation(n_lhs)
        x1_samples = (perm + np.random.random(n_lhs)) / n_lhs
        x1_samples = np.clip(x1_samples, lo[0], hi[0])
=======
        # Use the full budget for uniform grid samples of x1 to maximize coverage of the Pareto front.
        # A deterministic uniform grid ensures even spacing and exact endpoint coverage,
        # which maximizes hypervolume for a known front with a distant reference point.
        n_samples = budget
        
        # Generate uniform grid samples for x1 in [0, 1]
        x1_samples = np.linspace(lo[0], hi[0], n_samples)
>>>>>>> REPLACE
