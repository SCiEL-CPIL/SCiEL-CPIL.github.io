            result = differential_evolution(
                wrapper,
                bounds=[(-5, 5)] * dim,
                seed=42,
                maxiter=1000,
                maxfev=budget,
                tol=1e-8,
                init='sobol',
                polish=False,
                mutation=(0.5, 1.5),
                recombination=0.5,
                popsize=5
            )
