            res_de = scipy.optimize.differential_evolution(
                wrapped_func,
                bounds=[(-5, 5)] * self.dim,
                maxiter=100,
                popsize=5, # 25 candidates per gen
                tol=1e-8,
                atol=1e-8,
                init='sobol',
                seed=42,
                callback=de_callback,
                polish=False,
                update_interval=0
            )
