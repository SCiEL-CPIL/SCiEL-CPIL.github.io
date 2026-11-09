<<<<<<< SEARCH
            # Evaluate
            xs = (x + sigma * ys).T
            fxs = np.array([func(np.clip(xi, -5, 5)) for xi in xs])
=======
            # Evaluate
            xs = x + sigma * ys
            # Ensure finite values and correct shape
            valid_mask = np.all(np.isfinite(xs), axis=1)
            if not np.any(valid_mask):
                # Fallback if all invalid
                xs = np.array([x + sigma * rng.normal(0, 1, dim) for _ in range(lambda_)])
                valid_mask = np.all(np.isfinite(xs), axis=1)
            
            # Only evaluate valid points, fill others with inf
            fxs = np.full(lambda_, np.inf)
            for i in range(lambda_):
                if valid_mask[i]:
                    fxs[i] = func(np.clip(xs[i], -5, 5))
>>>>>>> REPLACE
