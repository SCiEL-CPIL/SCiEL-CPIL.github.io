        import numpy as np
        from scipy.stats import norm

        def acquisition(mu, sigma, f_best, cost):
            sigma_safe = np.maximum(sigma, 1e-9)
            z = (f_best - mu) / sigma_safe
            pi = norm.cdf(z)
            score = pi / cost
            return score
        