<<<<<<< SEARCH
            # Update sigma
            sigma *= np.exp(cs / chiN * (norm_ps / chiN - 1.0))
            if not np.isfinite(sigma) or sigma <= 0:
                sigma = 1.0
            if sigma < 1e-12:
                break
=======
            # Update sigma
            sigma *= np.exp(cs / chiN * (norm_ps / chiN - 1.0))
            if not np.isfinite(sigma) or sigma <= 0:
                sigma = 1.0
            if sigma > 100.0:
                sigma = 100.0
            if sigma < 1e-12:
                break
>>>>>>> REPLACE
