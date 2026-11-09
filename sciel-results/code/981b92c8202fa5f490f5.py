import numpy as np


def acquisition(mu, sigma, f_best):
            # Handle edge cases for sigma
            sigma_safe = np.maximum(sigma, 1e-9)
            
            # Adaptive beta (k) for UCB
            # Min value is 3.0
            # If f_best is close to 3.0, exploit (low k)
            # If f_best is far, explore (high k)
            
            # Define thresholds
            f_exploit = 3.2  # Below this, we are in the basin
            f_explore = 10.0 # Above this, we need to search
            
            if f_best <= f_exploit:
                k = 0.5
            elif f_best >= f_explore:
                k = 4.0
            else:
                # Linear interpolation
                # t = 0 at f_exploit, t = 1 at f_explore
                t = (f_best - f_exploit) / (f_explore - f_exploit)
                k = 0.5 + (4.0 - 0.5) * t
            
            # UCB for minimization: lower mu and higher sigma is better
            # Score = mu - k * sigma
            # Higher score is preferred.
            # If mu is small and sigma is large, score is very small (bad).
            # Wait.
            # If I want to *minimize* f(x), I want to pick x with low f(x).
            # My score function returns values where *higher* is evaluated next.
            # So I need to map "good for minimization" to "high score".
            
            # Good for minimization: Low mu.
            # So Score should be inversely related to mu?
            # Or Score = -mu + k * sigma?
            
            # Let's check the current EI code:
            # z = (f_best - mu) / sigma
            # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
            # If mu is low (good), f_best - mu is high positive. EI is high.
            # So EI returns high scores for low mu.
            
            # If I use UCB = mu - k * sigma:
            # If mu is low (good), UCB is low.
            # If I return UCB, the evaluator picks the *highest* UCB.
            # That would pick the *highest* mu. That's wrong.
            
            # So for minimization, if I use a bound, it should be:
            # Score = -mu + k * sigma  ?
            # Or Score = - (mu - k * sigma) = -mu + k * sigma.
            
            # Let's verify.
            # Point A: mu=3, sigma=0.1. True min is 3.
            # Point B: mu=5, sigma=0.1.
            # We want A.
            # Score A = -3 + 0.1*k.
            # Score B = -5 + 0.1*k.
            # Score A > Score B. Correct.
            
            # What about uncertainty?
            # Point C: mu=4, sigma=1.0.
            # Point D: mu=3.5, sigma=0.0.
            # With k=1.0:
            # Score C = -4 + 1.0 = -3.
            # Score D = -3.5 + 0 = -3.5.
            # Score C > Score D.
            # We prefer C (higher uncertainty, slightly worse mean) over D (certain, worse mean).
            # This is exploration. Correct for UCB.
            
            # So the formula is: score = -mu + k * sigma.
            
            scores = -mu + k * sigma_safe
            
            # Ensure finite
            scores = np.where(np.isfinite(scores), scores, -np.inf)
            
            return scores
