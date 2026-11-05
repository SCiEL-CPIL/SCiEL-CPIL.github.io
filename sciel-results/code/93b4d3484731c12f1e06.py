# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary linear contextual bandit using Adaptive Discounted Least Squares.
    
    Improvements:
    1. Cached Theta and Cholesky factors to speed up select().
    2. More aggressive adaptive beta based on residual CUSUM/EMA.
    3. Lower beta_min to allow faster forgetting during detected non-stationarity.
    4. Tuned exploration parameters.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.lam = 1.0  # Ridge regularization

        # Per-arm model parameters
        self.A = [np.eye(dim) * self.lam for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Cached solutions
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.chol = [None for _ in range(n_arms)]
        
        # Initialize theta
        for a in range(n_arms):
            self._update_arm_solution(a)

        # Statistics for adaptive beta
        self.residual_ema = 0.0
        self.residual_alpha = 0.1
        
        # Beta bounds - Lower min for faster adaptation
        self.beta_min = 0.70
        self.beta_max = 0.995
        
        # Exploration parameters
        self.alpha_base = 1.5
        self.alpha_nonstat_weight = 1.5
        self.alpha_early_boost = 1.5
        self.early_phase_steps = 100
        self.step = 0

        # Time management
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0

    def _safe(self):
        return time.time() >= self.deadline

    def _increment_step(self):
        self.step += 1

    def _update_arm_solution(self, a):
        """Update theta and Cholesky factor for arm a."""
        try:
            L = np.linalg.cholesky(self.A[a])
            y = np.linalg.solve(L, self.b[a])
            theta = np.linalg.solve(L.T, y)
            self.theta[a] = theta
            self.chol[a] = L
        except np.linalg.LinAlgError:
            # Fallback
            try:
                self.theta[a] = np.linalg.lstsq(self.A[a], self.b[a], rcond=None)[0]
                # If Cholesky fails, we can't easily compute variance without inverse.
                # For safety, set chol to None and handle in select.
                self.chol[a] = None
            except Exception:
                self.theta[a] = np.zeros(self.d)
                self.chol[a] = None

    def _get_adaptive_beta(self):
        """
        Determine the discount factor beta based on recent prediction errors.
        """
        r = self.residual_ema
        # More aggressive mapping
        # If r > 1.0, we are likely in a non-stationary phase.
        # Map r to [0, 1] with a steeper curve.
        low_r = 0.2
        high_r = 1.5
        
        if r <= low_r:
            nonstat_score = 0.0
        elif r >= high_r:
            nonstat_score = 1.0
        else:
            # Quadratic interpolation for steeper response
            x = (r - low_r) / (high_r - low_r)
            nonstat_score = x * x

        beta = self.beta_max - nonstat_score * (self.beta_max - self.beta_min)
        return beta, nonstat_score

    def select(self, context):
        if self._safe():
            # Emergency: pick best mean
            scores = []
            for a in range(self.n):
                try:
                    scores.append(float(self.theta[a] @ context))
                except Exception:
                    scores.append(0.0)
            return int(np.argmax(scores))

        beta, nonstat_score = self._get_adaptive_beta()
        
        # Adaptive alpha
        early_boost = self.alpha_early_boost * max(0, 1 - self.step / self.early_phase_steps)
        alpha = self.alpha_base + self.alpha_nonstat_weight * nonstat_score + early_boost
        
        scores = []
        for a in range(self.n):
            try:
                pred = float(self.theta[a] @ context)
                
                if self.chol[a] is not None:
                    L = self.chol[a]
                    v = np.linalg.solve(L, context)
                    var_term = float(v @ v)
                else:
                    # Fallback: approximate variance if Cholesky failed
                    # This is rare, but handle it
                    try:
                        cov = np.linalg.inv(self.A[a])
                        var_term = float(context @ cov @ context)
                    except Exception:
                        var_term = 0.0
                
                ucb_score = pred + alpha * np.sqrt(max(var_term, 0.0))
                scores.append(ucb_score)
            except Exception:
                scores.append(0.0)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self._increment_step()
        
        # Compute prediction error
        try:
            pred = float(self.theta[arm] @ context)
            residual = reward - pred
            residual_sq = residual * residual
            
            # Update EMA of squared residuals
            self.residual_ema = (1 - self.residual_alpha) * self.residual_ema + self.residual_alpha * residual_sq
        except Exception:
            pass

        beta, _ = self._get_adaptive_beta()

        # Update the model for the chosen arm
        self.A[arm] = beta * self.A[arm] + np.outer(context, context)
        self.b[arm] = beta * self.b[arm] + reward * context
        
        # Update cached solution for this arm
        self._update_arm_solution(arm)

# EVOLVE-BLOCK-END
