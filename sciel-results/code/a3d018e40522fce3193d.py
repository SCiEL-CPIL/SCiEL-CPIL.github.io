# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Policy:
    """LinUCB with per-arm regime-switching (stable/volatile) Kalman states."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.round_count = 0

        # Base LinUCB state (used as the "stable" mode)
        self.A_inv = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Volatile (high-noise) mode: separate A^{-1}, b with a stronger prior
        # (smaller effective lambda) so it adapts faster to recent data.
        lam_vol = 0.1
        self.A_inv_vol = [lam_vol * np.eye(dim) for _ in range(n_arms)]
        self.b_vol = [np.zeros(dim) for _ in range(n_arms)]

        # Regime probabilities: P(stable) per arm; P(volatile) = 1 - p
        self.p_stable = [0.95 for _ in range(n_arms)]

        # Innovation statistics per arm (exponentially weighted)
        self.innov_mean = [0.0 for _ in range(n_arms)]
        self.innov_std = [1.0 for _ in range(n_arms)]
        self.innov_w = 0.9  # EWMA weight for older values

        # Regime-switch parameters
        self.switch_threshold = 2.0   # z-score threshold for switching to volatile
        self.recover_threshold = 0.8  # z-score below which we recover to stable
        self.switch_rate = 0.5        # how fast p_stable moves toward target
        self.recover_rate = 0.3

        # Exploration
        self.C = 1.0

        # Deadline guard
        self._start_time = time.time()
        self._deadline = 120.0

    def _shrink_inverse(self, V, lam):
        """Return (lam*I)^{-1} blended with V via inverse update:
        We maintain A^{-1}. To add a prior lam*I, we effectively set
        A_new = A_old + lam*I, so A_new^{-1} = (A_old^{-1}^{-1} + lam*I)^{-1}.
        We do this via: V_new = V - V*(lam*I)*V / (I + lam*V)  -- not straightforward.
        Simpler: just scale V by a factor < 1 to emulate forgetting (increases effective noise).
        """
        # Forgetting: V_new = V / gamma  (equivalent to A *= gamma)
        return V

    def _update_linucb(self, A_inv, b, context, reward):
        """Standard LinUCB update using Sherman-Morrison."""
        x = context
        b_new = b + reward * x
        V = A_inv
        Vx = V @ x
        denom = 1.0 + x @ Vx
        V_new = V - np.outer(Vx, Vx) / denom
        V_new = 0.5 * (V_new + V_new.T)
        return V_new, b_new

    def _theta_and_uncertainty(self, A_inv, b, context):
        theta = A_inv @ b
        pred = theta @ context
        var = context @ A_inv @ context
        return pred, np.sqrt(max(var, 0.0))

    def select(self, context):
        # Deadline guard
        if time.time() - self._start_time > self._deadline - 5.0:
            scores = []
            for a in range(self.n):
                p_s = self.p_stable[a]
                pred_s, unc_s = self._theta_and_uncertainty(self.A_inv[a], self.b[a], context)
                pred_v, unc_v = self._theta_and_uncertainty(self.A_inv_vol[a], self.b_vol[a], context)
                pred = p_s * pred_s + (1 - p_s) * pred_v
                unc = p_s * unc_s + (1 - p_s) * unc_v
                scores.append(pred + self.C * unc)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            p_s = self.p_stable[a]
            pred_s, unc_s = self._theta_and_uncertainty(self.A_inv[a], self.b[a], context)
            pred_v, unc_v = self._theta_and_uncertainty(self.A_inv_vol[a], self.b_vol[a], context)
            # Blend predictions and uncertainties based on regime probability
            pred = p_s * pred_s + (1 - p_s) * pred_v
            unc = p_s * unc_s + (1 - p_s) * unc_v
            scores.append(pred + self.C * unc)
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.round_count += 1

        # Deadline guard
        if time.time() - self._start_time > self._deadline - 2.0:
            return

        x = context

        # Compute innovations from both modes
        pred_s, unc_s = self._theta_and_uncertainty(self.A_inv[arm], self.b[arm], x)
        pred_v, unc_v = self._theta_and_uncertainty(self.A_inv_vol[arm], self.b_vol[arm], x)

        innov_s = reward - pred_s
        innov_v = reward - pred_v

        # Update innovation statistics (EWMA of squared innovations for stable mode)
        w = self.innov_w
        self.innov_mean[arm] = w * self.innov_mean[arm] + (1 - w) * innov_s
        # Track standard deviation via EWMA of squared deviation
        sq_dev = (innov_s - self.innov_mean[arm]) ** 2
        self.innov_std[arm] = np.sqrt(w * self.innov_std[arm] ** 2 + (1 - w) * sq_dev)

        # Compute z-score of the stable-mode innovation
        std_s = max(self.innov_std[arm], 1e-6)
        z = abs(innov_s) / std_s

        # Regime-switching logic
        p_s = self.p_stable[arm]
        if z > self.switch_threshold:
            # Switch toward volatile mode
            target = 0.2  # mostly volatile
            rate = self.switch_rate
        elif z < self.recover_threshold:
            # Recover toward stable mode
            target = 0.95
            rate = self.recover_rate
        else:
            # In between: keep current
            target = p_s
            rate = 0.0

        if rate > 0:
            p_s = p_s + rate * (target - p_s)
            p_s = np.clip(p_s, 0.05, 0.98)
            self.p_stable[arm] = p_s

        # Update both modes (stable always; volatile always to keep it fresh)
        self.A_inv[arm], self.b[arm] = self._update_linucb(self.A_inv[arm], self.b[arm], x, reward)
        self.A_inv_vol[arm], self.b_vol[arm] = self._update_linucb(self.A_inv_vol[arm], self.b_vol[arm], x, reward)

        # Apply forgetting to the volatile mode to keep it responsive
        # Scale A_inv_vol up (equivalent to shrinking A) to forget older data
        # Use a forgetting factor: A_inv_vol /= gamma_vol where gamma_vol < 1
        # This increases uncertainty, making the volatile mode more adaptive.
        gamma_vol = 0.98
        self.A_inv_vol[arm] = self.A_inv_vol[arm] / gamma_vol
        # Symmetrize
        self.A_inv_vol[arm] = 0.5 * (self.A_inv_vol[arm] + self.A_inv_vol[arm].T)
# EVOLVE-BLOCK-END
