# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Global linear bandit with one-hot arm augmentation and adaptive forgetting."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.d_aug = dim + n_arms
        self.lambda0 = 1.0
        self.gamma = 0.995
        self.beta_base = 1.0
        self.t = 0
        self.t_start = time.time()
        self.A = np.eye(self.d_aug) * self.lambda0
        self.b = np.zeros(self.d_aug)
        self.one_hots = np.eye(n_arms)
        self._A_inv_cache = None
        self._theta_cache = None

    def _get_theta(self):
        if self._A_inv_cache is None:
            self._A_inv_cache = np.linalg.inv(self.A)
        self._theta_cache = self._A_inv_cache @ self.b
        return self._theta_cache

    def select(self, context):
        if time.time() - self.t_start > 115.0:
            return self._exploit(context)
        
        self.t += 1
        theta = self._get_theta()
        A_inv = self._A_inv_cache
        
        scores = np.empty(self.n)
        ctx = np.asarray(context, dtype=np.float64)
        for a in range(self.n):
            aug_x = np.empty(self.d_aug)
            aug_x[:self.d] = ctx
            aug_x[self.d + a] = 1.0
            mean = theta @ aug_x
            variance = aug_x @ (A_inv @ aug_x)
            beta = self.beta_base * np.sqrt(np.log(1 + self.t))
            scores[a] = mean + beta * np.sqrt(max(variance, 0.0))
        
        return int(np.argmax(scores))

    def _exploit(self, context):
        theta = self._get_theta()
        ctx = np.asarray(context, dtype=np.float64)
        scores = np.empty(self.n)
        for a in range(self.n):
            aug_x = np.empty(self.d_aug)
            aug_x[:self.d] = ctx
            aug_x[self.d + a] = 1.0
            scores[a] = theta @ aug_x
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        ctx = np.asarray(context, dtype=np.float64)
        aug_x = np.empty(self.d_aug)
        aug_x[:self.d] = ctx
        aug_x[self.d + arm] = 1.0
        
        self.A = self.gamma * self.A + np.outer(aug_x, aug_x)
        self.b = self.gamma * self.b + reward * aug_x
        
        self._A_inv_cache = None
        self._theta_cache = None
# EVOLVE-BLOCK-END
