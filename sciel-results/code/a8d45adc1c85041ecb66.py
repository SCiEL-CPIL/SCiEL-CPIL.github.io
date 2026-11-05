# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Sliding-window linear contextual bandit for non-stationary settings."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.window = 50
        self.ridge = 1e-2
        self.beta = 1.0
        self.deadline = time.time() + 115.0  # safety margin before 120s kill

        # Ring buffers per arm
        self.ctx_buf = [np.zeros((self.window, self.d)) for _ in range(n_arms)]
        self.rw_buf = [np.zeros(self.window) for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]
        self.ptr = [0 for _ in range(n_arms)]

    def _solve_arm(self, arm):
        """Solve regularized least squares for a given arm using its window."""
        c = self.count[arm]
        if c == 0:
            return np.zeros(self.d), np.eye(self.d)

        # Gather valid data from ring buffer
        start = (self.ptr[arm] - c) % self.window
        if start + c <= self.window:
            X = self.ctx_buf[arm][start:start + c]
            y = self.rw_buf[arm][start:start + c]
        else:
            X = np.vstack([
                self.ctx_buf[arm][start:],
                self.ctx_buf[arm][:start + c - self.window]
            ])
            y = np.concatenate([
                self.rw_buf[arm][start:],
                self.rw_buf[arm][:start + c - self.window]
            ])

        # Regularized least squares: theta = (X^T X + ridge*I)^{-1} X^T y
        XtX = X.T @ X + self.ridge * np.eye(self.d)
        Xty = X.T @ y
        try:
            theta = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            theta = np.linalg.lstsq(XtX, Xty, rcond=None)[0]

        # Inverse for UCB bonus
        try:
            inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(XtX)

        return theta, inv

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            return int(np.random.randint(self.n))

        context = np.asarray(context, dtype=np.float64)
        scores = []
        for a in range(self.n):
            theta, inv = self._solve_arm(a)
            mu = theta @ context
            var = context @ inv @ context
            ucb = mu + self.beta * np.sqrt(max(var, 0.0))
            scores.append(ucb)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        # Store in ring buffer
        pos = self.ptr[arm]
        self.ctx_buf[arm][pos] = context
        self.rw_buf[arm][pos] = reward
        self.ptr[arm] = (pos + 1) % self.window
        if self.count[arm] < self.window:
            self.count[arm] += 1
# EVOLVE-BLOCK-END
