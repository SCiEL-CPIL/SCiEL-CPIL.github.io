"""Rewritten program: global linear model with context-augmented features."""
# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """
    Global linear contextual bandit policy using context-augmented features.
    Features = [one_hot(arm), context] for each arm, but we use a shared
    parameter vector with arm-specific bias terms and a shared context weight.
    
    Actually, to follow the directive precisely: use a single global linear model
    where the feature vector for arm a given context x is:
        phi(a, x) = [one_hot_a, x]
    and we maintain a single A matrix and b vector for the full feature space.
    The dimension of phi is n_arms + dim.
    
    We use a LinUCB-style exploration with a forgetting factor to handle
    non-stationarity.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.total_dim = n_arms + dim  # one-hot arm + context
        
        # Initialize the global linear model
        # A is a (total_dim x total_dim) matrix, b is (total_dim,)
        self.A = np.eye(self.total_dim)
        self.b = np.zeros(self.total_dim)
        
        # Forgetting factor for non-stationarity
        self.forgetting = 0.99
        # Exploration parameter
        self.alpha = 1.0

    def _feature(self, arm, context):
        """Construct the augmented feature vector for a given arm and context."""
        feat = np.zeros(self.total_dim)
        feat[arm] = 1.0  # one-hot encoding of arm
        feat[self.n:] = context  # context features
        return feat

    def select(self, context):
        """Select the arm with the highest upper confidence bound."""
        scores = []
        for a in range(self.n):
            phi = self._feature(a, context)
            # Compute theta = A^{-1} b
            # Use solve instead of explicit inverse for numerical stability
            try:
                theta = np.linalg.solve(self.A, self.b)
            except np.linalg.LinAlgError:
                theta = np.zeros(self.total_dim)
            
            mu = theta @ phi
            # Confidence bound
            # We need phi^T A^{-1} phi. Since we have theta = A^{-1}b, 
            # we can compute A^{-1} phi separately or use the fact that
            # phi^T A^{-1} phi can be computed via solve.
            try:
                A_inv_phi = np.linalg.solve(self.A, phi)
            except np.linalg.LinAlgError:
                A_inv_phi = np.zeros(self.total_dim)
            confidence = self.alpha * np.sqrt(phi @ A_inv_phi)
            scores.append(mu + confidence)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update the global linear model with observed reward."""
        phi = self._feature(arm, context)
        
        # Apply forgetting factor
        self.A = self.forgetting * self.A
        self.b = self.forgetting * self.b
        
        # Add new observation
        self.A += np.outer(phi, phi)
        self.b += reward * phi
# EVOLVE-BLOCK-END
