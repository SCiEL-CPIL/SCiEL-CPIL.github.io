import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: The evaluator kills at 330s. We aim to finish well before.
        # LHS evaluation is extremely fast (just arithmetic + problem call), 
        # so we can use the full budget comfortably.
        deadline = 325.0 
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # We will use the entire budget for LHS sampling.
        # This ensures maximum coverage of the search space.
        n_samples = self.budget
        
        # Generate Latin Hypercube Sample
        # For each dimension, we divide the range into n_samples equal intervals
        # and place exactly one sample in each interval.
        # The position within the interval is randomized.
        
        # To ensure reproducibility and utilize the seed set by the evaluator,
        # we use np.random. Since the evaluator sets np.random.seed(seed),
        # this will be deterministic for each seed.
        
        # Constructing LHS matrix
        # For dimension j:
        #   Generate a permutation of [0, n_samples)
        #   Generate n_samples random numbers in [0, 1)
        #   Combine: (perm + rand) / n_samples
        #   Scale to [lo, hi]
        
        lhs = np.zeros((n_samples, self.dim))
        for j in range(self.dim):
            perm = np.random.permutation(n_samples)
            rand = np.random.random(n_samples)
            # Normalize to [0, 1)
            u = (perm + rand) / n_samples
            # Scale to [lo, hi]
            lhs[:, j] = lo[j] + u * (hi[j] - lo[j])
        
        # Evaluate all samples
        # We must check time periodically, but LHS generation is O(N*D).
        # Evaluation is N calls to problem.
        # Problem calls are likely the bottleneck if they are expensive, 
        # but WFG4 is typically cheap.
        
        # Batch evaluation isn't possible if problem expects a single vector,
        # so we loop.
        
        F = np.zeros((n_samples, self.n_obj))
        
        for i in range(n_samples):
            if time.time() > deadline:
                break
            # problem(x) returns 3 objective values
            # The contract says: "problem(x) clips a finite length-12 vector to the bounds and returns 3 objective values."
            # We assume it returns a numpy array or list of length 3.
            try:
                F[i] = problem(lhs[i])
            except Exception:
                # If an exception occurs, we should stop to avoid invalidating the whole result 
                # if it happens before the first eval, but the contract says "exception later keeps the evaluations made".
                # However, if problem fails, it's better to break.
                break
        
        return F
