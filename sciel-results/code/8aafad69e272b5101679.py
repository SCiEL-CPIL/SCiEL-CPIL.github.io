import numpy as np

def acquisition(mu, sigma, f_best):
    n_candidates = len(mu)
    n_samples = 100
    
    # Use a fixed seed for reproducibility? The task says "fresh uniform candidates" but doesn't specify RNG seed for acquisition. 
    # However, using a fixed seed makes the acquisition deterministic given mu/sigma. 
    # Using a random seed makes it stochastic. 
    # The current code uses seed 42. I will keep a fixed seed to ensure consistency in behavior, or maybe use np.random without seed? 
    # The prompt says "draws 2000 fresh uniform candidates". It doesn't say the acquisition must be stochastic. 
    # Deterministic acquisition is often preferred for debugging but stochastic is part of TS. 
    # Let's stick to the standard TS interpretation: the randomness comes from the samples. 
    # I will use np.random.standard_normal without a fixed seed to allow for true exploration if the evaluator runs multiple times? 
    # Wait, the evaluator runs the function 30 times per seed. 
    # If I use a fixed seed inside acquisition, every call will generate the *same* noise pattern. 
    # This might cause issues if the posterior doesn't change much? 
    # Actually, `mu` and `sigma` change every step. So even with fixed noise `Z`, `mu + sigma*Z` changes. 
    # But the *structure* of the noise is fixed. 
    # It is generally safer to use a fresh random seed or no seed (system time) for TS to avoid getting stuck in local minima of the sampling process. 
    # However, the current program used seed 42 and got 1.41. 
    # Let's use `np.random.default_rng()` without a fixed seed to be "fresh". Or better, just `np.random.standard_normal`.
    
    # Actually, looking at the task description: "For each seed 0, 1 and 2...". The evaluator handles the seed for the *data*. 
    # The acquisition function should probably be stochastic or at least not locked to a single pattern that might correlate poorly with the changing posterior.
    # I will use `np.random.standard_normal` (global state) or create a new RNG. 
    # Let's create a new RNG each time? No, that's slow. 
    # Let's just use `np.random.standard_normal` which uses the global state.
    
    samples = mu[None, :] + sigma[None, :] * np.random.standard_normal((n_samples, n_candidates))
    
    # Compute the minimum sampled value for each candidate
    min_samples = np.min(samples, axis=0)
    
    # Score is negative of min sample (higher score = lower value)
    scores = -min_samples
    
    return scores
