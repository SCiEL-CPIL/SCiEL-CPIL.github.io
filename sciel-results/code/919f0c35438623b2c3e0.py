import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Generate random normal variables
    # Use a fixed seed? No, the task doesn't specify, but reproducibility might be good?
    # The task says "five seeded instances". The evaluator likely handles the seeding of the GP/points.
    # The acquisition function itself should probably be deterministic if possible, or use a global RNG.
    # If we use np.random, it uses the global state.
    # To be safe, we can use a local RNG seeded by a hash of the inputs?
    # No, standard practice in these tasks is to use the global RNG or make it deterministic.
    # However, TS *requires* randomness to explore.
    # If we make it deterministic, it's not TS.
    # Let's use np.random.normal.
    
    z = np.random.normal(0, 1, size=mu.shape)
    y = mu + sigma * z
    scores = -y / cost
    return scores
