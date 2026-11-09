import numpy as np

def acquisition(mu, sigma, f_best):
    # Thompson Sampling: sample a single realization from the Gaussian posterior
    # For minimization, we want to minimize mu - sigma * z where z ~ N(0,1)
    # Since the evaluator picks the highest score, we return the negative of the sampled value
    # sampled_value = mu - sigma * z
    # score = -sampled_value = -mu + sigma * z
    z = np.random.standard_normal(len(mu))
    return -mu + sigma * z
