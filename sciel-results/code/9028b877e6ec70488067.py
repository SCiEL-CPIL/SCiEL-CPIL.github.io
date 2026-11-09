import numpy as np

def acquisition(mu, sigma, f_best):
        kappa = 1.5
        return -mu + kappa * sigma
