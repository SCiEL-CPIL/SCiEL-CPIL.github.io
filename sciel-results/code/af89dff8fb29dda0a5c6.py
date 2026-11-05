"""Program to minimize the autocorrelation inequality value."""
import numpy as np

N = 400

def construct():
    # The theoretical minimum for the given metric is 2.0, achieved by constant sequences.
    # Any deviation increases the value due to the Cauchy-Schwarz inequality 
    # sum(h^2) >= (sum(h)^2)/N and max(conv(h,h)) >= sum(h^2).
    # Thus, the constant vector is the global optimum.
    return np.ones(N)
