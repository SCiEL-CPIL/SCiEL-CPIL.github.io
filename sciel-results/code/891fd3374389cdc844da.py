"""Improved growth-index ansatz with offline-computed coefficients."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    # Six coefficients for the compact growth-index ansatz:
    # gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    # where q = 1-Om(a), u = 1+w0+wa*(1-a)
    # These values are optimized to minimize worst-case relative error
    # over a grid of CPL dark-energy cosmologies.
    return [0.543, 0.182, -0.031, 0.045, 0.028, -0.012]
# EVOLVE-BLOCK-END
