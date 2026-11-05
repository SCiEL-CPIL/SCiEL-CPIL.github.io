"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    # Six-coefficient growth-index approximation for CPL dark energy.
    # c0: base gamma (matter-only ~0.545)
    # c1: linear correction in q = 1-Om(a)
    # c2: quadratic correction in q
    # c3: correction from u = 1+w0+wa*(1-a)
    # c4: correction from wa*a
    # c5: cross-term u*q
    # Tuned to minimize worst-case error across typical CPL cosmologies.
    return [0.545, 0.28, 0.08, -0.05, 0.03, 0.02]
# EVOLVE-BLOCK-END
