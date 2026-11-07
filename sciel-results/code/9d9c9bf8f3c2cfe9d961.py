"""Piecewise-linear stress-strain model with a smooth hinge for the elastic/plastic transition."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature.

    Piecewise-linear model with a smooth hinge separating the elastic and
    plastic regimes:
      - elastic slope E * strain (linear Hooke's law at low strain)
      - plastic plateau Y * strain * (1 - exp(-k * strain)) which saturates
        at the yield stress Y, with k a tunable yield threshold
      - temperature softening: strength scales by (1 - c * (temp - t0))
      - small post-yield hardening term
    """
    s = np.asarray(strain, dtype=float)
    t = np.asarray(temp, dtype=float)

    # Unpack the 10 fitted constants
    E = params[0]          # elastic modulus (initial linear slope)
    Y = params[1]          # yield stress (plastic plateau level)
    k = params[2]          # hinge sharpness / yield threshold location
    c = params[3]          # temperature softening coefficient
    t0 = params[4]         # reference temperature
    h = params[5]          # post-yield hardening coefficient
    a = params[6]          # elastic-plateau blend weight
    b = params[7]          # elastic-plateau blend weight
    d = params[8]          # quadratic hardening coefficient
    off = params[9]        # offset

    # Smooth hinge: ~ strain for small strain, ~ 1 for large strain.
    # Clip the exponent argument to avoid overflow for large k*strain.
    z = np.clip(k * s, -50.0, 50.0)
    hinge = s * (1.0 - np.exp(-z))          # elastic->plateau shape

    # Temperature softening factor (strength decreases as temp rises).
    soft = 1.0 - c * (t - t0)

    # Elastic part: linear, but blend toward the plateau as strain grows.
    elastic = E * s * (1.0 - a * hinge)
    # Plastic part: saturates at Y, softened by temperature.
    plastic = Y * soft * hinge
    # Post-yield hardening: gentle rise beyond the plateau.
    hardening = h * soft * (hinge ** 2) + d * (s ** 2)

    stress = elastic + plastic + hardening + off
    return stress
# EVOLVE-BLOCK-END
