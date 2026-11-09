"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a piecewise-linear approximation via ramp (clipped) functions to model
    the elastic region, yield point, and strain hardening, with temperature effects.
    """
    _t0 = time.perf_counter()
    _DEADLINE = 135.0  # safety margin before 150s kill

    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Ensure inputs are arrays
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)

    # Define key strain breakpoints for aluminium stress-strain curve
    # Typical aluminium: yield around 0.002-0.005 strain, hardening up to ~0.5+
    # We use fixed breakpoints to create a piecewise-linear structure
    # The parameters will be the slopes and intercepts at each segment

    # Ramp function: max(0, x - a) creates a kink at x=a
    # This gives us a piecewise-linear, convex function

    # Elastic region: linear up to yield point
    # Yield at approximately strain = 0.003 (typical for aluminium)
    # Hardening region: power-law or linear hardening after yield

    # We'll use several ramp functions at different strain levels
    # to capture the multi-regime behavior

    # Strain breakpoints (fixed, not fitted - to keep only 10 params)
    # These are chosen based on typical aluminium behavior
    a1 = 0.001   # very small, captures initial elastic slope change
    a2 = 0.003   # yield point
    a3 = 0.01    # early plastic
    a4 = 0.05    # mid plastic
    a5 = 0.2     # late plastic
    a6 = 0.5     # near max strain in training

    # Temperature breakpoints
    b1 = 0.5     # mid-range temperature

    # Ramp functions in strain
    r1 = np.maximum(0.0, strain - a1)
    r2 = np.maximum(0.0, strain - a2)
    r3 = np.maximum(0.0, strain - a3)
    r4 = np.maximum(0.0, strain - a4)
    r5 = np.maximum(0.0, strain - a5)
    r6 = np.maximum(0.0, strain - a6)

    # Negative ramps (for concave regions if needed)
    nr1 = np.maximum(0.0, a1 - strain)
    nr2 = np.maximum(0.0, a2 - strain)
    nr3 = np.maximum(0.0, a3 - strain)

    # Temperature ramps
    tr1 = np.maximum(0.0, temp - b1)
    ntr1 = np.maximum(0.0, b1 - temp)

    # Base linear term in strain (elastic modulus)
    # The model: sigma = p0 + p1*strain + p2*r2 + p3*r3 + p4*r4 + p5*r5
    #             + p6*temp + p7*(strain*temp) + p8*r2*temp + p9*r3*temp

    # This creates a piecewise-linear function in strain with slopes that
    # change at the breakpoints, allowing BFGS to fit the elastic modulus,
    # yield stress, and hardening modulus separately.

    prediction = (
        p0 +
        p1 * strain +
        p2 * r2 +
        p3 * r3 +
        p4 * r4 +
        p5 * r5 +
        p6 * temp +
        p7 * (strain * temp) +
        p8 * (r2 * temp) +
        p9 * (r3 * temp)
    )

    # Deadline check
    if time.perf_counter() - _t0 > _DEADLINE:
        pass

    return prediction
# EVOLVE-BLOCK-END
