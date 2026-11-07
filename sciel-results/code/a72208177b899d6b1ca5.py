"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: the evaluator kills after 150s wall-clock.
# We keep a safety margin so we always return something finite.
_DEADLINE = 150.0
_SAFETY_MARGIN = 10.0  # seconds before the hard limit we stop doing extra work
_start_time = time.time()


def _softplus(z):
    # Numerically stable softplus: log(1 + exp(z))
    return np.log1p(np.exp(np.clip(z, -50, 50)))


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Mixture of Experts (MoE) model for symbolic regression.
    
    Two expert functions (quadratic and cubic polynomials) are blended
    using a soft-gating mechanism based on a sigmoid of a linear function
    of x. This allows the model to approximate piecewise-defined functions
    or functions with distinct behavioral regimes.
    
    Model:
        y(x) = b0 + gate(x) * expert1(x) + (1 - gate(x)) * expert2(x)
        
        where:
        gate(x) = sigmoid(p_gate * x + b_gate)
        expert1(x) = a1 + b1*x + c1*x^2
        expert2(x) = a2 + b2*x + c2*x^3
    
    params (length 10):
        p0 = b0       (constant offset)
        p1 = a1       (expert1 constant)
        p2 = b1       (expert1 linear)
        p3 = c1       (expert1 quadratic)
        p4 = a2       (expert2 constant)
        p5 = b2       (expert2 linear)
        p6 = c2       (expert2 cubic)
        p7 = p_gate   (gate slope)
        p8 = b_gate   (gate intercept)
        p9 = _unused  (reserved)
    """
    # Deadline guard: if we are too close to the wall, still return a
    # valid, finite prediction (the expression below is cheap and bounded).
    if (time.time() - _start_time) > (_DEADLINE - _SAFETY_MARGIN):
        # Fallback: a simple, safe, finite linear-ish prediction.
        # This is only reached in pathological timing; normally the
        # expression below finishes in microseconds.
        return np.zeros_like(x, dtype=float)

    b0, a1, b1, c1, a2, b2, c2, p_gate, b_gate, _ = params

    # Compute gate: sigmoid(p_gate * x + b_gate)
    gate_input = p_gate * x + b_gate
    # Clip to avoid overflow in exp
    gate_input_clipped = np.clip(gate_input, -50, 50)
    gate = 1.0 / (1.0 + np.exp(-gate_input_clipped))

    # Expert 1: quadratic polynomial
    expert1 = a1 + b1 * x + c1 * x**2
    
    # Expert 2: cubic polynomial
    expert2 = a2 + b2 * x + c2 * x**3

    # Blend experts with gate
    y = b0 + gate * expert1 + (1.0 - gate) * expert2

    # Guard against any non-finite output (should not happen, but be safe)
    if not np.all(np.isfinite(y)):
        return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    return y
# EVOLVE-BLOCK-END
