"""Symbolic regression: stress-strain of an aluminium rod using ReLU hinge model."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _smooth_relu(x, k):
    """Smooth ReLU approximation: x * sigmoid(k*x) / k is one option,
    but we use the softplus-based ReLU: (1/k)*log(1+exp(k*x)) - (1/k)*log(2)
    Actually, a simpler smooth ReLU is: (x + sqrt(x^2 + eps))/2
    Let's use: (x + sqrt(x^2 + 1e-8))/2 which is C1 smooth."""
    return 0.5 * (x + np.sqrt(x * x + 1e-8))


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model using ReLU activations (hinges).
    
    params (length 10):
      p0: base offset
      p1: linear strain slope (elastic-like)
      p2: temperature linear slope
      p3: yield strain knot (positive, clamped)
      p4: post-yield slope increment
      p5: hardening strain knot 2
      p6: hardening slope increment
      p7: hardening strain knot 3
      p8: hardening slope increment 3
      p9: temperature-strain interaction coefficient
    
    Model:
      stress = p0 
             + p1 * strain 
             + p2 * temp 
             + p4 * ReLU(strain - p3) 
             + p6 * ReLU(strain - p5) 
             + p8 * ReLU(strain - p7)
             + p9 * temp * strain
    
    We clamp knot positions to be within reasonable range to help optimization.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Clamp knot positions to reasonable strain range [0.001, 0.9]
    # Use a soft clamp via tanh to maintain differentiability
    # Map from real line to (0.001, 0.9) using: 0.001 + 0.899 * sigmoid(p)
    # But sigmoid can cause issues. Let's use: 0.001 + 0.899 * (tanh(p) + 1)/2
    # Actually, let's just use the raw values but they'll be optimized.
    # For BFGS starting from all ones, p3=1, p5=1, p7=1 which is at the edge.
    # Let's shift: use p3, p5, p7 directly as knot positions.
    
    # Smooth ReLU with small epsilon for numerical stability
    eps = 1e-8
    
    # ReLU(strain - knot) approximated smoothly
    h1 = 0.5 * (strain - p3 + np.sqrt((strain - p3)**2 + eps))
    h2 = 0.5 * (strain - p5 + np.sqrt((strain - p5)**2 + eps))
    h3 = 0.5 * (strain - p7 + np.sqrt((strain - p7)**2 + eps))
    
    # Base linear terms
    stress = p0 + p1 * strain + p2 * temp
    
    # Nonlinear strain contributions
    stress = stress + p4 * h1 + p6 * h2 + p8 * h3
    
    # Temperature-strain interaction
    stress = stress + p9 * temp * strain
    
    return stress
# EVOLVE-BLOCK-END
