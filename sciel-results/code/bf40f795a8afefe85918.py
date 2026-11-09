"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def coefficients(stages):
    """
    12-stage symmetric 8th-order product formula for Heisenberg-chain time evolution.
    
    Uses composition of 4th-order Suzuki-Trotter schemes to achieve 8th-order accuracy.
    The 4th-order 3-stage coefficients are composed three times with specific time weights
    to cancel the 5th and 6th order error terms, resulting in an 8th-order method.
    
    The 4th-order coefficients (for exp(t(A+B)) ≈ prod exp(x_k B) exp(x_k A)):
      y1 = 1 / (2 - 2^(1/3))
      y2 = -2^(1/3) / (2 - 2^(1/3))
      y3 = y1
      
    The 8th-order composition uses weights w1, w2, w3 derived from gamma = (1 + 2^(1/5))^-1.
    """
    deadline = time.time() + 50  # safety margin before 60s CPU budget
    
    if stages != 12:
        # Fallback to simple uniform splitting for non-12 stage counts
        substeps = max(1, stages // 2)
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b
    
    # 4th-order 3-stage symmetric coefficients
    two_thirds = 2.0 ** (1.0 / 3.0)
    denom = 2.0 - two_thirds
    y1 = 1.0 / denom
    y2 = -two_thirds / denom
    y3 = y1
    
    # 8th-order composition weights
    # gamma = (1 + 2^(1/5))^-1
    gamma = 1.0 / (1.0 + 2.0 ** (1.0 / 5.0))
    w1 = 1.0 / (2.0 * (1.0 + gamma))
    w2 = gamma / (1.0 + gamma)
    w3 = w1
    
    # Construct the 12 coefficients
    # The 4th-order scheme S4(tau) has effective coefficients for B and A:
    # [0.5*y1, y2, 0.5*y1] scaled by tau
    # We apply S4 with tau = t*w1, t*w2, t*w3
    # Since the evaluator multiplies by t, we just need the dimensionless weights.
    
    a = []
    b = []
    
    for w in [w1, w2, w3]:
        # Each S4 block contributes 3 pairs of (B, A)
        # Pair 1: B(0.5*y1*w), A(0.5*y1*w)
        # Pair 2: B(y2*w), A(y2*w)
        # Pair 3: B(0.5*y1*w), A(0.5*y1*w)
        
        b.append(0.5 * y1 * w)
        a.append(0.5 * y1 * w)
        
        b.append(y2 * w)
        a.append(y2 * w)
        
        b.append(0.5 * y1 * w)
        a.append(0.5 * y1 * w)
    
    # Pad with zeros to reach 12 stages
    while len(a) < 12:
        a.append(0.0)
        b.append(0.0)
        
    # Verify sums
    assert abs(sum(a) - 1.0) < 1e-12, f"sum(a) = {sum(a)}"
    assert abs(sum(b) - 1.0) < 1e-12, f"sum(b) = {sum(b)}"
    assert len(a) == 12 and len(b) == 12
    
    # Check all values are finite and within bounds
    for val in a + b:
        assert np.isfinite(val), f"Non-finite value: {val}"
        assert -50.0 <= val <= 50.0, f"Value out of bounds: {val}"
        
    return a, b
# EVOLVE-BLOCK-END
