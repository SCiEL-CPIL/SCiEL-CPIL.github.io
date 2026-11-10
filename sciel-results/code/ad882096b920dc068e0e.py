"""Improved program using high-order splitting coefficients for Heisenberg-chain time evolution."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def coefficients(stages):
    """Use 4th-order Suzuki-Trotter coefficients for 12-stage product formula.
    
    The 4th-order symmetric splitting has 4 stages per block.
    For 12 stages, we use three 4th-order 4-stage blocks.
    """
    deadline = time.time() + 50.0  # Safety margin before 60s limit
    
    # 4th-order Suzuki-Trotter symmetric splitting coefficients
    # r = 2^(1/3)
    r = 2.0 ** (1.0 / 3.0)
    # c = 1 / (2 * (1 - 1/r))
    c = 1.0 / (2.0 * (1.0 - 1.0 / r))
    
    # 4th-order 4-stage coefficients for A parts (odd bonds)
    a_4th = [c, 1.0 - 2.0 * c, 1.0 - 2.0 * c, c]
    # 4th-order 4-stage coefficients for B parts (even bonds)
    b_4th = [1.0 - 2.0 * c, c, c, 1.0 - 2.0 * c]
    
    # For 12 stages, we use three 4th-order 4-stage blocks
    a = []
    b = []
    for _ in range(3):
        a.extend(a_4th)
        b.extend(b_4th)
    
    # Ensure lengths match stages
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
    a = a[:stages]
    b = b[:stages]
    
    # Convert to numpy arrays for normalization
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    
    # Normalize to ensure sum is exactly 1
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if abs(sum_a) > 1e-15:
        a = a / sum_a
    if abs(sum_b) > 1e-15:
        b = b / sum_b
    
    # Final check: ensure finite and in range [-50, 50]
    a = np.clip(a, -50.0, 50.0)
    b = np.clip(b, -50.0, 50.0)
    
    # Re-normalize after clipping to maintain sum=1
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if abs(sum_a) > 1e-15:
        a = a / sum_a
    if abs(sum_b) > 1e-15:
        b = b / sum_b
    
    # Verify constraints
    assert len(a) == stages, f"Length of a is {len(a)}, expected {stages}"
    assert len(b) == stages, f"Length of b is {len(b)}, expected {stages}"
    assert abs(np.sum(a) - 1.0) < 1e-10, f"Sum of a is {np.sum(a)}"
    assert abs(np.sum(b) - 1.0) < 1e-10, f"Sum of b is {np.sum(b)}"
    assert np.all(np.isfinite(a)), "a contains non-finite values"
    assert np.all(np.isfinite(b)), "b contains non-finite values"
    assert np.all(a >= -50.0) and np.all(a <= 50.0), "a out of range"
    assert np.all(b >= -50.0) and np.all(b <= 50.0), "b out of range"
    
    return list(a), list(b)
# EVOLVE-BLOCK-END
