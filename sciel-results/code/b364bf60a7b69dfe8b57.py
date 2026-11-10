"""12-stage product formula using 4x 3rd-order Suzuki-Trotter substeps for Heisenberg-chain time evolution."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def coefficients(stages):
    """12-stage product formula using four identical 3rd-order symmetric Suzuki-Trotter substeps.
    
    Each 3rd-order substep uses 3 stages with coefficients derived from the standard
    3rd-order symmetric splitting constants. The four substeps are concatenated to
    form the 12-stage product formula.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    if stages != 12:
        # Fallback: uniform Strang-like splitting
        n = max(1, stages)
        a = [1.0 / n] * n
        b = [1.0 / n] * n
        return a, b
    
    # 3rd-order symmetric Suzuki-Trotter splitting coefficients
    # c = 1 / (2 * (1 - 2^(-1/3)))
    two_thirds = 2.0 ** (-1.0 / 3.0)
    c = 1.0 / (2.0 * (1.0 - two_thirds))
    # Middle coefficient: 1 - 2*c
    mid = 1.0 - 2.0 * c
    
    # Each 3-stage block has sum = 1, so scale by 1/4 for 4 blocks to sum to 1
    scale = 1.0 / 4.0
    a_block = [c * scale, mid * scale, c * scale]
    b_block = [c * scale, mid * scale, c * scale]
    
    # Concatenate 4 identical blocks
    a = []
    b = []
    for _ in range(4):
        a.extend(a_block)
        b.extend(b_block)
    
    # Verify consistency
    assert len(a) == 12 and len(b) == 12
    assert abs(sum(a) - 1.0) < 1e-10, f"sum(a) = {sum(a)}"
    assert abs(sum(b) - 1.0) < 1e-10, f"sum(b) = {sum(b)}"
    # Ensure all values are finite and in range
    assert all(np.isfinite(x) and -50 <= x <= 50 for x in a + b)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        pass  # Should not happen for this simple computation
    
    return a, b
# EVOLVE-BLOCK-END
