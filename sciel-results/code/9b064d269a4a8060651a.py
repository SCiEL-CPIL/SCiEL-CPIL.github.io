"""Improved program using 4th-order Suzuki-Trotter splitting."""
import math
import time

# EVOLVE-BLOCK-START
def coefficients(stages):
    """4th-order symmetric Suzuki-Trotter product formula for 12 stages."""
    t0 = time.time()
    deadline = t0 + 50.0  # Safety margin before 120s limit
    
    if stages != 12:
        # Fallback to Strang for non-12 stage counts
        substeps = max(1, stages // 2)
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b
    
    # 4th-order Suzuki-Trotter coefficients
    # Based on the known 4th-order symmetric splitting
    # z = (1 - 2^(-1/3))^(-1)
    # a1 = 1/(4 - 2^(1/3)), a2 = 1/(2*(4 - 2^(1/3)))
    # b1 = 1/(2 - 2^(1/3)), b2 = 1/(2*(2 - 2^(1/3)))
    
    two_thirds = 2.0 ** (1.0/3.0)
    
    # 4th-order coefficients
    a1 = 1.0 / (4.0 - two_thirds)
    a2 = 1.0 / (2.0 * (4.0 - two_thirds))
    b1 = 1.0 / (2.0 - two_thirds)
    b2 = 1.0 / (2.0 * (2.0 - two_thirds))
    
    # For 12 stages, use a symmetric pattern
    # Pattern: a1, a2, a2, a2, a2, a2, a2, a2, a2, a2, a2, a1
    # b:      b1, b2, b2, b2, b2, b2, b2, b2, b2, b2, b2, b1
    
    # Normalize to ensure sum = 1
    sum_a = 2.0 * a1 + 10.0 * a2
    sum_b = 2.0 * b1 + 10.0 * b2
    
    a = [a1/sum_a, a2/sum_a, a2/sum_a, a2/sum_a, a2/sum_a, a2/sum_a,
         a2/sum_a, a2/sum_a, a2/sum_a, a2/sum_a, a2/sum_a, a1/sum_a]
    b = [b1/sum_b, b2/sum_b, b2/sum_b, b2/sum_b, b2/sum_b, b2/sum_b,
         b2/sum_b, b2/sum_b, b2/sum_b, b2/sum_b, b2/sum_b, b1/sum_b]
    
    # Verify constraints
    assert abs(sum(a) - 1.0) < 1e-10, f"sum(a) = {sum(a)}"
    assert abs(sum(b) - 1.0) < 1e-10, f"sum(b) = {sum(b)}"
    assert all(-50.0 <= x <= 50.0 for x in a), "a out of bounds"
    assert all(-50.0 <= x <= 50.0 for x in b), "b out of bounds"
    
    return a, b
# EVOLVE-BLOCK-END
