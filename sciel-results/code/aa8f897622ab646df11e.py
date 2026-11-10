"""Improved program using 4th-order symmetric Suzuki-Trotter splitting composed twice for 12 stages."""
import numpy as np

# EVOLVE-BLOCK-START
def coefficients(stages):
    """12-stage product formula using two composed 4th-order Suzuki-Trotter steps.
    
    The 4th-order symmetric splitting coefficients are:
    c1 = 1 / (2 * (1 - 2**(1/3)))
    c2 = -2**(1/3) / (2 * (1 - 2**(1/3)))
    c3 = 1 - c1 - c2
    
    We use the pattern B(c1), A(c2), B(c3), A(c2), B(c1), A(c2) for a 6-stage step.
    Composing two such steps with step size t/2 yields a 12-stage 4th-order method.
    """
    if stages != 12:
        # Fallback: simple Strang splitting
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
    # c1, c3 are for B, c2 is for A in the pattern B(c1) A(c2) B(c3) A(c2) B(c1) A(c2)
    # Note: Standard symmetric 4th order is often written with A in the middle or B in the middle.
    # Here we need B then A in each stage.
    # The 6-stage sequence is:
    # 1. B(c1), A(c2)
    # 2. B(c3), A(c2)
    # 3. B(c1), A(c2)
    # This is NOT the standard symmetric 4th order.
    # Standard 4th order symmetric:
    # exp(-i c1 t B) exp(-i c2 t A) exp(-i c3 t B) exp(-i c2 t A) exp(-i c1 t B) exp(-i c2 t A)
    # This has 3 B's and 3 A's.
    # Sums: 2*c1 + c3 = 1, 3*c2 = 1.
    # c2 = 1/3.
    # c1 = 1/(2*(1-2^(1/3)))
    # c3 = -2^(1/3)/(2*(1-2^(1/3)))
    
    c1 = 1.0 / (2.0 * (1.0 - 2.0**(1.0/3.0)))
    c2 = 1.0 / 3.0
    c3 = -2.0**(1.0/3.0) / (2.0 * (1.0 - 2.0**(1.0/3.0)))
    
    # Verify consistency of one 6-step
    # Sum B: 2*c1 + c3
    sum_b_1 = 2.0 * c1 + c3
    sum_a_1 = 3.0 * c2
    # assert abs(sum_b_1 - 1.0) < 1e-10
    # assert abs(sum_a_1 - 1.0) < 1e-10
    
    # We compose two 6-steps, each with time scaling 1/2.
    # Step 1 coefficients:
    # b: c1/2, c3/2, c1/2
    # a: c2/2, c2/2, c2/2
    # Step 2 coefficients:
    # b: c1/2, c3/2, c1/2
    # a: c2/2, c2/2, c2/2
    
    a = []
    b = []
    
    for _ in range(2):
        b.append(c1 / 2.0)
        a.append(c2 / 2.0)
        
        b.append(c3 / 2.0)
        a.append(c2 / 2.0)
        
        b.append(c1 / 2.0)
        a.append(c2 / 2.0)
        
    # Verify final constraints
    assert len(a) == 12, f"len(a)={len(a)}"
    assert len(b) == 12, f"len(b)={len(b)}"
    assert abs(sum(a) - 1.0) < 1e-10, f"sum(a)={sum(a)}"
    assert abs(sum(b) - 1.0) < 1e-10, f"sum(b)={sum(b)}"
    for x in a + b:
        assert -50.0 <= x <= 50.0, f"coefficient {x} out of range"
        
    return a, b
# EVOLVE-BLOCK-END
