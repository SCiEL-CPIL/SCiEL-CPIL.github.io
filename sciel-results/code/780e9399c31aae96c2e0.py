"""Improved program using high-order symmetric Suzuki-Trotter splitting for Heisenberg-chain time evolution."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def coefficients(stages):
    """12-stage product formula using 4th-order symmetric Suzuki-Trotter splitting.
    
    Uses composed 4th-order Suzuki steps to achieve high accuracy.
    The 4th-order Suzuki-Trotter coefficients are:
    c = 1/(2*(1 - 2**(1/3)))
    d = -2**(1/3)/(2*(1 - 2**(1/3)))
    e = 1 - c - d
    
    We compose 4 sub-steps of the 4th-order splitting to fill 12 stages.
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
    # For exp(-i*t*(A+B)) ≈ product of exp(-i*b_k*t*B)*exp(-i*a_k*t*A)
    # 
    # Standard 4th-order symmetric splitting (6 stages):
    # B(c), A(d), B(e), A(d), B(c), A(e)  -- not quite
    #
    # The correct 4th-order Suzuki-Trotter:
    # c1 = 1/(2*(1 - 2**(1/3)))
    # c2 = -2**(1/3)/(2*(1 - 2**(1/3)))
    # c3 = 1 - c1 - c2
    #
    # Symmetric 6-stage form: B(c1), A(c3), B(c2), A(c3), B(c1), A(c2)
    # Check: sum(b) = c1 + c2 + c1 = 2*c1 + c2
    #        sum(a) = c3 + c3 + c2 = 2*c3 + c2
    # Need both = 1
    
    c1 = 1.0 / (2.0 * (1.0 - 2.0**(1.0/3.0)))
    c2 = -2.0**(1.0/3.0) / (2.0 * (1.0 - 2.0**(1.0/3.0)))
    c3 = 1.0 - c1 - c2
    
    # 6-stage 4th-order symmetric splitting (B,A pattern):
    # Stage 1: B(c1), A(c3)
    # Stage 2: B(c2), A(c3)  
    # Stage 3: B(c1), A(c2)
    # Stage 4: B(c1), A(c2)
    # Stage 5: B(c2), A(c3)
    # Stage 6: B(c1), A(c3)
    # 
    # This is symmetric: stages 1-3 mirror stages 6-4
    # sum(b) = c1+c2+c1+c1+c2+c1 = 4*c1 + 2*c2
    # sum(a) = c3+c3+c2+c2+c3+c3 = 4*c3 + 2*c2
    # 
    # Check: 4*c1 + 2*c2 = 4/(2(1-2^(1/3))) + 2*(-2^(1/3))/(2(1-2^(1/3)))
    # = 2/(1-2^(1/3)) - 2^(1/3)/(1-2^(1/3))
    # = (2 - 2^(1/3))/(1-2^(1/3))
    # = (2 - 2^(1/3))/((1-2^(1/3)))
    # Note: 1 - 2^(1/3) = -(2^(1/3) - 1)
    # 2 - 2^(1/3) = 2(1 - 2^(1/3)/2)
    # This doesn't equal 1. Let me recalculate.
    
    # Actually, the standard 4th-order Suzuki-Trotter for exp(-i*t*(A+B)):
    # S_4(t) = exp(-i*c1*t*B)*exp(-i*c2*t*A)*exp(-i*c3*t*B)*exp(-i*c2*t*A)*exp(-i*c1*t*B)*exp(-i*c2*t*A)
    # where c1 = 1/(2-2^(1/3)), c2 = (1-2^(1/3))/(2-2^(1/3)), c3 = -2^(1/3)/(2-2^(1/3))
    # 
    # Wait, let me use the well-known coefficients:
    # For 4th order: alpha = 1/(2*(1-2^(1/3))), beta = -2^(1/3)/(2*(1-2^(1/3)))
    # The symmetric splitting: B(alpha), A(1-alpha-beta), B(beta), A(1-alpha-beta), B(alpha), A(beta)
    # sum(b) = alpha + beta + alpha = 2*alpha + beta
    # sum(a) = (1-alpha-beta) + (1-alpha-beta) + beta = 2 - 2*alpha - 2*beta + beta = 2 - 2*alpha - beta
    # For consistency: 2*alpha + beta = 1 AND 2 - 2*alpha - beta = 1
    # Both give: 2*alpha + beta = 1. Check: 2/(2(1-2^(1/3))) + (-2^(1/3))/(2(1-2^(1/3)))
    # = (1 - 2^(1/3))/(1-2^(1/3)) = 1. ✓
    # And 2 - 2*alpha - beta = 2 - (2*alpha + beta) = 2 - 1 = 1. ✓
    
    alpha = 1.0 / (2.0 * (1.0 - 2.0**(1.0/3.0)))
    beta = -2.0**(1.0/3.0) / (2.0 * (1.0 - 2.0**(1.0/3.0)))
    gamma = 1.0 - alpha - beta  # This should be the A coefficient in middle steps
    
    # 6-stage 4th-order symmetric splitting:
    # B(alpha), A(gamma), B(beta), A(gamma), B(alpha), A(beta)
    # sum(b) = alpha + beta + alpha = 2*alpha + beta = 1 ✓
    # sum(a) = gamma + gamma + beta = 2*gamma + beta = 2*(1-alpha-beta) + beta = 2-2*alpha-beta = 1 ✓
    
    # For 12 stages, compose 2 such 6-stage splittings (each with half the time):
    # Or better: use the full 6-stage coefficients but scale for 2 sub-steps
    
    # Option 1: Two 6-stage 4th-order splittings with t/2 each
    # a = [gamma/2, gamma/2, beta/2, gamma/2, gamma/2, beta/2, gamma/2, gamma/2, beta/2, gamma/2, gamma/2, beta/2]
    # b = [alpha/2, beta/2, alpha/2, alpha/2, beta/2, alpha/2, alpha/2, beta/2, alpha/2, alpha/2, beta/2, alpha/2]
    # This gives 4th order overall
    
    # Option 2: Use a 12-stage 6th-order splitting
    # For simplicity and reliability, use Option 1 (composed 4th-order)
    
    a = [gamma/2.0, gamma/2.0, beta/2.0, gamma/2.0, gamma/2.0, beta/2.0,
         gamma/2.0, gamma/2.0, beta/2.0, gamma/2.0, gamma/2.0, beta/2.0]
    b = [alpha/2.0, beta/2.0, alpha/2.0, alpha/2.0, beta/2.0, alpha/2.0,
         alpha/2.0, beta/2.0, alpha/2.0, alpha/2.0, beta/2.0, alpha/2.0]
    
    # Verify constraints
    assert len(a) == 12, f"len(a)={len(a)}"
    assert len(b) == 12, f"len(b)={len(b)}"
    assert abs(sum(a) - 1.0) < 1e-10, f"sum(a)={sum(a)}"
    assert abs(sum(b) - 1.0) < 1e-10, f"sum(b)={sum(b)}"
    for x in a + b:
        assert -50.0 <= x <= 50.0, f"coefficient {x} out of range"
    
    return a, b
# EVOLVE-BLOCK-END
