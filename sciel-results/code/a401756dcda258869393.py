"""Physics: product-formula design for Heisenberg-chain time evolution."""
# EVOLVE-BLOCK-START
import math

def coefficients(stages):
    """Return (a, b) coefficient lists for a 12-stage product formula.
    
    Uses 4th-order symmetric Suzuki-Trotter splitting to minimize
    the spectral-norm error of the time-evolution approximation.
    """
    if stages != 12:
        # Fallback for unexpected stage counts
        n = max(1, stages)
        a = [1.0 / n] * n
        b = [1.0 / n] * n
        return a, b
    
    # 4th-order Suzuki-Trotter: S_4(t) = S_2(c1*t) * S_2(c2*t) * S_2(c3*t)
    # where S_2(t) = exp(-i t/2 A) exp(-i t B) exp(-i t/2 A)
    # c1 = c3 = 1/(2 - 2^(1/3)), c2 = -2^(1/3)/(2 - 2^(1/3))
    
    c = 2.0 ** (1.0 / 3.0)  # 2^(1/3)
    denom = 2.0 - c
    c1 = 1.0 / denom
    c2 = -c / denom
    c3 = c1  # symmetric
    
    # Each S_2 block contributes:
    #   exp(-i (c_k * t / 2) A) exp(-i (c_k * t) B) exp(-i (c_k * t / 2) A)
    # 
    # Concatenating S_2(c1*t) * S_2(c2*t) * S_2(c3*t):
    # The A-exponents between blocks combine:
    #   ... exp(-i c1*t/2 A) exp(-i c1*t B) exp(-i c1*t/2 A)
    #   exp(-i c2*t/2 A) exp(-i c2*t B) exp(-i c2*t/2 A)
    #   exp(-i c3*t/2 A) exp(-i c3*t B) exp(-i c3*t/2 A)
    #
    # Adjacent A factors merge: exp(-i c1*t/2 A) * exp(-i c2*t/2 A) = exp(-i (c1+c2)*t/2 A)
    #
    # The product formula is U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
    # We need to map the Suzuki form to this (B, A) ordering.
    #
    # Suzuki form (A, B ordering): exp(-i a A) exp(-i b B) exp(-i a A) ...
    # Our form:                    exp(-i b B) exp(-i a A) exp(-i b B) exp(-i a A) ...
    #
    # Let's rewrite Suzuki in (B, A) order by re-indexing.
    # S_2(c*t) = exp(-i c*t/2 A) exp(-i c*t B) exp(-i c*t/2 A)
    # 
    # In (B,A) form this is: [exp(-i c*t B)] between two [exp(-i c*t/2 A)]
    # 
    # Full product: 
    #   [A: c1/2] [B: c1] [A: c1/2] [A: c2/2] [B: c2] [A: c2/2] [A: c3/2] [B: c3] [A: c3/2]
    # = [A: c1/2] [B: c1] [A: (c1+c2)/2] [B: c2] [A: (c2+c3)/2] [B: c3] [A: c3/2]
    #
    # Now convert to (B, A) ordering for 12 stages:
    # We have 3 B factors and 4 A factors (with 2 boundary A's).
    # To get 12 (B,A) pairs, we split the A factors across stages.
    #
    # Strategy: distribute the A coefficients across 12 a_k values,
    # and the B coefficients across 12 b_k values.
    #
    # The 12-stage (B,A) product: 
    #   prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A)
    #
    # We want this to equal the Suzuki 4th-order formula.
    # 
    # Suzuki in (B,A) interleaved form (12 stages):
    # Split each of the 3 S_2 blocks into 4 stages:
    # S_2(c*t) = exp(-i c/2 A) exp(-i c B) exp(-i c/2 A)
    #   = [exp(-i 0 B) exp(-i c/2 A)] [exp(-i c B) exp(-i 0 A)] [exp(-i 0 B) exp(-i c/2 A)] [exp(-i 0 B) exp(-i 0 A)]
    # That's not right either. Let me think more carefully.
    #
    # Actually, the simplest approach: use the known optimal 12-step coefficients.
    # For a 12-stage (B,A) product, we can use the 4th-order Suzuki coefficients
    # distributed as follows:
    #
    # The 4th order Suzuki gives 7 non-zero "slots" in (A,B) order:
    # A, B, A, B, A, B, A  with coefficients:
    #   c1/2, c1, (c1+c2)/2, c2, (c2+c3)/2, c3, c3/2
    #
    # In (B,A) order with 12 stages, assign:
    #   Stage 1:  b=0,   a=c1/2
    #   Stage 2:  b=c1,  a=0
    #   Stage 3:  b=0,   a=(c1+c2)/2
    #   Stage 4:  b=c2,  a=0
    #   Stage 5:  b=0,   a=(c2+c3)/2
    #   Stage 6:  b=c3,  a=0
    #   Stage 7:  b=0,   a=c3/2
    #   Stages 8-12: b=0, a=0
    #
    # But this only uses 7 stages. We can redistribute for better accuracy
    # by using all 12 stages with a higher-order scheme.
    #
    # Better approach: Use 6th-order or optimize coefficients.
    # For now, use the 4th-order Suzuki with zero-padding, which is already
    # much better than 2nd-order Strang.
    
    a = [0.0] * 12
    b = [0.0] * 12
    
    # 4th-order Suzuki coefficients in (B,A) stage form
    # Non-zero entries:
    a[0] = c1 / 2.0
    b[1] = c1
    a[2] = (c1 + c2) / 2.0
    b[3] = c2
    a[4] = (c2 + c3) / 2.0
    b[5] = c3
    a[6] = c3 / 2.0
    
    # Verify sum constraint
    sa = sum(a)
    sb = sum(b)
    
    # The sums should be 1.0 by construction:
    # sum(a) = c1/2 + (c1+c2)/2 + (c2+c3)/2 + c3/2 = (c1 + c1 + c2 + c2 + c3 + c3)/2 = (2c1+2c2+2c3)/2 = c1+c2+c3
    # c1 + c2 + c3 = 2*c1 + c2 = 2/denom - c/denom = (2-c)/denom = (2-c)/(2-c) = 1. Good.
    # sum(b) = c1 + c2 + c3 = 1. Good.
    
    return a, b
# EVOLVE-BLOCK-END
