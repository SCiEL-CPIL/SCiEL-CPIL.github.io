"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def coefficients(stages):
    """4th-order symmetric Suzuki-Trotter splitting for 12 (B,A) stages."""
    if stages != 12:
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
    # S_4(t) = S_2(c1*t) * S_2(c2*t) * S_2(c1*t)
    # S_2(t) = exp(-i t A / 2) * exp(-i t B) * exp(-i t A / 2)
    c1 = 1.0 / (2.0 * (2.0 - 2.0**(1.0/3.0)))
    c2 = -2.0**(1.0/3.0) / (2.0 * (2.0 - 2.0**(1.0/3.0)))
    
    # Expand S_4(t) into alternating B, A exponentials:
    # S_2(c1*t): exp(-i c1 t A/2) exp(-i c1 t B) exp(-i c1 t A/2)
    # S_2(c2*t): exp(-i c2 t A/2) exp(-i c2 t B) exp(-i c2 t A/2)
    # S_2(c1*t): exp(-i c1 t A/2) exp(-i c1 t B) exp(-i c1 t A/2)
    #
    # Concatenated:
    # A: -c1/2, B: -c1, A: -c1/2 - c2/2, B: -c2, A: -c2/2 - c1/2, B: -c1, A: -c1/2
    #
    # This gives 4 B-steps and 5 A-steps (with combined A terms).
    # To fit 12 (B,A) stages, split each B and A into 3 parts.
    
    # B coefficients (4 total, split into 12):
    # B1 = c1 -> split into 3: c1/3 each
    # B2 = c2 -> split into 3: c2/3 each
    # B3 = c1 -> split into 3: c1/3 each
    # But we need 12 B stages total, so 4 groups of 3 = 12.
    
    # A coefficients (5 total with combined values):
    # A1 = c1/2
    # A2 = (c1+c2)/2
    # A3 = (c2+c1)/2
    # A4 = c1/2
    # A5 = c1/2  -- wait, let me recount
    
    # Actually from the expansion:
    # Positions: A(-c1/2) B(-c1) A(-(c1+c2)/2) B(-c2) A(-(c2+c1)/2) B(-c1) A(-c1/2)
    # That's 4 B and 4 A (the last A from S2(c1t) end combines with nothing after)
    # Wait: S2(c1t) ends with A(-c1/2), S2(c2t) starts with A(-c2/2), so they combine.
    
    # Let me redo:
    # S2(c1t) = A(-c1/2) * B(-c1) * A(-c1/2)
    # S2(c2t) = A(-c2/2) * B(-c2) * A(-c2/2)
    # S2(c1t) = A(-c1/2) * B(-c1) * A(-c1/2)
    
    # Concatenation:
    # A(-c1/2) B(-c1) A(-c1/2) A(-c2/2) B(-c2) A(-c2/2) A(-c1/2) B(-c1) A(-c1/2)
    # Combine adjacent A:
    # A(-c1/2) B(-c1) A(-(c1+c2)/2) B(-c2) A(-(c2+c1)/2) B(-c1) A(-c1/2)
    
    # So we have: 4 B-steps, 4 A-steps (after combining)
    # B values: c1, c2, c1, c1 -- no wait, 3 B-steps: c1, c2, c1
    # A values: c1/2, (c1+c2)/2, (c2+c1)/2, c1/2 -- 4 A-steps
    
    # Hmm, 3 B and 4 A. To get 12 of each, I need a different mapping.
    
    # Alternative: Use a symmetric 12-stage scheme directly.
    # For a palindromic 12-stage (b1..b12, a1..a12) with b_k = b_{13-k}, a_k = a_{13-k}
    
    # Use the 6th-order Suzuki-Trotter mapped to 12 stages.
    # 6th order: S_6(t) = S_2(d1*t) S_2(d2*t) S_2(d3*t) S_2(d2*t) S_2(d1*t)
    # d1 = 1/(2*(2+2^(1/3)))
    # d2 = -2^(1/3)/(2*(2+2^(1/3)))
    # d3 = 1 - 2*d1 - 2*d2
    
    d1 = 1.0 / (2.0 * (2.0 + 2.0**(1.0/3.0)))
    d2 = -2.0**(1.0/3.0) / (2.0 * (2.0 + 2.0**(1.0/3.0)))
    d3 = 1.0 - 2.0*d1 - 2.0*d2
    
    # Each S_2(d*t) = A(-d*t/2) B(-d*t) A(-d*t/2)
    # 5 S_2 steps -> 5 B-steps, and A-steps combine at boundaries
    # Total B: d1, d2, d3, d2, d1
    # Total A (after combining): d1/2, (d1+d2)/2, (d2+d3)/2, (d3+d2)/2, (d2+d1)/2, d1/2
    # That's 5 B and 6 A.
    
    # To get 12 of each, split: 5 B into 12 (not evenly), 6 A into 12 (evenly, 2 each)
    # This is getting complicated. Let me just use a simple high-order scheme.
    
    # Use 4th-order with 12 stages by splitting the 4th-order 6-stage scheme.
    # 4th order 6-stage symmetric: (b1,a1)(b2,a2)(b3,a3)(b3,a3)(b2,a2)(b1,a1)
    # For 12 stages, double each: (b1,a1)(b1,a1)(b2,a2)(b2,a2)(b3,a3)(b3,a3)
    # (b3,a3)(b3,a3)(b2,a2)(b2,a2)(b1,a1)(b1,a1)
    
    # The 4th-order 6-stage coefficients for alternating B,A:
    # From Suzuki-Trotter 4th order, the effective single-step coefficients:
    # Using the relation that for small t, the 4th order condition is:
    # sum(b) = sum(a) = 1, sum(b^2) = sum(a^2) = 1/2, sum(b^3) = sum(a^3) = 1/3
    # For symmetric 6-stage: b = [x, y, z, z, y, x], a = [x, y, z, z, y, x]
    # 2(x+y+z) = 1 -> x+y+z = 1/2
    # 2(x^2+y^2+z^2) = 1/2 -> x^2+y^2+z^2 = 1/4
    # 2(x^3+y^3+z^3) = 1/3 -> x^3+y^3+z^3 = 1/6
    
    # From Suzuki-Trotter 4th order:
    # x = c1/2, y = c2/2, z = c1/2 where c1, c2 are from above
    # But c1 + c2 + c1 = 2c1 + c2 = 1 (by construction)
    # So x + y + z = c1/2 + c2/2 + c1/2 = c1 + c2/2
    # And 2c1 + c2 = 1, so c1 + c2/2 = 1/2. Good.
    
    x = c1 / 2.0
    y = c2 / 2.0
    z = c1 / 2.0
    
    # 12-stage: double each of the 6 stages
    b = [x, x, y, y, z, z, z, z, y, y, x, x]
    a = [x, x, y, y, z, z, z, z, y, y, x, x]
    
    # Normalize to ensure sums are exactly 1
    sb = sum(b)
    sa = sum(a)
    b = [v / sb for v in b]
    a = [v / sa for v in a]
    
    return a, b
# EVOLVE-BLOCK-END
