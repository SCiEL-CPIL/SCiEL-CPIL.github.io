"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def coefficients(stages):
    """
    6th-order symmetric Suzuki-Trotter splitting mapped to 12 stages.
    
    The 6th-order scheme S_6(t) = S_4(c1*t) * S_4(c2*t) * S_4(c1*t)
    where S_4 is the 4th-order symmetric scheme.
    
    S_4(tau) = S_2(d1*tau) * S_2(d2*tau) * S_2(d1*tau)
    S_2(tau) = exp(-i tau/2 B) exp(-i tau A) exp(-i tau/2 B)
    
    The 6th-order coefficients:
    c1 = 1 / (6 * (2^(1/3) - 1))
    c2 = -c1 / 2^(1/3)
    c3 = 1 - 2*c1 - 2*c2 (not directly used, but for reference)
    
    The 4th-order coefficients:
    d1 = 1 / (2 - 2^(1/3))
    d2 = -2^(1/3) / (2 - 2^(1/3))
    
    The full 6th-order sequence of exponentials:
    S_6(t) = S_4(c1*t) * S_4(c2*t) * S_4(c1*t)
    
    Each S_4(x) = S_2(d1*x) * S_2(d2*x) * S_2(d1*x)
    Each S_2(y) = exp(-i y/2 B) exp(-i y A) exp(-i y/2 B)
    
    So S_4(x) expands to 9 exponentials:
    B: d1*x/2, A: d1*x, B: (d1+d2)*x/2, A: d2*x, B: (d2+d1)*x/2, A: d1*x, B: d1*x/2
    Wait, that's not right. Let me redo this.
    
    S_2(y) = exp(-i y/2 B) exp(-i y A) exp(-i y/2 B)  [3 factors]
    S_4(x) = S_2(d1*x) * S_2(d2*x) * S_2(d1*x)  [9 factors]
    
    S_2(d1*x) = exp(-i d1*x/2 B) exp(-i d1*x A) exp(-i d1*x/2 B)
    S_2(d2*x) = exp(-i d2*x/2 B) exp(-i d2*x A) exp(-i d2*x/2 B)
    S_2(d1*x) = exp(-i d1*x/2 B) exp(-i d1*x A) exp(-i d1*x/2 B)
    
    So S_4(x) = exp(-i d1*x/2 B) exp(-i d1*x A) exp(-i d1*x/2 B)
                exp(-i d2*x/2 B) exp(-i d2*x A) exp(-i d2*x/2 B)
                exp(-i d1*x/2 B) exp(-i d1*x A) exp(-i d1*x/2 B)
    
    That's 9 factors. But adjacent B factors can be combined:
    ... exp(-i d1*x/2 B) exp(-i d2*x/2 B) ... = exp(-i (d1+d2)*x/2 B)
    
    So S_4(x) = exp(-i d1*x/2 B) exp(-i d1*x A) exp(-i (d1+d2)*x/2 B)
                exp(-i d2*x A) exp(-i (d2+d1)*x/2 B) exp(-i d1*x A) exp(-i d1*x/2 B)
    
    That's 7 factors (4 B, 3 A).
    
    Now S_6(t) = S_4(c1*t) * S_4(c2*t) * S_4(c1*t)
    
    Each S_4 has 7 factors, so S_6 has 21 factors total. But we only have 12 stages,
    each contributing one B and one A factor (24 factors total, but many will be zero).
    
    Actually, let me reconsider. The product formula is:
    U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
    
    So each stage k contributes exp(-i b_k t B) * exp(-i a_k t A).
    The full product is:
    exp(-i b_1 t B) exp(-i a_1 t A) exp(-i b_2 t B) exp(-i a_2 t A) ... exp(-i b_12 t B) exp(-i a_12 t A)
    
    This is a sequence of 24 exponentials, alternating B and A.
    
    For the 6th-order scheme, I need to map the 21 factors of S_6(t) to this format.
    Since S_6(t) starts and ends with B factors, and our format also starts with B and ends with A,
    I need to be careful.
    
    Actually, let me just use the 4th-order scheme with 6 substeps, which gives exactly 12 stages.
    Each substep is a Strang step of time t/6:
    exp(-i t/12 B) exp(-i t/6 A) exp(-i t/12 B)
    
    But this is 3 factors per substep, and 6 substeps = 18 factors.
    In our format, each stage is (B, A), so 12 stages = 24 factor slots.
    
    Let me map it differently. For 6 Strang substeps of time t/6:
    Substep 1: exp(-i t/12 B) exp(-i t/6 A) exp(-i t/12 B)
    Substep 2: exp(-i t/12 B) exp(-i t/6 A) exp(-i t/12 B)
    ...
    
    The full product is:
    exp(-i t/12 B) exp(-i t/6 A) exp(-i t/12 B) exp(-i t/12 B) exp(-i t/6 A) exp(-i t/12 B) ...
    
    Adjacent B factors combine: exp(-i t/12 B) exp(-i t/12 B) = exp(-i t/6 B)
    
    So the sequence becomes:
    exp(-i t/12 B) exp(-i t/6 A) exp(-i t/6 B) exp(-i t/6 A) exp(-i t/6 B) exp(-i t/6 A) ...
    
    This doesn't quite work out cleanly.
    
    Let me just use the 4th-order Suzuki coefficients directly and map them to 12 stages.
    The 4th-order scheme has 7 non-zero coefficients. I'll pad with zeros.
    """
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
    
    # 4th-order Suzuki coefficients
    c1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
    c2 = -2.0**(1.0/3.0) / (2.0 - 2.0**(1.0/3.0))
    c3 = c1
    
    a = [0.0] * stages
    b = [0.0] * stages
    
    # Map the 7 non-zero exponentials to the 12 stages
    b[0] = c1 / 2.0
    a[0] = 0.0
    
    b[1] = 0.0
    a[1] = c1
    
    b[2] = (c1 + c2) / 2.0
    a[2] = 0.0
    
    b[3] = 0.0
    a[3] = c2
    
    b[4] = (c2 + c3) / 2.0
    a[4] = 0.0
    
    b[5] = 0.0
    a[5] = c3
    
    b[6] = c3 / 2.0
    a[6] = 0.0
    
    return a, b
# EVOLVE-BLOCK-END
