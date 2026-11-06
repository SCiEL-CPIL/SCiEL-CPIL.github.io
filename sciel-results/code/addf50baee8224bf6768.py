"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

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
    # S_4(t) = S_2(c1*t) * S_2(c2*t) * S_2(c3*t)
    # where S_2(t) = exp(-i t/2 A) exp(-i t B) exp(-i t/2 A)
    # and c1 = c3 = 1/(2 - 2^(1/3)), c2 = -2^(1/3)/(2 - 2^(1/3))
    
    c = 2.0 ** (1.0 / 3.0)
    c1 = 1.0 / (2.0 - c)
    c2 = -c / (2.0 - c)
    
    # Each S_2 contributes: a_half, b_full, a_half
    # S_4 = S_2(c1) * S_2(c2) * S_2(c3) with c3 = c1
    # So the sequence of (a, b) pairs is:
    # (c1/2, 0) (0, c1) (c1/2, 0) (c2/2, 0) (0, c2) (c2/2, 0) (c1/2, 0) (0, c1) (c1/2, 0)
    # But we need to merge adjacent a terms.
    # Actually, let's just build the list of 12 (a_k, b_k) pairs.
    
    # S_2(c1) gives: (c1/2, 0), (0, c1), (c1/2, 0)
    # S_2(c2) gives: (c2/2, 0), (0, c2), (c2/2, 0)
    # S_2(c1) gives: (c1/2, 0), (0, c1), (c1/2, 0)
    
    # Total 9 pairs, but we need 12. Let me reconsider.
    # The product formula is U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
    # So each stage k has a B part (b_k) and an A part (a_k).
    # The 4th order method can be written as a sequence of 12 such stages.
    
    # Let me use the standard 4th-order Suzuki-Trotter decomposition.
    # The 4th order method for exp(-i t (A+B)) is:
    # exp(-i c1 t/2 A) exp(-i c1 t B) exp(-i c1 t/2 A) 
    # exp(-i c2 t/2 A) exp(-i c2 t B) exp(-i c2 t/2 A)
    # exp(-i c3 t/2 A) exp(-i c3 t B) exp(-i c3 t/2 A)
    # where c1 = c3 = 1/(2 - 2^(1/3)), c2 = -2^(1/3)/(2 - 2^(1/3))
    
    # This gives 9 (a, b) pairs if we write them as:
    # (c1/2, 0), (0, c1), (c1/2, 0), (c2/2, 0), (0, c2), (c2/2, 0), (c3/2, 0), (0, c3), (c3/2, 0)
    
    # But we need 12 stages. We can pad with zeros or use a different decomposition.
    # Actually, looking at the problem statement, it says "twelve-stage product formula"
    # and "U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A) in order k = 1..12"
    
    # So we need exactly 12 (a_k, b_k) pairs. The 4th-order Suzuki-Trotter gives 9 non-zero terms.
    # We can distribute them across 12 stages by adding zeros or splitting terms.
    
    # Let me just use the 9 terms and pad to 12 with zeros.
    a_list = [c1/2, 0, c1/2, c2/2, 0, c2/2, c1/2, 0, c1/2]
    b_list = [0, c1, 0, 0, c2, 0, 0, c3 if 'c3' in dir() else c1, 0]
    
    # Wait, c3 = c1, so let me fix that.
    c3 = c1
    b_list = [0, c1, 0, 0, c2, 0, 0, c3, 0]
    
    # Now pad to 12
    while len(a_list) < 12:
        a_list.append(0.0)
        b_list.append(0.0)
    
    a = a_list[:12]
    b = b_list[:12]
    
    # Ensure sums are 1
    sum_a = sum(a)
    sum_b = sum(b)
    
    # The sums should be 1 for consistency, but let's check.
    # For 4th-order Suzuki-Trotter, the sum of a coefficients should be 1 and sum of b should be 1.
    # Let's verify: sum(a) = c1/2 + c1/2 + c2/2 + c2/2 + c1/2 + c1/2 = c1 + c2 + c1 = 2*c1 + c2
    # c1 = 1/(2-c), c2 = -c/(2-c)
    # 2*c1 + c2 = 2/(2-c) - c/(2-c) = (2-c)/(2-c) = 1. Good.
    # sum(b) = c1 + c2 + c3 = c1 + c2 + c1 = 2*c1 + c2 = 1. Good.
    
    return a, b
# EVOLVE-BLOCK-END
