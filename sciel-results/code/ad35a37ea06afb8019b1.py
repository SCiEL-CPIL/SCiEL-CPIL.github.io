"""Product-formula design for Heisenberg-chain time evolution using 4th-order Suzuki-Trotter."""
import numpy as np

# EVOLVE-BLOCK-START
def coefficients(stages):
    """
    4th-order symmetric Suzuki-Trotter product formula for 12 stages.
    
    Uses 3 substeps of the 4th-order Yoshida decomposition.
    The product formula is U(t) = prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A).
    
    Yoshida 4th order coefficients:
    c1 = 1 / (2 - 2**(1/3))
    c2 = -2**(1/3) / (2 - 2**(1/3))
    
    One S4 step (time tau) expands to:
    A: c1/2, (c1+c2)/2, (c1+c2)/2, c1/2
    B: c1, c2, c1
    
    3 substeps of tau = t/3.
    Combining adjacent A's from consecutive steps.
    """
    if stages != 12:
        # Fallback to simple Strang if stages is not 12
        substeps = stages // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b

    # Yoshida 4th order coefficients
    c1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
    c2 = -2.0**(1.0/3.0) / (2.0 - 2.0**(1.0/3.0))
    
    # Coefficients for one substep (normalized to sum 1 for B, sum 1 for A)
    # A coeffs for one step: [c1/2, (c1+c2)/2, (c1+c2)/2, c1/2]
    # B coeffs for one step: [c1, c2, c1]
    
    # We have 3 substeps.
    # Total A coeffs (10 non-zero after combining):
    # Step 1: A1, A2, A3, A4
    # Step 2: A5, A6, A7, A8  (A4 and A5 combine)
    # Step 3: A9, A10, A11, A12 (A8 and A9 combine)
    
    a1 = c1 / 2.0
    a2 = (c1 + c2) / 2.0
    a3 = (c1 + c2) / 2.0
    a4 = c1 / 2.0
    
    # Combined A4 and A5 (start of step 2)
    a_comb1 = a4 + a1 # c1/2 + c1/2 = c1
    
    a5 = a2
    a6 = a3
    a7 = a4
    
    # Combined A7 and A8 (start of step 3)
    a_comb2 = a7 + a1 # c1
    
    a8 = a2
    a9 = a3
    a10 = a4
    
    # The 10 non-zero A coefficients
    A_coeffs = [a1, a2, a3, a_comb1, a5, a6, a_comb2, a8, a9, a10]
    
    # B coeffs for 3 steps (9 non-zero)
    b1 = c1
    b2 = c2
    b3 = c1
    b4 = c1
    b5 = c2
    b6 = c1
    b7 = c1
    b8 = c2
    b9 = c1
    
    B_coeffs = [b1, b2, b3, b4, b5, b6, b7, b8, b9]
    
    # Map to 12 (B, A) pairs
    # Sequence: A1, B1, A2, B2, A3, B3, A4, B4, A5, B5, A6, B6, A7, B7, A8, B8, A9, B9, A10
    # Target: B1, A1, B2, A2, ..., B12, A12
    
    a_out = [0.0] * 12
    b_out = [0.0] * 12
    
    # b1=0, a1=A1
    b_out[0] = 0.0
    a_out[0] = A_coeffs[0]
    
    # b2=B1, a2=A2
    b_out[1] = B_coeffs[0]
    a_out[1] = A_coeffs[1]
    
    # b3=B2, a3=A3
    b_out[2] = B_coeffs[1]
    a_out[2] = A_coeffs[2]
    
    # b4=B3, a4=A4
    b_out[3] = B_coeffs[2]
    a_out[3] = A_coeffs[3]
    
    # b5=B4, a5=A5
    b_out[4] = B_coeffs[3]
    a_out[4] = A_coeffs[4]
    
    # b6=B5, a6=A6
    b_out[5] = B_coeffs[4]
    a_out[5] = A_coeffs[5]
    
    # b7=B6, a7=A7
    b_out[6] = B_coeffs[5]
    a_out[6] = A_coeffs[6]
    
    # b8=B7, a8=A8
    b_out[7] = B_coeffs[6]
    a_out[7] = A_coeffs[7]
    
    # b9=B8, a9=A9
    b_out[8] = B_coeffs[7]
    a_out[8] = A_coeffs[8]
    
    # b10=B9, a10=A10
    b_out[9] = B_coeffs[8]
    a_out[9] = A_coeffs[9]
    
    # b11=0, a11=0
    b_out[10] = 0.0
    a_out[10] = 0.0
    
    # b12=0, a12=0
    b_out[11] = 0.0
    a_out[11] = 0.0
    
    return a_out, b_out
# EVOLVE-BLOCK-END
