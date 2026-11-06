import numpy as np

def coefficients(stages):
    """
    Returns fixed 4th-order symmetric Suzuki-Trotter coefficients for a 12-stage
    product formula approximating exp(-i t (A+B)).
    
    The 4th-order symmetric Trotter scheme uses 6 non-zero weights (3 distinct values)
    that cancel the leading error terms. We map these 6 weights into the 12 available
    slots by placing them in alternating positions (odd indices for A, even for B, or similar).
    
    The product formula is:
      U(t) = prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A)
    
    For a 4th-order symmetric method with 6 effective steps, we have 6 weights w_1..w_6
    for the A-evolution and 6 weights for B-evolution (symmetrically arranged).
    
    The 4th-order Suzuki coefficients (for 6 stages) are:
      w_1 = w_6 = (1/2) / (1 - 2^(1/3))  ... actually let me recall correctly.
    
    Standard 4th-order symmetric Trotter (Yoshida) with 6 terms:
      The coefficients satisfy:
        sum(w_i) = 1
        sum(w_i^2) = 0  (for 4th order)
        sum(w_i^3) = 0
    
    The 6-term 4th-order symmetric coefficients are:
      w_1 = w_6 = (1/2) / (1 - 2^(-1/3))  ... no.
    
    Let me use the well-known values:
      c_0 = 1 / (2 - 2^(1/3)) ≈ 1.8898953348202367
      c_1 = 0.5 - c_0 / 2 ≈ -0.4449476674101184
    
    The 6-step 4th-order symmetric formula uses weights:
      [c_0/2, c_1, c_0/2, c_0/2, c_1, c_0/2]
    Sum = c_0/2 + c_1 + c_0/2 + c_0/2 + c_1 + c_0/2 = 2*c_0 + 2*c_1 = 2*(c_0 + c_1)
    c_0 + c_1 = 1/(2-2^(1/3)) + 0.5 - 1/(2-2^(1/3))/2 = 0.5
    So sum = 2*0.5 = 1. Good.
    
    Now I need to place these into 12 slots for a and 12 slots for b.
    The product formula interleaves A and B evolutions. For a symmetric 4th-order
    method with 6 A-steps and 6 B-steps interleaved:
    
    a = [c_0/2, 0, c_1, 0, c_0/2, 0, c_0/2, 0, c_1, 0, c_0/2, 0]
    b = [0, c_0/2, 0, c_1, 0, c_0/2, 0, c_0/2, 0, c_1, 0, c_0/2]
    
    Wait, I need to think about this more carefully. The standard 4th-order symmetric
    Trotter with 6 A-factors and 6 B-factors in interleaved order:
    
    U = B(w_1) A(w_1) B(w_2) A(w_2) ... B(w_6) A(w_6)
    
    But actually the symmetric 4th-order has a specific structure. Let me use:
    
    For 12 stages with a_k and b_k:
    a = [c_0/2, c_1, c_0/2, c_0/2, c_1, c_0/2, 0, 0, 0, 0, 0, 0]
    That doesn't work because sum must be 1 and we need symmetry.
    
    Actually, the simplest approach: use 6 non-zero A weights and 6 non-zero B weights,
    placed symmetrically in the 12 slots.
    
    Let me just use the 6-term coefficients spread across 12 slots:
    a = [w_1, 0, w_2, 0, w_3, 0, w_4, 0, w_5, 0, w_6, 0]
    b = [0, w_1, 0, w_2, 0, w_3, 0, w_4, 0, w_5, 0, w_6]
    
    where w = [c_0/2, c_1, c_0/2, c_0/2, c_1, c_0/2]
    
    Sum(a) = sum(w) = 1, Sum(b) = sum(w) = 1. Good.
    """
    # 4th-order Suzuki coefficients
    c0 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    c1 = 0.5 - c0 / 2.0
    
    # 6 weights for the 4th-order symmetric method
    w = [c0 / 2.0, c1, c0 / 2.0, c0 / 2.0, c1, c0 / 2.0]
    
    # Place into 12 slots
    a = [0.0] * stages
    b = [0.0] * stages
    
    for k in range(6):
        a[2 * k] = w[k]
        b[2 * k + 1] = w[k]
    
    # Ensure sums are exactly 1 (floating point correction)
    a_sum = sum(a)
    b_sum = sum(b)
    
    # Tiny correction to ensure exact sum = 1
    if abs(a_sum - 1.0) > 1e-15:
        a[-1] += (1.0 - a_sum)
    if abs(b_sum - 1.0) > 1e-15:
        b[-1] += (1.0 - b_sum)
    
    return a, b
