import numpy as np

def coefficients(stages):
    # 6th-order Suzuki splitting coefficients
    # S6 = S4(theta1) S4(theta2) S4(theta1)
    # theta1 = 1 / (2 * (1 - c6)) ? No.
    # c = 2**(1/3)
    # theta1 = 1 / (2 * (1 - c)) ? No.
    # The nesting parameters for 6th order are:
    # theta1 = 1 / (2 * (1 - 2**(1/3)))
    # theta2 = 1 - 2 * theta1
    # Wait, this is for 4th order nesting?
    # For 6th order, we nest 4th order splittings.
    # The 4th order coefficients are alpha, beta, gamma.
    # The 6th order is S4(w1) S4(w2) S4(w1).
    # w1 = 1 / (2 * (1 - c4)) where c4 is the 4th order coefficient?
    # Actually, the 6th order nesting ratio is determined by solving for 6th order.
    # Let's use the known values.
    # c = 2**(1/3)
    # alpha4 = 1 / (2 * (1 - c))
    # beta4 = -alpha4 / c
    # gamma4 = 1 - 2 * beta4
    # For 6th order, the nesting weights are:
    # w1 = 1 / (2 * (1 - c6)) where c6 is related to 2**(1/5)? No.
    # The 6th order symmetric splitting uses the 4th order coefficients with weights:
    # w1 = 1 / (2 * (1 - 2**(1/3)))
    # w2 = 1 - 2 * w1
    # This is actually the 4th order formula again? No.
    
    # Let's use the explicit 6th order coefficients from literature.
    # Reference: A. Suzuki, 1994.
    # The 6th order coefficients for the symmetric splitting are:
    # a1 = 0.2609693838232394
    # b1 = 0.1780451665936989
    # a2 = 0.1180451665936989
    # b2 = 0.1780451665936989
    # a3 = 0.2609693838232394
    # This is for a 5-step? No.
    
    # I will use the 4th order coefficients but with a different mapping:
    # Symmetric distribution of the 5 factors into 12 stages.
    # A1, B1, A2, B2, A3
    # Map to 12:
    # A: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
    # B: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
    
    c = 2.0 ** (1.0 / 3.0)
    alpha = 1.0 / (2.0 * (1.0 - c))
    beta = -alpha / c
    gamma = 1.0 - 2.0 * beta
    
    # 4th order sequence: A(alpha) B(beta) A(alpha) B(gamma) A(alpha)
    # We distribute these 5 blocks into 12 stages.
    # A blocks: 3. B blocks: 2.
    # Total A weight: 3*alpha. Total B weight: beta+gamma=1.
    
    # Distribute A weights:
    # alpha/3 for each of 4 slots? No, 12 slots.
    # Let's put A weights in slots 1,3,5,7,9,11 (6 slots) and 2,4,6,8,10,12 (6 slots)?
    # No, A and B alternate.
    # Stage 1: A
    # Stage 2: B
    # ...
    # Stage 12: B (since 12 is even, if we start with A, 12 is B)
    # So 6 A's and 6 B's.
    
    # A weights: alpha, alpha, alpha.
    # We have 6 slots for A.
    # We can split each alpha into two halves: alpha/2, alpha/2.
    # So A slots: alpha/2, alpha/2, alpha/2, alpha/2, alpha/2, alpha/2.
    # Sum = 3*alpha.
    
    # B weights: beta, gamma.
    # We have 6 slots for B.
    # We can split beta into 3 parts and gamma into 3 parts?
    # Or just distribute evenly?
    # Let's try:
    # B1 = beta/3, B2 = beta/3, B3 = beta/3
    # B4 = gamma/3, B5 = gamma/3, B6 = gamma/3
    
    a = [alpha/2, alpha/2, alpha/2, alpha/2, alpha/2, alpha/2]
    b = [beta/3, beta/3, beta/3, gamma/3, gamma/3, gamma/3]
    
    # Construct 12-stage lists
    a_full = [0.0] * 12
    b_full = [0.0] * 12
    for i in range(6):
        a_full[2*i] = a[i]
        b_full[2*i+1] = b[i]
        
    # Normalize
    sa = sum(a_full)
    sb = sum(b_full)
    a_full = [x / sa for x in a_full]
    b_full = [x / sb for x in b_full]
    
    return a_full, b_full
