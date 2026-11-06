"""Product-formula design for Heisenberg-chain time evolution using 4th-order Suzuki-Trotter."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def coefficients(stages):
    """
    4th-order symmetric Suzuki-Trotter product formula for 12 stages.
    
    We use the 4th-order Yoshida decomposition which has 3 B steps and 4 A steps 
    (after combining adjacent A exponentials). We map this to the 12-stage 
    alternating B-A format by inserting zero coefficients.
    """
    # 4th-order Suzuki-Trotter coefficients
    c1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    c2 = -2.0 ** (1.0 / 3.0) / (2.0 - 2.0 ** (1.0 / 3.0))
    
    # The 4th order formula is:
    # exp(-i A_1 t) exp(-i B_1 t) exp(-i A_2 t) exp(-i B_2 t) exp(-i A_3 t) exp(-i B_3 t) exp(-i A_4 t)
    # where:
    # B_1 = c1, B_2 = c2, B_3 = c1
    # A_1 = c1/2
    # A_2 = (c1 + c2)/2
    # A_3 = (c2 + c1)/2
    # A_4 = c1/2
    
    B_coeffs = [c1, c2, c1]
    A_coeffs = [c1 / 2.0, (c1 + c2) / 2.0, (c2 + c1) / 2.0, c1 / 2.0]
    
    a = [0.0] * stages
    b = [0.0] * stages
    
    # Map the 7-step sequence (A, B, A, B, A, B, A) to the 12-stage (B, A) format.
    # Step 1: A_1 -> a[0] (b[0]=0)
    a[0] = A_coeffs[0]
    # Step 2: B_1 -> b[1] (a[1]=0)
    b[1] = B_coeffs[0]
    # Step 3: A_2 -> a[2] (b[2]=0)
    a[2] = A_coeffs[1]
    # Step 4: B_2 -> b[3] (a[3]=0)
    b[3] = B_coeffs[1]
    # Step 5: A_3 -> a[4] (b[4]=0)
    a[4] = A_coeffs[2]
    # Step 6: B_3 -> b[5] (a[5]=0)
    b[5] = B_coeffs[2]
    # Step 7: A_4 -> a[6] (b[6]=0)
    a[6] = A_coeffs[3]
    
    # Normalize to ensure sums are exactly 1 (should be, but floating point might drift)
    sum_a = sum(a)
    sum_b = sum(b)
    
    if abs(sum_a) > 1e-10:
        a = [x / sum_a for x in a]
    if abs(sum_b) > 1e-10:
        b = [x / sum_b for x in b]
        
    # Clip to [-50, 50]
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    
    # Re-normalize after clipping
    sum_a = sum(a)
    sum_b = sum(b)
    if abs(sum_a) > 1e-10:
        a = [x / sum_a for x in a]
    if abs(sum_b) > 1e-10:
        b = [x / sum_b for x in b]
        
    return a, b
# EVOLVE-BLOCK-END
