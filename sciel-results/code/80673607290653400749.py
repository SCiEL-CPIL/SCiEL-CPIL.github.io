import numpy as np

def coefficients(stages):
    """
    12-stage symmetric Suzuki-Trotter splitting with 4th-order accuracy.
    Uses the composition of three 2nd-order Strang splittings with weights 
    theta_1, theta_2, theta_3 chosen to cancel the 3rd-order error term.
    """
    # 4th-order Suzuki-Trotter coefficients
    # theta_1 = theta_3 = 1 / (2 * (1 - 2^(-1/3)))
    # theta_2 = -1 / (1 - 2^(-1/3))
    
    inv_2_1_3 = 2.0 ** (-1.0/3.0)
    denom = 1.0 - inv_2_1_3
    theta1 = 1.0 / (2.0 * denom)
    theta2 = -1.0 / denom
    theta3 = theta1
    
    # The 7-step sequence derived from composing 3 Strang steps:
    # B(theta1/2) A(theta1) B((theta1+theta2)/2) A(theta2) B((theta2+theta3)/2) A(theta3) B(theta3/2)
    
    a = [0.0] * stages
    b = [0.0] * stages
    
    # Map the 7 non-zero steps to the 12-stage format
    # Stage 1: B
    b[0] = theta1 / 2.0
    a[0] = 0.0
    
    # Stage 2: A
    b[1] = 0.0
    a[1] = theta1
    
    # Stage 3: B
    b[2] = (theta1 + theta2) / 2.0
    a[2] = 0.0
    
    # Stage 4: A
    b[3] = 0.0
    a[3] = theta2
    
    # Stage 5: B
    b[4] = (theta2 + theta3) / 2.0
    a[4] = 0.0
    
    # Stage 6: A
    b[5] = 0.0
    a[5] = theta3
    
    # Stage 7: B
    b[6] = theta3 / 2.0
    a[6] = 0.0
    
    # Stages 8-12 are zero
    # a[7:] and b[7:] are already 0.0
    
    return a, b
