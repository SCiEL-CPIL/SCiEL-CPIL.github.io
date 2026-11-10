"""
Program using 4th-order Yoshida splitting repeated 3 times.
"""
# EVOLVE-BLOCK-START
def coefficients(stages):
    # stages must be 12
    # 4th order symmetric splitting coefficients
    # lambda = 1 / (2 - 2^(1/3))
    import math
    lam = 1.0 / (2.0 - math.pow(2.0, 1.0/3.0))
    
    # Coefficients for one 4th order block starting with B
    # Structure: B A B A B A B
    # Merged into 4 B slots and 4 A slots (last A is 0)
    
    beta_1 = lam / 2.0
    beta_2 = (1.0 - lam) / 2.0
    beta_3 = (1.0 - lam) / 2.0
    beta_4 = lam / 2.0
    
    alpha_1 = lam / 2.0
    alpha_2 = (1.0 - lam) / 2.0
    alpha_3 = lam / 2.0
    alpha_4 = 0.0
    
    # One block uses 4 B coeffs and 4 A coeffs
    b_block = [beta_1, beta_2, beta_3, beta_4]
    a_block = [alpha_1, alpha_2, alpha_3, alpha_4]
    
    # We need 12 stages total.
    # Repeat the block 3 times.
    # Scale by 1/3 to ensure sum is 1.
    scale = 1.0 / 3.0
    
    b_list = []
    a_list = []
    
    for _ in range(3):
        for i in range(4):
            b_list.append(b_block[i] * scale)
            a_list.append(a_block[i] * scale)
            
    return a_list, b_list
# EVOLVE-BLOCK-END
