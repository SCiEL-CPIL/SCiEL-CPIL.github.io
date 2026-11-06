import numpy as np

def coefficients(stages):
    """
    Returns coefficients for a 4th-order symmetric Suzuki-Trotter splitting.
    The scheme is S_4(t) = S_2(c1*t) S_2(c2*t) S_2(c3*t)
    where S_2(tau) = exp(-i tau/2 B) exp(-i tau A) exp(-i tau/2 B).
    
    This expands to 7 non-zero exponentials. We map these to the first 7 stages
    of the 12-stage product formula, padding the rest with zeros.
    
    The evaluator computes U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A).
    We align the expanded SSP4 terms with this structure.
    """
    # 4th order Suzuki-Trotter coefficients
    # c1 = c3 = 1 / (2 - 2^(1/3))
    # c2 = -2^(1/3) / (2 - 2^(1/3))
    c1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
    c2 = -2.0**(1.0/3.0) / (2.0 - 2.0**(1.0/3.0))
    c3 = c1
    
    # The expansion is:
    # exp(-i c1/2 t B) exp(-i c1 t A) 
    # exp(-i (c1+c2)/2 t B) exp(-i c2 t A) 
    # exp(-i (c2+c3)/2 t B) exp(-i c3 t A) 
    # exp(-i c3/2 t B)
    
    # Mapping to a (A coeffs) and b (B coeffs) for k=1..12
    # Stage k: exp(-i b_k t B) exp(-i a_k t A)
    
    # Term 1: B coeff c1/2, A coeff 0
    b1 = c1 / 2.0
    a1 = 0.0
    
    # Term 2: B coeff 0, A coeff c1
    b2 = 0.0
    a2 = c1
    
    # Term 3: B coeff (c1+c2)/2, A coeff 0
    b3 = (c1 + c2) / 2.0
    a3 = 0.0
    
    # Term 4: B coeff 0, A coeff c2
    b4 = 0.0
    a4 = c2
    
    # Term 5: B coeff (c2+c3)/2, A coeff 0
    b5 = (c2 + c3) / 2.0
    a5 = 0.0
    
    # Term 6: B coeff 0, A coeff c3
    b6 = 0.0
    a6 = c3
    
    # Term 7: B coeff c3/2, A coeff 0
    b7 = c3 / 2.0
    a7 = 0.0
    
    # Initialize lists with zeros
    a = [0.0] * stages
    b = [0.0] * stages
    
    # Fill in the non-zero coefficients
    a[0] = a1
    b[0] = b1
    a[1] = a2
    b[1] = b2
    a[2] = a3
    b[2] = b3
    a[3] = a4
    b[3] = b4
    a[4] = a5
    b[4] = b5
    a[5] = a6
    b[5] = b6
    a[6] = a7
    b[6] = b7
    
    return a, b
