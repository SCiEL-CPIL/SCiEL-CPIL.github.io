import numpy as np

def coefficients(stages):
    """
    12-stage product formula using a 4th-order Suzuki-Trotter splitting.
    The formula is: prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A)
    
    We use the standard 4th-order Suzuki coefficients:
    c1 = 1 / (2 * (1 - 2^(1/3)))
    c2 = -2^(1/3) / (2 * (1 - 2^(1/3)))
    d1 = c2
    d2 = c1
    
    The 4th-order splitting for one step is:
    A(c1) B(d1) A(c2) B(d2) A(c1)
    
    To fit into 12 A and 12 B slots, we use 4 sub-steps of this 4th-order scheme.
    Each sub-step uses 3 A's and 2 B's.
    We map the 4 sub-steps to the 12 slots as follows:
    
    Sub-step 1: A1 B2 A2 B3 A3  (B1 = 0)
    Sub-step 2: A4 B5 A5 B6 A6  (B4 = 0)
    Sub-step 3: A7 B8 A7 B9 A9  (B7 = 0)
    Sub-step 4: A10 B11 A11 B12 A12 (B10 = 0)
    
    Wait, the structure is exp(-i b_k t B) exp(-i a_k t A).
    So the sequence is:
    B1 A1 B2 A2 B3 A3 B4 A4 B5 A5 B6 A6 B7 A7 B8 A8 B9 A9 B10 A10 B11 A11 B12 A12
    
    We want to implement:
    A(c1) B(d1) A(c2) B(d2) A(c1)
    
    This can be written as:
    B(0) A(c1) B(d1) A(c2) B(d2) A(c1) B(0) ...
    
    So for 4 sub-steps:
    B1=0, A1=c1/4, B2=d1/4, A2=c2/4, B3=d2/4, A3=c1/4
    B4=0, A4=c1/4, B5=d1/4, A5=c2/4, B6=d2/4, A6=c1/4
    B7=0, A7=c1/4, B8=d1/4, A8=c2/4, B9=d2/4, A9=c1/4
    B10=0, A10=c1/4, B11=d1/4, A11=c2/4, B12=d2/4, A12=c1/4
    
    Let's verify the sums:
    sum(a) = 4 * (c1/4 + c2/4 + c1/4) = c1 + c2/4 + c1/4? No.
    sum(a) = (c1/4 + c2/4 + c1/4) * 4 = c1 + c2/4 + c1? No.
    sum(a) = 4 * (c1/4 + c2/4 + c1/4) = c1 + c2/4 + c1 = 2*c1 + c2/4?
    
    Wait, the 4th order Suzuki splitting is:
    exp(-i (A+B) t) ≈ exp(-i A c1 t) exp(-i B d1 t) exp(-i A c2 t) exp(-i B d2 t) exp(-i A c1 t)
    
    where c1 = 1/(2(1-2^(1/3))), c2 = -2^(1/3)/(2(1-2^(1/3))), d1 = c2, d2 = c1.
    
    Sum of A coefficients: 2*c1 + c2 = 2/(2(1-2^(1/3))) - 2^(1/3)/(2(1-2^(1/3)))
    = (2 - 2^(1/3))/(2(1-2^(1/3))) = (1 - 2^(1/3))/(1-2^(1/3)) = 1.
    
    Sum of B coefficients: d1 + d2 = c2 + c1 = 1.
    
    So for 4 sub-steps, each scaled by 1/4:
    a = [c1/4, c2/4, c1/4, c1/4, c2/4, c1/4, c1/4, c2/4, c1/4, c1/4, c2/4, c1/4]
    b = [0, d1/4, d2/4, 0, d1/4, d2/4, 0, d1/4, d2/6, 0, d1/4, d2/4]
    
    Wait, let me re-check the mapping.
    The product is: exp(-i b1 t B) exp(-i a1 t A) exp(-i b2 t B) exp(-i a2 t A) ...
    
    We want to group them as:
    (exp(-i b1 t B) exp(-i a1 t A) exp(-i b2 t B) exp(-i a2 t A) exp(-i b3 t B) exp(-i a3 t A))
    (exp(-i b4 t B) exp(-i a4 t A) exp(-i b5 t B) exp(-i a5 t A) exp(-i b6 t B) exp(-i a6 t A))
    ...
    
    We want each group to be: exp(-i A (c1/4) t) exp(-i B (d1/4) t) exp(-i A (c2/4) t) exp(-i B (d2/4) t) exp(-i A (c1/4) t)
    
    This requires:
    b1 = 0, a1 = c1/4
    b2 = d1/4, a2 = c2/4
    b3 = d2/4, a3 = c1/4
    b4 = 0, a4 = c1/4
    b5 = d1/4, a5 = c2/4
    b6 = d2/4, a6 = c1/4
    b7 = 0, a7 = c1/4
    b8 = d1/4, a8 = c2/4
    b9 = d2/4, a9 = c1/4
    b10 = 0, a10 = c1/4
    b11 = d1/4, a11 = c2/4
    b12 = d2/4, a12 = c1/4
    
    Let's check the sums:
    sum(a) = 4 * (c1/4 + c2/4 + c1/4) = c1 + c2/4 + c1? No.
    sum(a) = (c1/4 + c2/4 + c1/4) * 4 = c1 + c2/4 + c1? No.
    sum(a) = 4 * (2*c1/4 + c2/4) = 2*c1 + c2 = 1.
    
    sum(b) = 4 * (0 + d1/4 + d2/4) = d1 + d2 = c2 + c1 = 1.
    
    This is correct.
    """
    c = 2.0 ** (1.0 / 3.0)
    c1 = 1.0 / (2.0 * (1.0 - c))
    c2 = -c / (2.0 * (1.0 - c))
    d1 = c2
    d2 = c1
    
    a = [c1/4.0, c2/4.0, c1/4.0,
         c1/4.0, c2/4.0, c1/4.0,
         c1/4.0, c2/4.0, c1/4.0,
         c1/4.0, c2/4.0, c1/4.0]
    
    b = [0.0, d1/4.0, d2/4.0,
         0.0, d1/4.0, d2/4.0,
         0.0, d1/4.0, d2/4.0,
         0.0, d1/4.0, d2/4.0]
    
    # Normalize to ensure exact sum=1
    sa = sum(a)
    sb = sum(b)
    a = [x / sa for x in a]
    b = [x / sb for x in b]
    
    return a, b
