import numpy as np

def coefficients(stages):
    """
    Uses a 4th-order Yoshida splitting scheme repeated 3 times to fill 12 stages.
    The base 4th-order scheme has 4 B-steps and 4 A-steps.
    Repeating it 3 times gives 12 B-steps and 12 A-steps.
    The coefficients are scaled by 1/3 to ensure the total time is 1.
    """
    if stages != 12:
        # Fallback to Strang if stages is not 12
        substeps = stages // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b

    # Yoshida 4th-order coefficients
    # c1 = 1 / (2 * (1 - 2^(1/3)))
    # c2 = -2^(1/3) / (2 * (1 - 2^(1/3)))
    # c3 = c1
    # c4 = 0
    
    root2_3 = 2.0 ** (1.0 / 3.0)
    denom = 2.0 * (1.0 - root2_3)
    c1 = 1.0 / denom
    c2 = -root2_3 / denom
    c3 = c1
    c4 = 0.0
    
    # One block of 4 steps
    b_block = [c1, c2, c3, c4]
    a_block = [1.0, 1.0, 1.0, 1.0]
    
    # Repeat 3 times and scale by 1/3
    b = []
    a = []
    for _ in range(3):
        b.extend([x / 3.0 for x in b_block])
        a.extend([x / 3.0 for x in a_block])
        
    return a, b
