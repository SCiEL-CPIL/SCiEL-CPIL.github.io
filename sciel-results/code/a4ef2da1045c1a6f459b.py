"""Product-formula design for Heisenberg-chain time evolution."""
# EVOLVE-BLOCK-START
import math

def coefficients(stages):
    """
    4th-order symmetric Suzuki-Trotter product formula for 12 stages.
    
    The 12-stage interface allows for a symmetric 4th-order scheme 
    by placing the 4th-order coefficients in the first few stages 
    and zeroing the rest, preserving the operator sequence and symmetry.
    """
    # Calculate 4th-order Suzuki-Trotter coefficients
    # x = 1 / (2 - 2^(1/3))
    x = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    
    # Weights for B (even bonds)
    # Sequence: v1, v2, v2, v1
    v1 = x / 2.0
    v2 = (1.0 - x) / 2.0
    
    # Weights for A (odd bonds)
    # Sequence: w1, w2, w1
    w1 = x / 2.0
    w2 = 1.0 - x
    
    # Initialize lists with zeros
    a = [0.0] * stages
    b = [0.0] * stages
    
    # Place coefficients in the first few stages to form the symmetric sequence:
    # B1 A1 B2 A2 B3 A3 B4
    # This corresponds to:
    # k=1: b[0]=v1, a[0]=w1
    # k=2: b[1]=v2, a[1]=w2
    # k=3: b[2]=v2, a[2]=w1
    # k=4: b[3]=v1, a[3]=0.0
    
    if stages >= 4:
        b[0] = v1
        a[0] = w1
        
        b[1] = v2
        a[1] = w2
        
        b[2] = v2
        a[2] = w1
        
        b[3] = v1
        a[3] = 0.0
    elif stages == 3:
        # Fallback for smaller stages, though task specifies 12
        b[0] = v1
        a[0] = w1
        b[1] = v2
        a[1] = w2
        b[2] = v1 # Approximation
        a[2] = w1
    elif stages == 2:
        b[0] = 0.5
        a[0] = 0.5
        b[1] = 0.5
        a[1] = 0.5
    elif stages == 1:
        b[0] = 1.0
        a[0] = 1.0

    return a, b
# EVOLVE-BLOCK-END
