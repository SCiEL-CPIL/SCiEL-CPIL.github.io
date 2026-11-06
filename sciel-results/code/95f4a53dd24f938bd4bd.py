"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def coefficients(stages):
    """4th-order Suzuki-Trotter splitting mapped to 12 stages."""
    # Standard 4th-order symmetric splitting coefficients
    # c1, c2, c3 for the structure B_c3 A_c2 B_c1 A_c1 B_c2 A_c3
    # These coefficients sum to 1/2 in some conventions, or 1 in others.
    # We calculate them and normalize to sum to 1.
    
    # Using the standard Suzuki-Trotter 4th order coefficients
    # c1 = 1 / (2 * (2 - 2^(1/3)))
    # c2 = -2^(1/3) / (2 * (2 - 2^(1/3)))
    # c3 = (1 - 2^(1/3)) / (2 - 2^(1/3))
    
    c_root = 2.0 ** (1.0 / 3.0)
    denom = 2.0 - c_root
    
    c1 = 1.0 / (2.0 * denom)
    c2 = -c_root / (2.0 * denom)
    c3 = (1.0 - c_root) / denom
    
    # The sum of these coefficients is S. We need the sum to be 1.
    S = c1 + c2 + c3
    if abs(S) < 1e-10:
        # Fallback to Strang if something went wrong
        a = [0.5] * stages
        b = [0.5] * stages
        return a, b
        
    c1_norm = c1 / S
    c2_norm = c2 / S
    c3_norm = c3 / S
    
    a = [0.0] * stages
    b = [0.0] * stages
    
    # Map to the first 3 stages of the 12-stage structure
    # Structure: B1 A1 B2 A2 B3 A3 ...
    # Suzuki:    Bc3 Ac2 Bc1 Ac1 Bc2 Ac3
    
    if stages >= 3:
        b[0] = c3_norm
        a[0] = c2_norm
        b[1] = c1_norm
        a[1] = c1_norm
        b[2] = c2_norm
        a[2] = c3_norm
        
    # If stages > 3, we can distribute the remaining error or keep them 0.
    # Keeping them 0 is valid as exp(0) = 1.
    
    return a, b
# EVOLVE-BLOCK-END
