"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled as a 4th-order multivariate polynomial in normalized inputs.
    
    Inputs:
        b: population density
        s: substrate concentration
        temp: temperature
        pH: pH level
        params: array of 10 fitted constants
    
    The inputs are normalized to approximately [0, 1] based on training ranges:
        b: [1, 4.998] -> (b - 1) / 3.998
        s: [1, 4.999] -> (s - 1) / 3.999
        temp: [1.002, 39.98] -> (temp - 1.002) / 38.978
        pH: [2, 12] -> (pH - 2) / 10
    
    We construct a 4th-order polynomial with exactly 10 terms chosen to capture
    main effects, key interactions, and higher-order dependencies within the
    10-parameter budget.
    """
    # Normalize inputs to [0, 1] range
    b_norm = (b - 1.0) / 3.998
    s_norm = (s - 1.0) / 3.999
    temp_norm = (temp - 1.002) / 38.978
    ph_norm = (pH - 2.0) / 10.0

    # Clip to avoid extreme values outside training range
    b_norm = np.clip(b_norm, 0.0, 1.0)
    s_norm = np.clip(s_norm, 0.0, 1.0)
    temp_norm = np.clip(temp_norm, 0.0, 1.0)
    ph_norm = np.clip(ph_norm, 0.0, 1.0)

    # 10 polynomial terms:
    # 1. constant
    # 2. b
    # 3. s
    # 4. temp
    # 5. pH
    # 6. b * s
    # 7. b * temp
    # 8. s * temp
    # 9. temp * pH
    # 10. s * pH
    #
    # This allows main effects and key pairwise interactions.
    # For a more aggressive 4th-order approach, we could include:
    # b^2, s^2, temp^2, pH^2, b*s*temp, etc.
    # But with only 10 params, we balance between main effects, interactions, and higher-order.
    
    # Let's use a mix that includes some quadratic terms for non-monotonicity:
    # 1. constant
    # 2. b
    # 3. s
    # 4. temp
    # 5. pH
    # 6. b * s
    # 7. s * temp
    # 8. temp^2  (captures optimal temperature)
    # 9. pH^2    (captures optimal pH)
    # 10. b * temp
    
    term1 = np.ones_like(b_norm)
    term2 = b_norm
    term3 = s_norm
    term4 = temp_norm
    term5 = ph_norm
    term6 = b_norm * s_norm
    term7 = s_norm * temp_norm
    term8 = temp_norm ** 2
    term9 = ph_norm ** 2
    term10 = b_norm * temp_norm

    prediction = (
        params[0] * term1 +
        params[1] * term2 +
        params[2] * term3 +
        params[3] * term4 +
        params[4] * term5 +
        params[5] * term6 +
        params[6] * term7 +
        params[7] * term8 +
        params[8] * term9 +
        params[9] * term10
    )
    
    return prediction
# EVOLVE-BLOCK-END
