"""Improved growth index approximation using a compact ansatz."""
# EVOLVE-BLOCK-START
def construct():
    # Improved coefficients based on asymptotic behavior and multi-point fitting.
    # c0: Base gamma for Lambda CDM at z=0 (q~0.7). Value ~0.55.
    # c1: Linear q-dependence. Gamma increases as q decreases (early times).
    #      Estimated slope ~ -0.05 to capture rise from 0.55 to ~0.58 at q=0.
    # c2: Quadratic q-correction. Set to 0 for simplicity, linear fit is robust.
    # c3: Linear u-dependence. Gamma shifts from ~0.55 (u=0) to ~0.60 (u=1, Matter).
    #      Slope ~ 0.05 is more conservative and accurate than 0.11.
    # c4: wa dependence. Set to 0 as wa effects are often secondary or correlated.
    # c5: u*q cross term. Set to 0 to avoid over-fitting with limited terms.
    return [0.55, -0.05, 0.0, 0.05, 0.0, 0.0]
# EVOLVE-BLOCK-END
