"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Growth-index coefficients for a smooth CPL dark-energy cosmology.
    # c0 anchors the flat-LCDM value gamma ~ 0.55 (u = 0 there, so the
    # LCDM result is unchanged). c3 and c4 add physically-motivated
    # dark-energy corrections: gamma ~ 0.55 + 0.3*(w0+1) + 0.15*wa*(1-a),
    # expressed through u = 1+w0+wa*(1-a) and the wa*a term.
    return [0.55, 0.0, 0.0, 0.3, 0.15, 0.0]
# EVOLVE-BLOCK-END
