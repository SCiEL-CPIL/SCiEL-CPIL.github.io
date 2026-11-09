"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Growth-index coefficients for CPL dark energy:
    # gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    # where q = 1 - Om(a), u = 1 + w0 + wa*(1-a)
    # Base 0.55 for LambdaCDM; corrections for varying w.
    c0 = 1.0
    c1 = -0.50
    c2 = -0.20
    c3 = 0.10
    c4 = 0.0
    c5 = 0.0
    return [c0, c1, c2, c3, c4, c5]
# EVOLVE-BLOCK-END
