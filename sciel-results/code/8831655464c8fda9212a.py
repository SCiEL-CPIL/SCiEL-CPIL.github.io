"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Optimized coefficients for the growth-index ansatz:
    # gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    # Derived from fitting to CDM and CPL cosmologies.
    # c0: Base offset
    # c1: Linear dependence on q (1-Omega_m)
    # c3: Dependence on u (1+w)
    return [0.96, -0.20, 0.0, -0.34, 0.0, 0.0]
# EVOLVE-BLOCK-END
