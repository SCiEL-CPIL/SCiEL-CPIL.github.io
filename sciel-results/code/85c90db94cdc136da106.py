"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Optimized coefficients for the growth-index ansatz:
    # gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    # Fitted to match CDM and CPL dark energy models
    # c0: Base growth index (~0.55-0.65 for standard models)
    # c1: Linear dependence on q (1-Omega_m) - negative as DE dominates
    # c2: Quadratic correction in q
    # c3: Dependence on u (1+w0+wa*(1-a))
    # c4: wa*a correction term
    # c5: Interaction term u*q
    return [0.60, -0.25, 0.05, -0.10, 0.02, 0.03]
# EVOLVE-BLOCK-END
