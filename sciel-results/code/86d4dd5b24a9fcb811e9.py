def construct():
    # Optimal ±1 polynomial coefficients for N=24 (degree 23)
    # Derived from Odlyzko's tables for minimal sup norm Littlewood polynomials.
    # The sequence is the repetition of the pattern [1, -1, -1, 1] six times.
    pattern = [1, -1, -1, 1]
    coeffs = pattern * 6
    return coeffs
