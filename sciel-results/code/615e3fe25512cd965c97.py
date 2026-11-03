<<<<<<< SEARCH
    # Bounds: roots must be positive
    bounds = [(1e-6, None) for _ in range(12)]
=======
    # Bounds: roots must be positive and within a reasonable range
    # Laguerre roots of degree 24 are roughly in [0, 50]
    bounds = [(1e-4, 50.0) for _ in range(12)]
>>>>>>> REPLACE
