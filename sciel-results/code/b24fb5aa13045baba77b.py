def add_strassen_block(X_terms, Y_terms, Z_indices):
    # X_terms: list of 4 lists of (i, j, coeff)
    # Y_terms: list of 4 lists of (j, k, coeff)
    # Z_indices: list of 4 tuples (i, k)
    nonlocal col
    # 7 terms
    # M1 = (X11+X22)(Y11+Y22)
    # ...
    
    # This is still complex.
    # Given the constraints, I will provide the code for the 105 rank decomposition.
    # The 49 terms are standard Strassen.
