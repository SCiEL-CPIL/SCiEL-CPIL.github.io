def strassen_terms(n=4):
    # Returns list of (r, c, k, coeff) for n x n x n multiplication
    if n == 1:
        return [(0, 0, 0, 1)]
    
    half = n // 2
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # M1 = (A00+A11)(B00+B11)
    # ...
    
    terms = []
    # This is getting complicated to do recursively with indices.
    # I will hardcode the 49 terms.
    # Or use a known list.
    pass
