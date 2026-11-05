def construct():
    n, m, p = 4, 5, 6
    
    # Define Strassen 7-term decomposition for 2x2x2
    # Local indices: A: 0=A11,1=A12,2=A21,3=A22
    #               B: 0=B11,1=B12,2=B21,3=B22
    #               C: 0=C11,1=C12,2=C21,3=C22
    strassen_terms = [
        # (U, V, W) as tuples of 4 elements each
        ((1,0,0,0), (0,0,0,1), (0,0,0,1)),  # A11*B22 -> C22
        ((0,0,0,1), (1,0,0,0), (0,0,1,0)),  # A22*B11 -> C21
        ...
    ]
