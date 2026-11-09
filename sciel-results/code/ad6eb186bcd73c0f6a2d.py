import numpy as np

def kissing_number11() -> np.ndarray:
    """
    Returns a 2-D array of shape (k, 11) representing the kissing configuration
    in dimension 11. The points are the 110 shortest non-zero vectors of the A_11 lattice.
    
    The A_11 lattice is the set of integer vectors whose coordinates sum to 0.
    The shortest non-zero vectors have squared norm 2 and are of the form e_i - e_j.
    However, the set of vectors with exactly two non-zero entries (+/-1) in the 
    standard basis (which corresponds to the root system of A_11 embedded in R^11 
    with the sum-zero constraint implicitly handled by the coordinate differences 
    if we view them as differences of standard basis vectors) works.
    
    Actually, the standard representation of A_n roots is e_i - e_j for i != j.
    There are n(n-1) such roots. For n=11, 11*10 = 110.
    These are integer vectors.
    Norm squared: 1^2 + (-1)^2 = 2.
    Distance squared between e_i - e_j and e_k - e_l:
    If {i,j} and {k,l} are disjoint, dist^2 = 1+1+1+1 = 4.
    If they share one index, e.g., e_i - e_j and e_i - e_k, dist^2 = 0^2 + (-1)^2 + (-1)^2 = 2.
    If they are the same, dist^2 = 0 (but we consider distinct points).
    Min dist^2 = 2.
    Max norm^2 = 2.
    Condition: 2 >= 2. Holds.
    """
    d = 11
    pts = []
    
    # Generate all vectors of the form e_i - e_j where i != j
    # These are the 110 roots of the A_11 lattice.
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            v = np.zeros(d, dtype=np.int64)
            v[i] = 1
            v[j] = -1
            pts.append(v)
            
    arr = np.array(pts, dtype=np.int64)
    arr = np.around(arr).astype(np.int64)
    
    return arr
