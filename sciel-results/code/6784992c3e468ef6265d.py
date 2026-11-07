def construct():
    """
    Returns 16 distinct integers in [2, 40] such that all 2^16 subset products are distinct.
    
    Strategy: For each prime p, select a subset of {p, p^2, p^3, ...} within [2,40]
    such that the subset sums of exponents are all distinct.
    
    - p=2: {2, 4, 16} (exponents 1, 2, 4) -> subset sums {0,1,2,3,4,5,6,7} all distinct
    - p=3: {3, 9} (exponents 1, 2) -> subset sums {0,1,2,3} all distinct  
    - p=5: {5, 25} (exponents 1, 2) -> subset sums {0,1,2,3} all distinct
    - p>=7: {p} (exponent 1) -> subset sums {0,1} all distinct
    
    Since prime supports are disjoint, all 2^16 total subset products are distinct.
    """
    return [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
