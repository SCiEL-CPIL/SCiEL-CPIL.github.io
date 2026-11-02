    # Specific (4,10)-cage construction
    # GF(4) arithmetic
    GF4_add = lambda a, b: a ^ b
    GF4_mul_table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    GF4_mul = lambda a, b: GF4_mul_table[a][b]
    
    # The (4,10)-cage is the L-U-W graph with q=4, d=4
    # d=4 corresponds to girth 2*d + 2 = 10
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    
    q = 4
    d = 4
