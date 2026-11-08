    # W mapping: 4x4 W row index 4k'+i -> 4x8 W row index 4k+i
    for i in range(4):
        for k_prime in range(4):
            W[4*k_prime + i, 0:49] = W4[4*k_prime + i, 0:49]
            
    # Second instance
    for i in range(4):
        for k_prime in range(4):
            W[4*(k_prime+4) + i, 49:98] = W4[4*k_prime + i, 0:49]
