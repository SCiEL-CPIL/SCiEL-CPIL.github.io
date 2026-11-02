def _bound(coeffs):
    a0, a1, a2, a3 = coeffs
    P = a0 * H0_c + a1 * H4_c + a2 * H8_c + a3 * H12_c
    P2 = P * P
    P4 = P2 * P2
    
    I2 = 0.0
    # P2.c is array of coefficients from highest degree to lowest
    deg = len(P2.c) - 1
    for i, c in enumerate(P2.c):
        power = deg - i
        if power % 2 == 0 and c != 0:
            I2 += c * _int_gaussian_even_power(power // 2, 1.0)
            
    if I2 <= 0:
        return 1e18
        
    I4 = 0.0
    deg4 = len(P4.c) - 1
    for i, c in enumerate(P4.c):
        power = deg4 - i
        if power % 2 == 0 and c != 0:
            I4 += c * _int_gaussian_even_power(power // 2, 2.0)
            
    return I4 / I2
