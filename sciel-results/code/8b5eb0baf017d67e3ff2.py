import time
from fractions import Fraction
import math

def construct():
    target = Fraction(5, 31)
    
    # Start with a known solution
    best_solution = [8, 28, 1736]
    best_max_denom = 1736
    
    # Direct algebraic search for 3-term solutions
    # 1/x + 1/y + 1/z = 5/31 with x < y < z
    # x must satisfy 1/x < 5/31 => x > 6.2 => x >= 7
    # and 3/x > 5/31 => x < 18.6 => x <= 18
    for x in range(7, 19):
        # R = 5/31 - 1/x
        R_num = 5 * x - 31
        R_den = 31 * x
        if R_num <= 0:
            continue
        
        # 1/y + 1/z = R_num / R_den
        # y < 2 / (R_num / R_den) = 2 * R_den / R_num
        # y > x
        max_y = (2 * R_den) // R_num
        
        for y in range(x + 1, max_y + 1):
            # 1/z = R_num/R_den - 1/y = (R_num*y - R_den) / (R_den*y)
            z_num = R_den * y
            z_den = R_num * y - R_den
            if z_den <= 0:
                continue
            if z_num % z_den != 0:
                continue
            z = z_num // z_den
            if z <= y:
                continue
            
            # Found a solution [x, y, z]
            current_max = z
            if current_max < best_max_denom:
                best_max_denom = current_max
                best_solution = [x, y, z]
                
    # Check for 2-term solutions just in case (1/x + 1/y = 5/31)
    # 1/x < 5/31 => x > 6.2 => x >= 7
    # 2/x > 5/31 => x < 62/5 = 12.4 => x <= 12
    for x in range(7, 13):
        R_num = 5 * x - 31
        R_den = 31 * x
        if R_num <= 0:
            continue
        # 1/y = R_num / R_den => y = R_den / R_num
        if R_den % R_num == 0:
            y = R_den // R_num
            if y > x:
                current_max = y
                if current_max < best_max_denom:
                    best_max_denom = current_max
                    best_solution = [x, y]

    # Check for 1-term solution (impossible for 5/31 but good for completeness)
    if target.numerator == 1:
        d = target.denominator
        if d < best_max_denom:
            best_max_denom = d
            best_solution = [d]

    return best_solution
