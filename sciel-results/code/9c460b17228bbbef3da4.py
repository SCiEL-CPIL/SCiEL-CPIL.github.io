import math
import time

def construct():
    """
    Returns a list of distinct integers strictly larger than 210 and at most 10210
    such that 210 * product(list) is a perfect square.
    Minimizes max(list) - 210.
    """
    # The known optimal solution found by analysis
    # 224 = 16 * 14, sf(224) = 14
    # 240 = 16 * 15, sf(240) = 15
    # 14 * 15 = 210
    # 210 * 224 * 240 = 210 * (14 * 15) = 210 * 210 = 210^2
    # Max is 240. Value is 30.
    
    # Verify constraints
    result = [224, 240]
    
    # Check distinct
    if len(set(result)) != len(result):
        # Fallback to a larger valid set if something goes wrong (should not happen)
        return [210 * 4] # 840, sf=1? No. 840=8*105. sf=105. 210*840 = 210*8*105 = 210*8*210/2 = 210^2 * 4. Square.
        
    # Check bounds
    for x in result:
        if not (210 < x <= 10210):
            return [840]
            
    # Check square condition
    prod = 210
    for x in result:
        prod *= x
        
    sqrt_val = math.isqrt(prod)
    if sqrt_val * sqrt_val != prod:
        # Fallback
        return [840]
        
    return result
