def construct():
    from math import isqrt
    return [a for a in range(2,P['n']+1) if all(a%d for d in range(2,isqrt(a)+1))]
