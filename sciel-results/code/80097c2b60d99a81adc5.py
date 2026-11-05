import math
from functools import reduce

def construct():
    target = 255
    # Find smallest prime factor of target
    spf = None
    d = 2
    x = target
    while d * d <= x:
        if x % d == 0:
            spf = d
            break
        d += 1
    if spf is None:
        spf = target

    # Build the maximal pairwise-noncoprime set: all multiples of spf in [1, target]
    multiples = [spf * k for k in range(1, target // spf + 1)]

    # Ensure target is present (it should be, since spf divides target)
    if target not in multiples:
        multiples.append(target)
        multiples.sort()

    return multiples
