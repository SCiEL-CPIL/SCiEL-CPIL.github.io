def max_tail(lo, hi):
    if hi < lo: return 0.0
    if hi == lo: return 1.0/lo
    return 1.0/lo + math.log(hi/lo)
