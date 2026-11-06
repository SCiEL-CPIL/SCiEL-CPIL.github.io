def popcount(x):
    return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')
