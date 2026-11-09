def _popcount(x):
    return x.bit_count() if hasattr(int, 'bit_count') else bin(x).count('1')
