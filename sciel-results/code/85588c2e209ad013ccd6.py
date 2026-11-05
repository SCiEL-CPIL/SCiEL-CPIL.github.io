try:
    (0).bit_count()
    def pc(x): return x.bit_count()
except AttributeError:
    def pc(x): return bin(x).count('1')
