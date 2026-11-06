try:
    (1).bit_count()
    def popcount(x):
        return x.bit_count()
except AttributeError:
    def popcount(x):
        return bin(x).count('1')
