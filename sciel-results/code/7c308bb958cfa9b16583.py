    POPCOUNT = [bin(i).count('1') for i in range(1 << 16)]
    
    def popcount(x):
        return POPCOUNT[x & 0xFFFF]
