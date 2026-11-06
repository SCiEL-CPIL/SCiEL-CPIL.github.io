# EVOLVE-BLOCK-START
def _splitmix64(x):
    x = (x + 0x9e3779b97f4a7c15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & 0xFFFFFFFFFFFFFFFF
    x = x ^ (x >> 31)
    return x

def priority(vector, n):
    # Convert vector (tuple of 0,1,2) to base-3 integer
    val = 0
    for i in range(n):
        val = val * 3 + vector[i]
    # Mix with SplitMix64 to get a pseudo-random priority
    return float(_splitmix64(val))
# EVOLVE-BLOCK-END
