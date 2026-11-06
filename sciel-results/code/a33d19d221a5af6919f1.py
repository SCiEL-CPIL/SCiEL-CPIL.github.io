import time

def construct():
    # Construct a valid set of size 64: {1, ..., 13} U {50, ..., 100}
    # Max sum of smalls: 91
    # Min sum of 2 tails: 50+51=101
    # 91 + 101 = 192 < 200
    # So no subset of smalls + 2 tails can sum to 200.
    # Max sum of smalls: 91
    # Min tail: 50
    # 91 + 50 = 141 < 200
    # So we only need to check if smalls + 1 tail = 200.
    # S_small + y = 200 => S_small = 200 - y
    # y in [50, 100] => 200 - y in [100, 150]
    # Max S_small = 91
    # So S_small < 100, never equals 200 - y.
    # Thus, no subset sums to 200.
    
    smalls = list(range(1, 14))
    tails = list(range(50, 101))
    return smalls + tails

