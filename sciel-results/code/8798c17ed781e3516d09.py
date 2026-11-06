# EVOLVE-BLOCK-START

def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) % 120 == 0.
    
    Mathematical analysis:
    120 = 2^3 * 3 * 5.
    We need phi(n) to be a multiple of 120.
    
    Check candidates in increasing order:
    - n=1: phi=1
    - n=2: phi=1
    - n=3: phi=2
    - n=4: phi=2
    - n=5: phi=4
    - n=6: phi=2
    - n=7: phi=6
    - n=8: phi=4
    - n=9: phi=6
    - n=10: phi=4
    - n=11: phi=10
    - n=12: phi=4
    - n=13: phi=12
    - n=14: phi=6
    - n=15: phi=8
    - n=16: phi=8
    - n=17: phi=16
    - n=18: phi=6
    - n=19: phi=18
    - n=20: phi=8
    - n=21: phi=12
    - n=22: phi=10
    - n=23: phi=22
    - n=24: phi=8
    - n=25: phi=20
    - n=26: phi=12
    - n=27: phi=18
    - n=28: phi=12
    - n=29: phi=28
    - n=30: phi=8
    - n=31: phi=30
    - n=32: phi=16
    - n=33: phi=20
    - n=34: phi=16
    - n=35: phi=24
    - n=36: phi=12
    - n=37: phi=36
    - n=38: phi=18
    - n=39: phi=24
    - n=40: phi=16
    - n=41: phi=40
    - n=42: phi=12
    - n=43: phi=42
    - n=44: phi=20
    - n=45: phi=24
    - n=46: phi=22
    - n=47: phi=46
    - n=48: phi=16
    - n=49: phi=42
    - n=50: phi=20
    - n=51: phi=32
    - n=52: phi=24
    - n=53: phi=52
    - n=54: phi=18
    - n=55: phi=40
    - n=56: phi=24
    - n=57: phi=36
    - n=58: phi=28
    - n=59: phi=58
    - n=60: phi=16
    - n=61: phi=60
    - n=62: phi=30
    - n=63: phi=36
    - n=64: phi=32
    - n=65: phi=48
    - n=66: phi=20
    - n=67: phi=66
    - n=68: phi=32
    - n=69: phi=44
    - n=70: phi=24
    - n=71: phi=70
    - n=72: phi=24
    - n=73: phi=72
    - n=74: phi=36
    - n=75: phi=40
    - n=76: phi=36
    - n=77: phi=60
    - n=78: phi=24
    - n=79: phi=78
    - n=80: phi=32
    - n=81: phi=54
    - n=82: phi=40
    - n=83: phi=82
    - n=84: phi=24
    - n=85: phi=64
    - n=86: phi=42
    - n=87: phi=56
    - n=88: phi=40
    - n=89: phi=88
    - n=90: phi=24
    - n=91: phi=72
    - n=92: phi=44
    - n=93: phi=60
    - n=94: phi=46
    - n=95: phi=72
    - n=96: phi=32
    - n=97: phi=96
    - n=98: phi=42
    - n=99: phi=60
    - n=100: phi=40
    - n=101: phi=100
    - n=102: phi=32
    - n=103: phi=102
    - n=104: phi=48
    - n=105: phi=48
    - n=106: phi=52
    - n=107: phi=106
    - n=108: phi=36
    - n=109: phi=108
    - n=110: phi=40
    - n=111: phi=72
    - n=112: phi=48
    - n=113: phi=112
    - n=114: phi=36
    - n=115: phi=88
    - n=116: phi=56
    - n=117: phi=72
    - n=118: phi=58
    - n=119: phi=96
    - n=120: phi=32
    - n=121: phi=110
    - n=122: phi=60
    - n=123: phi=80
    - n=124: phi=60
    - n=125: phi=100
    - n=126: phi=36
    - n=127: phi=126
    - n=128: phi=64
    - n=129: phi=84
    - n=130: phi=48
    - n=131: phi=130
    - n=132: phi=40
    - n=133: phi=120  -> 133 = 7*19, phi(133) = 6*18 = 108. Wait, 108 % 120 != 0.
      Let me recheck.
      133 = 7 * 19. phi(133) = 133 * (1 - 1/7) * (1 - 1/19) = 133 * 6/7 * 18/19 = 19 * 6 * 18 / 19 = 6 * 18 = 108. Not divisible by 120.
    - n=134: phi=66
    - n=135: phi=72
    - n=136: phi=64
    - n=137: phi=136
    - n=138: phi=44
    - n=139: phi=138
    - n=140: phi=48
    - n=141: phi=92
    - n=142: phi=70
    - n=143: phi=120  -> 143 = 11 * 13. phi(143) = 143 * (10/11) * (12/13) = 10 * 12 = 120. 120 % 120 == 0.
    
    So n=143 is the first one.
    """
    return 143

# EVOLVE-BLOCK-END
