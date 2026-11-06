"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Analysis:
    We need phi(n) % 120 == 0.
    phi(n) = n * prod_{p|n} (1 - 1/p)
    
    Let's check small n:
    n=1: phi(1) = 1
    n=2: phi(2) = 1
    n=3: phi(3) = 2
    n=4: phi(4) = 2
    n=5: phi(5) = 4
    n=6: phi(6) = 2
    n=7: phi(7) = 6
    n=8: phi(8) = 4
    n=9: phi(9) = 6
    n=10: phi(10) = 4
    n=11: phi(11) = 10
    n=12: phi(12) = 4
    n=13: phi(13) = 12
    n=14: phi(14) = 6
    n=15: phi(15) = 8
    n=16: phi(16) = 8
    n=17: phi(17) = 16
    n=18: phi(18) = 6
    n=19: phi(19) = 18
    n=20: phi(20) = 8
    n=21: phi(21) = 12
    n=22: phi(22) = 10
    n=23: phi(23) = 22
    n=24: phi(24) = 8
    n=25: phi(25) = 20
    n=26: phi(26) = 12
    n=27: phi(27) = 18
    n=28: phi(28) = 12
    n=29: phi(29) = 28
    n=30: phi(30) = 8
    n=31: phi(31) = 30
    n=32: phi(32) = 16
    n=33: phi(33) = 20
    n=34: phi(34) = 16
    n=35: phi(35) = 24
    n=36: phi(36) = 12
    n=37: phi(37) = 36
    n=38: phi(38) = 18
    n=39: phi(39) = 24
    n=40: phi(40) = 16
    n=41: phi(41) = 40
    n=42: phi(42) = 12
    n=43: phi(43) = 42
    n=44: phi(44) = 20
    n=45: phi(45) = 24
    n=46: phi(46) = 22
    n=47: phi(47) = 46
    n=48: phi(48) = 16
    n=49: phi(49) = 42
    n=50: phi(50) = 20
    n=51: phi(51) = 32
    n=52: phi(52) = 24
    n=53: phi(53) = 52
    n=54: phi(54) = 18
    n=55: phi(55) = 40
    n=56: phi(56) = 24
    n=57: phi(57) = 36
    n=58: phi(58) = 28
    n=59: phi(59) = 58
    n=60: phi(60) = 16
    n=61: phi(61) = 60
    n=62: phi(62) = 30
    n=63: phi(63) = 36
    n=64: phi(64) = 32
    n=65: phi(65) = 48
    n=66: phi(66) = 20
    n=67: phi(67) = 66
    n=68: phi(68) = 32
    n=69: phi(69) = 44
    n=70: phi(70) = 24
    n=71: phi(71) = 70
    n=72: phi(72) = 24
    n=73: phi(73) = 72
    n=74: phi(74) = 36
    n=75: phi(75) = 40
    n=76: phi(76) = 36
    n=77: phi(77) = 60
    n=78: phi(78) = 24
    n=79: phi(79) = 78
    n=80: phi(80) = 32
    n=81: phi(81) = 54
    n=82: phi(82) = 40
    n=83: phi(83) = 82
    n=84: phi(84) = 24
    n=85: phi(85) = 64
    n=86: phi(86) = 42
    n=87: phi(87) = 56
    n=88: phi(88) = 40
    n=89: phi(89) = 88
    n=90: phi(90) = 24
    n=91: phi(91) = 72
    n=92: phi(92) = 44
    n=93: phi(93) = 60
    n=94: phi(94) = 46
    n=95: phi(95) = 72
    n=96: phi(96) = 32
    n=97: phi(97) = 96
    n=98: phi(98) = 42
    n=99: phi(99) = 60
    n=100: phi(100) = 40
    ...
    
    The smallest n such that 120 divides phi(n):
    We need phi(n) to be a multiple of 120.
    
    Let's check n=143:
    143 = 11 * 13
    phi(143) = phi(11) * phi(13) = 10 * 12 = 120
    120 % 120 == 0, so n=143 works.
    
    Is there a smaller n?
    We need to check if any n < 143 has phi(n) divisible by 120.
    
    For phi(n) to be divisible by 120 = 8 * 3 * 5:
    - phi(n) must be divisible by 8, 3, and 5.
    
    Let's verify n=143 is indeed the minimum by checking the computed values:
    - n=1 to 100: max phi(n) is 96 (at n=97), none divisible by 120.
    - n=101 to 142:
      phi(101)=100, phi(102)=32, phi(103)=102, phi(104)=48, phi(105)=48,
      phi(106)=52, phi(107)=106, phi(108)=36, phi(109)=108, phi(110)=40,
      phi(111)=72, phi(112)=48, phi(113)=112, phi(114)=36, phi(115)=88,
      phi(116)=56, phi(117)=72, phi(118)=58, phi(119)=108, phi(120)=32,
      phi(121)=110, phi(122)=60, phi(123)=80, phi(124)=60, phi(125)=100,
      phi(126)=36, phi(127)=126, phi(128)=64, phi(129)=84, phi(130)=48,
      phi(131)=130, phi(132)=40, phi(133)=108, phi(134)=66, phi(135)=72,
      phi(136)=64, phi(137)=136, phi(138)=44, phi(139)=138, phi(140)=48,
      phi(141)=92, phi(142)=70
      
    None of these are divisible by 120. So n=143 is indeed the minimum.
    """
    return 143
# EVOLVE-BLOCK-END
