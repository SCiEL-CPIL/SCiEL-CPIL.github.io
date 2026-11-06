def construct():
    # Hard-coded known solution: 5 Egyptian fractions with denominators > 4
    # 1/5 + 1/6 + 1/7 + 1/42 + 1/140 = 1
    # Verify:
    # LCM of 5,6,7,42,140 = 420
    # 420/5 + 420/6 + 420/7 + 420/42 + 420/140
    # = 84 + 70 + 60 + 10 + 3 = 227... that's not 420.
    
    # Let me recalculate properly.
    # 1/5 + 1/6 + 1/7 + 1/42 + 1/140
    # LCM(5,6,7,42,140) = 420
    # 420/5 = 84
    # 420/6 = 70
    # 420/7 = 60
    # 420/42 = 10
    # 420/140 = 3
    # Sum = 84+70+60+10+3 = 227 ≠ 420
    
    # Let me find a valid 5-term solution.
    # Known: 1 = 1/2 + 1/3 + 1/7 + 1/42 + 1/140? No, 1/2 not allowed.
    
    # Let me use a known identity: 1 = 1/5 + 1/6 + 1/7 + 1/43 + 1/1806? Let me check.
    # Actually, let me just do a quick search at import time for a valid small solution.
    
    # For k=5, let me search:
    # 1 = 1/a + 1/b + 1/c + 1/d + 1/e where 5 <= a < b < c < d < e
    
    # Let me try a=5: 1 - 1/5 = 4/5
    # 4/5 = 1/b + 1/c + 1/d + 1/e
    # Try b=6: 4/5 - 1/6 = 24/30 - 5/30 = 19/30
    # 19/30 = 1/c + 1/d + 1/e
    # Try c=7: 19/30 - 1/7 = 133/210 - 30/210 = 103/210
    # 103/210 = 1/d + 1/e
    # 1/d + 1/e = 103/210
    # (d+e)/(de) = 103/210
    # 210(d+e) = 103de
    # 103de - 210d - 210e = 0
    # (103d - 210)(103e - 210) = 210^2 + 103*210... 
    # Standard: 1/d + 1/e = p/q => (pd - q)(pe - q) = q^2
    # Here p=103, q=210
    # (103d-210)(103e-210) = 210^2 = 44100
    # 210^2 = 44100
    # Factor pairs of 44100 where both factors are congruent to -210 mod 103...
    # 103d - 210 must divide 44100
    # 103d - 210 > 0 => d > 210/103 ≈ 2.04, so d >= 3 (but d > c = 7, so d >= 8)
    # 103*8 - 210 = 824 - 210 = 614
    # Does 614 divide 44100? 44100/614 ≈ 71.8, no.
    # Let me try other c values.
    
    # Try c=8: 19/30 - 1/8 = 76/120 - 15/120 = 61/120
    # 61/120 = 1/d + 1/e
    # (61d-120)(61e-120) = 120^2 = 14400
    # d > 8, so 61d - 120 > 61*9 - 120 = 549 - 120 = 429
    # 429 divides 14400? 14400/429 ≈ 33.57, no.
    # d=10: 610-120=490, 14400/490 ≈ 29.39, no.
    # d=11: 671-120=551, 14400/551 ≈ 26.13, no.
    # d=12: 732-120=612, 14400/612 ≈ 23.53, no.
    # d=13: 793-120=673, 14400/673 ≈ 21.4, no.
    # d=14: 854-120=734, 14400/734 ≈ 19.6, no.
    # d=15: 915-120=795, 14400/795 ≈ 18.1, no.
    # d=16: 976-120=856, 14400/856 ≈ 16.8, no.
    # d=17: 1037-120=917, 14400/917 ≈ 15.7, no.
    # d=18: 1098-120=978, 14400/978 ≈ 14.7, no.
    # d=19: 1159-120=1039, 14400/1039 ≈ 13.86, no.
    # d=20: 1220-120=1100, 14400/1100 ≈ 13.09, no.
    # d=21: 1281-120=1161, 14400/1161 ≈ 12.4, no.
    # d=22: 1342-120=1222, 14400/1222 ≈ 11.78, no.
    # d=23: 1403-120=1283, 14400/1283 ≈ 11.23, no.
    # d=24: 1464-120=1344, 14400/1344 ≈ 10.71, no.
    # d=25: 1525-120=1405, 14400/1405 ≈ 10.25, no.
    # d=26: 1586-120=1466, 14400/1466 ≈ 9.82, no.
    # d=27: 1647-120=1527, 14400/1527 ≈ 9.43, no.
    # d=28: 1708-120=1588, 14400/1588 ≈ 9.07, no.
    # d=29: 1769-120=1649, 14400/1649 ≈ 8.73, no.
    # d=30: 1830-120=1710, 14400/1710 ≈ 8.42, no.
    # d=31: 1891-120=1771, 14400/1771 ≈ 8.13, no.
    # d=32: 1952-120=1832, 14400/1832 ≈ 7.86, no.
    # d=33: 2013-120=1893, 14400/1893 ≈ 7.61, no.
    # d=34: 2074-120=1954, 14400/1954 ≈ 7.37, no.
    # d=35: 2135-120=2015, 14400/2015 ≈ 7.15, no.
    # d=36: 2196-120=2076, 14400/2076 ≈ 6.94, no.
    # d=37: 2257-120=2137, 14400/2137 ≈ 6.74, no.
    # d=38: 2318-120=2198, 14400/2198 ≈ 6.55, no.
    # d=39: 2379-120=2259, 14400/2259 ≈ 6.37, no.
    # d=40: 2440-120=2320, 14400/2320 ≈ 6.21, no.
    # d=41: 2501-120=2381, 14400/2381 ≈ 6.05, no.
    # d=42: 2562-120=2442, 14400/2442 ≈ 5.9, no.
    # d=43: 2623-120=2503, 14400/2503 ≈ 5.75, no.
    # d=44: 2684-120=2564, 14400/2564 ≈ 5.62, no.
    # d=45: 2745-120=2625, 14400/2625 ≈ 5.48, no.
    # d=46: 2806-120=2686, 14400/2686 ≈ 5.36, no.
    # d=47: 2867-120=2747, 14400/2747 ≈ 5.24, no.
    # d=48: 2928-120=2808, 14400/2808 ≈ 5.13, no.
    # d=49: 2989-120=2869, 14400/2869 ≈ 5.02, no.
    # d=50: 3050-120=2930, 14400/2930 ≈ 4.91, no.
    
    # This is getting tedious. Let me just use a known valid solution.
    # 1 = 1/5 + 1/6 + 1/7 + 1/42 + 1/140 is wrong.
    
    # Let me try: 1/5 + 1/6 + 1/8 + 1/9 + 1/10 + 1/180?
    # 1/5+1/6+1/8+1/9+1/10 = 0.2+0.1667+0.125+0.1111+0.1 = 0.7028
    # 1 - 0.7028 = 0.2972, 1/180 = 0.00556, not enough.
    
    # Let me just do a proper search at module load time for k=5.
    from fractions import Fraction
    
    def find_5_term():
        # 1/a + 1/b + 1/c + 1/d + 1/e = 1, 5 <= a < b < c < d < e
        for a in range(5, 15):
            r1 = Fraction(1) - Fraction(1, a)
            for b in range(a+1, 30):
                if Fraction(1, b) >= r1:
                    continue
                r2 = r1 - Fraction(1, b)
                for c in range(b+1, 60):
                    if Fraction(1, c) >= r2:
                        continue
                    r3 = r2 - Fraction(1, c)
                    for d in range(c+1, 200):
                        if Fraction(1, d) >= r3:
                            continue
                        r4 = r3 - Fraction(1, d)
                        if r4 <= 0:
                            continue
                        # Need 1/e = r4, so e = 1/r4 must be integer
                        if r4.denominator == 1:
                            e = r4.numerator
                            if e > d and e <= 10000000:
                                return [a, b, c, d, e]
                        else:
                            # e = 1/r4 = r4.denominator / r4.numerator
                            if r4.denominator % r4.numerator == 0:
                                e = r4.denominator // r4.numerator
                                if e > d and e <= 10000000:
                                    return [a, b, c, d, e]
        return None
    
    sol = find_5_term()
    if sol:
        return sol
    
    # Fallback: use a known 6-term solution if 5-term not found quickly
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + ... 
    # Let me just use a greedy approach as fallback
    rem = Fraction(1)
    denoms = []
    last = 4
    while rem > 0:
        d = last + 1
        while Fraction(1, d) > rem:
            d += 1
        denoms.append(d)
        rem -= Fraction(1, d)
        last = d
        if len(denoms) > 256:
            break
    return denoms
