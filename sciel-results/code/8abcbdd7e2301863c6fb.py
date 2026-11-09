import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    target = Fraction(5, 31)
    
    def check_3_terms(M):
        # Find a, b, c such that 7 <= a < b < c <= M and 1/a + 1/b + 1/c = target
        for a in range(7, M + 1):
            if time.time() > deadline:
                return None
            rem_a = target - Fraction(1, a)
            if rem_a <= 0:
                continue
            # Solve 1/b + 1/c = rem_a
            # rem_a = N/D
            N = rem_a.numerator
            D = rem_a.denominator
            # (D*b - N)(D*c - N) = N^2
            # Let X = D*b - N, Y = D*c - N. X*Y = N^2.
            # b = (X+N)/D, c = (Y+N)/D
            # We need b, c integers, a < b < c <= M
            # X < Y implies b < c
            # X < sqrt(N^2) = N
            # Also b > a => (X+N)/D > a => X > a*D - N
            # c <= M => (Y+N)/D <= M => Y <= M*D - N
            
            # Iterate divisors X of N^2
            # To be fast, we can just iterate X up to N
            # But N can be large.
            # However, a is small.
            # Let's just iterate possible b?
            # 1/b < rem_a => b > 1/rem_a
            # 1/b > rem_a / 2 => b < 2/rem_a (since c > b, 1/c < 1/b, so 1/b + 1/c < 2/b => rem_a < 2/b => b < 2/rem_a)
            
            b_min = int(1 / rem_a) + 1
            if b_min <= a:
                b_min = a + 1
            
            b_max = int(2 / rem_a)
            if b_max > M:
                b_max = M
                
            for b in range(b_min, b_max + 1):
                if time.time() > deadline:
                    return None
                rem_b = rem_a - Fraction(1, b)
                if rem_b <= 0:
                    continue
                # c = 1 / rem_b
                # Check if rem_b is a unit fraction 1/c
                if rem_b.numerator == 1:
                    c = rem_b.denominator
                    if c > b and c <= M:
                        return [a, b, c]
        return None

    def check_4_terms(M):
        # Find a, b, c, d such that 7 <= a < b < c < d <= M
        for a in range(7, M + 1):
            if time.time() > deadline:
                return None
            rem_a = target - Fraction(1, a)
            if rem_a <= 0:
                continue
            
            b_min = int(1 / rem_a) + 1
            if b_min <= a:
                b_min = a + 1
            b_max = int(2 / rem_a) # Actually for 3 terms remaining, max b is smaller?
            # 1/b + 1/c + 1/d = rem_a.
            # 1/b < rem_a => b > 1/rem_a
            # 1/b + 1/c + 1/d < 3/b => rem_a < 3/b => b < 3/rem_a
            b_max = int(3 / rem_a)
            if b_max > M:
                b_max = M
                
            for b in range(b_min, b_max + 1):
                if time.time() > deadline:
                    return None
                rem_b = rem_a - Fraction(1, b)
                if rem_b <= 0:
                    continue
                
                c_min = int(1 / rem_b) + 1
                if c_min <= b:
                    c_min = b + 1
                # 1/c + 1/d = rem_b.
                # 1/c < rem_b => c > 1/rem_b
                # 1/c + 1/d < 2/c => rem_b < 2/c => c < 2/rem_b
                c_max = int(2 / rem_b)
                if c_max > M:
                    c_max = M
                
                for c in range(c_min, c_max + 1):
                    if time.time() > deadline:
                        return None
                    rem_c = rem_b - Fraction(1, c)
                    if rem_c <= 0:
                        continue
                    if rem_c.numerator == 1:
                        d = rem_c.denominator
                        if d > c and d <= M:
                            return [a, b, c, d]
        return None

    # Iterate M from 7 upwards
    # We know 1,2,3,4,5,6 are impossible.
    for M in range(7, 1000):
        if time.time() > deadline:
            break
        
        sol = check_3_terms(M)
        if sol:
            return sol
        
        sol = check_4_terms(M)
        if sol:
            return sol
            
    # Fallback: Greedy or simple search if M > 1000
    # But likely M is small.
    # If not found, return a valid large solution
    r = target
    values = []
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values

# EVOLVE-BLOCK-START
def construct():
    import time
    from fractions import Fraction
    
    start_time = time.time()
    deadline = start_time + 85
    
    target = Fraction(5, 31)
    
    def check_3_terms(M):
        for a in range(7, M + 1):
            if time.time() > deadline:
                return None
            rem_a = target - Fraction(1, a)
            if rem_a <= 0:
                continue
            
            b_min = int(1 / rem_a) + 1
            if b_min <= a:
                b_min = a + 1
            b_max = int(2 / rem_a)
            if b_max > M:
                b_max = M
                
            for b in range(b_min, b_max + 1):
                if time.time() > deadline:
                    return None
                rem_b = rem_a - Fraction(1, b)
                if rem_b <= 0:
                    continue
                if rem_b.numerator == 1:
                    c = rem_b.denominator
                    if c > b and c <= M:
                        return [a, b, c]
        return None

    def check_4_terms(M):
        for a in range(7, M + 1):
            if time.time() > deadline:
                return None
            rem_a = target - Fraction(1, a)
            if rem_a <= 0:
                continue
            
            b_min = int(1 / rem_a) + 1
            if b_min <= a:
                b_min = a + 1
            b_max = int(3 / rem_a)
            if b_max > M:
                b_max = M
                
            for b in range(b_min, b_max + 1):
                if time.time() > deadline:
                    return None
                rem_b = rem_a - Fraction(1, b)
                if rem_b <= 0:
                    continue
                
                c_min = int(1 / rem_b) + 1
                if c_min <= b:
                    c_min = b + 1
                c_max = int(2 / rem_b)
                if c_max > M:
                    c_max = M
                
                for c in range(c_min, c_max + 1):
                    if time.time() > deadline:
                        return None
                    rem_c = rem_b - Fraction(1, c)
                    if rem_c <= 0:
                        continue
                    if rem_c.numerator == 1:
                        d = rem_c.denominator
                        if d > c and d <= M:
                            return [a, b, c, d]
        return None

    for M in range(7, 2000):
        if time.time() > deadline:
            break
        sol = check_3_terms(M)
        if sol:
            return sol
        sol = check_4_terms(M)
        if sol:
            return sol

    # Fallback
    r = target
    values = []
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
# EVOLVE-BLOCK-END
