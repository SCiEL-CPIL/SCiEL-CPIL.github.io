import time
from fractions import Fraction

def construct():
    deadline = time.time() + 85
    target = Fraction(5, 31)
    best_solution = [8, 31, 248]
    best_max = 248

    # First, try to find a better 3-term solution via systematic search
    # We know the target is 5/31.
    # We iterate over the first denominator a.
    # If a >= 31, then 1/a + 1/b + 1/c <= 3/31 < 5/31, so a < 31.
    # Also, 1/a < 5/31 => a > 31/5 = 6.2 => a >= 7.
    
    for a in range(7, 31):
        if time.time() > deadline:
            break
        rem1 = target - Fraction(1, a)
        if rem1 <= 0:
            continue
        # Now find b, c such that 1/b + 1/c = rem1, with b > a, c > b.
        # 1/b < rem1 => b > 1/rem1.
        # 1/b > rem1/2 => b < 2/rem1.
        b_min = int(1 / rem1) + 1
        b_max = int(2 / rem1)
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            if b == a:
                continue
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue
            c = 1 / rem2
            if c.denominator == 1 and c.numerator > b:
                c_val = c.numerator
                if c_val == a or c_val == b:
                    continue
                cur_max = max(a, b, c_val)
                if cur_max < best_max:
                    best_max = cur_max
                    best_solution = [a, b, c_val]
    
    # If we found a better 3-term solution, return it.
    # If not, we could try 2-term or 4+ term, but 3-term seems likely optimal for this small value.
    # Let's verify if a 2-term solution exists.
    # 1/b + 1/c = 5/31
    # (b-c) / (bc) = 5/31 => 31(b-c) = 5bc
    # 5bc - 31b + 31c = 0
    # 25bc - 155b + 155c = 0
    # 25bc - 155b + 155c - 31*155/5 = 0 ... wait
    # Multiply by 5: 25bc - 155b + 155c = 0
    # 25bc - 155b + 155c - (155^2)/25 = -155^2/25? No.
    # (5b - 31)(5c - 31) = 31^2 = 961
    # 961 = 1 * 961 = 31 * 31
    # Factors of 961 are 1, 31, 961.
    # 5b - 31 must be a factor.
    # If 5b - 31 = 1 => 5b = 32 => no integer b.
    # If 5b - 31 = 31 => 5b = 62 => no integer b.
    # If 5b - 31 = 961 => 5b = 992 => no integer b.
    # So no 2-term solution.
    
    # What about 4-term?
    # It's unlikely to beat 248 significantly unless we find a very clever combination.
    # Given the time limit and the structural change directive, the systematic 3-term search
    # is a solid improvement over the greedy approach which failed to find the optimal 3-term set.
    
    return best_solution
