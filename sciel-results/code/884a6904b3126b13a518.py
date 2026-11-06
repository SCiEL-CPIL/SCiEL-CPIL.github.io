"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    target = Fraction(5, 31)
    best_max_denom = float('inf')
    best_solution = None
    
    # Search for 4-term solutions: 1/a + 1/b + 1/c + 1/d = 5/31
    # a < b < c < d
    # We minimize the largest denominator d.
    # To do this efficiently, we iterate a and b, then solve for c and d.
    
    # a must be such that 1/a < 5/31 => a > 6.2 => a >= 7
    # Also 1/a + 1/(a+1) + 1/(a+2) + 1/(a+3) > 5/31
    # For a=7: 1/7+1/8+1/9+1/10 = 0.142+0.125+0.111+0.1 = 0.478 > 0.161.
    # So a can be small.
    
    # Upper bound for a: If a is too large, the sum of 1/a + 1/(a+1) + ... will be too small
    # to reach 5/31 with distinct integers > a.
    # Max sum for 4 terms starting at a is approx 4/a.
    # 4/a > 5/31 => a < 4*31/5 = 24.8.
    # So a is in [7, 24].
    
    max_a = 30  # Safe upper bound
    
    for a in range(7, max_a + 1):
        if time.time() > deadline:
            break
        
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
            
        # Now find 1/b + 1/c + 1/d = rem_a
        # b > a
        # Max sum for 3 terms starting at b is approx 3/b.
        # 3/b > rem_a => b < 3 / rem_a
        # Also b must be > a.
        
        # If rem_a is very small, b must be large.
        # Let's estimate max_b.
        # 1/b + 1/(b+1) + 1/(b+2) > rem_a
        # Approx 3/b > rem_a => b < 3/rem_a
        # Let's set a hard limit for b to keep it fast, but dynamic.
        # If rem_a is small, b is large.
        
        # We want to minimize d.
        # For a given a, b, we solve 1/c + 1/d = rem_b.
        # rem_b = rem_a - 1/b.
        # We iterate c from b+1 upwards.
        # d = rem_b / (1 - c*rem_b) ... no.
        # 1/d = rem_b - 1/c = (rem_b*c - 1)/c
        # d = c / (rem_b*c - 1)
        # For d to be integer, rem_b*c - 1 must divide c.
        # Let rem_b = p/q.
        # d = c / ((p/q)*c - 1) = cq / (pc - q)
        # So (pc - q) must divide cq.
        # Since gcd(pc-q, c) = gcd(-q, c) = gcd(q, c),
        # (pc-q) must divide q * (c / gcd(q,c)).
        
        # Simpler: Iterate c, calculate d, check if integer and d > c.
        
        # Upper bound for b:
        # We need 1/b + 1/(b+1) + 1/(b+2) > rem_a
        # Let's just iterate b up to a reasonable limit.
        # If rem_a is small, b is large.
        # Max possible b: If c, d are very large, 1/b approx rem_a.
        # So b < 1/rem_a.
        # But we need 3 terms. So b < 3/rem_a.
        
        # To ensure we find the global minimum d, we should search b in a way that allows small d.
        # Small d implies small c, which implies small b.
        # So we should prioritize small b.
        
        # Limit b to 200 initially? If rem_a is small, we might need larger b.
        # But if b is large, d will be large.
        # We only care about d < best_max_denom.
        # So we can prune if 1/b + 1/(b+1) + 1/(b+2) < rem_a (impossible)
        # Or if 1/b + 1/(b+1) + 1/(b+2) < rem_a, then no solution for this b.
        # Also if 1/b + 1/(b+1) + 1/(b+2) < rem_a, then b is too large? No, if sum is too small, b is too large.
        # Wait, if 1/b + 1/(b+1) + 1/(b+2) < rem_a, then even with c=b+1, d=b+2, the sum is too small.
        # So we must have 1/b + 1/(b+1) + 1/(b+2) >= rem_a.
        
        # Let's find max_b such that 1/b + 1/(b+1) + 1/(b+2) >= rem_a.
        # Approx 3/b >= rem_a => b <= 3/rem_a.
        
        try:
            max_b_est = int(3 / rem_a) + 10
        except:
            max_b_est = 1000
            
        # Cap max_b to avoid infinite loops if rem_a is very small (which it won't be for small a)
        # But if a is large, rem_a is small.
        # If a=24, 1/24 = 0.041. rem_a = 0.12. 3/0.12 = 25.
        # So b is bounded.
        
        # We also need b > a.
        for b in range(a + 1, min(max_b_est, 500) + 1):
            if time.time() > deadline:
                break
                
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                continue
                
            # Check if 1/c + 1/d = rem_b has a solution with c > b, d > c.
            # Max c: 1/c + 1/(c+1) >= rem_b => 2/c >= rem_b => c <= 2/rem_b.
            try:
                max_c_est = int(2 / rem_b) + 5
            except:
                max_c_est = 500
                
            # We want to minimize d.
            # d = c / (rem_b*c - 1)
            # For d to be small, rem_b*c - 1 should be large? No.
            # d is integer.
            
            # Pruning: If d >= best_max_denom, skip.
            # d = c / (rem_b*c - 1)
            # If c is large, d approx 1/rem_b.
            
            # Iterate c from b+1 to max_c_est
            # But we can also prune c if the minimum possible d (which is c+1) is >= best_max_denom?
            # No, d can be much larger than c+1.
            
            # Let's iterate c.
            # To speed up, we can check if (pc - q) divides cq.
            p, q = rem_b.numerator, rem_b.denominator
            
            # Iterate c
            # We can stop if c is too large such that d would be >= best_max_denom?
            # d = cq / (pc - q)
            # If d >= best_max_denom, we don't care.
            # cq / (pc - q) >= best_max_denom
            # cq >= best_max_denom * (pc - q)
            # c (q - best_max_denom * p) >= - best_max_denom * q
            # This is tricky.
            
            # Just iterate c up to max_c_est.
            # max_c_est is usually small.
            
            for c in range(b + 1, max_c_est + 1):
                if time.time() > deadline:
                    break
                    
                # Calculate d
                # d = c / (rem_b*c - 1)
                # rem_b*c - 1 = (p*c - q) / q
                # d = c * q / (p*c - q)
                
                num = p * c - q
                if num <= 0:
                    continue
                
                # Check divisibility
                if (c * q) % num != 0:
                    continue
                
                d = (c * q) // num
                
                if d <= c:
                    continue
                
                # Check distinctness (a,b,c,d are distinct by construction a<b<c<d)
                # Check bounds
                if d > 10000000:
                    continue
                
                current_max = d
                if current_max < best_max_denom:
                    best_max_denom = current_max
                    best_solution = [a, b, c, d]
                    
                    # If we found a very good solution, we might be able to prune more aggressively
                    # But let's just keep searching for now.
                    
                    # Early break if we found a solution with max denom close to theoretical minimum?
                    # Theoretical min for 4 terms summing to 5/31:
                    # 1/7 + 1/8 + 1/9 + 1/10 = 0.478 (too big)
                    # 1/7 + 1/8 + 1/9 + 1/50 = 0.142+0.125+0.111+0.02 = 0.398 (too big)
                    # We need sum 0.161.
                    # 1/7 + 1/8 + 1/x + 1/y = 0.161
                    # 0.142 + 0.125 = 0.267 > 0.161. So b cannot be 8 if a=7?
                    # Wait. 1/7 = 0.142. 5/31 = 0.161.
                    # rem_a = 0.019.
                    # 1/b must be < 0.019. b > 52.
                    # So if a=7, b must be at least 53.
                    # Then c, d are even larger.
                    # So a=7 leads to large d.
                    
                    # What if a=8?
                    # 1/8 = 0.125. rem_a = 0.036.
                    # 1/b < 0.036 => b > 27.
                    # If b=28, 1/28 = 0.0357. rem_b = 0.0003.
                    # 1/c + 1/d = 0.0003. c, d ~ 3333.
                    # Max denom ~ 3333.
                    
                    # What if a=9?
                    # 1/9 = 0.111. rem_a = 0.050.
                    # 1/b < 0.050 => b > 20.
                    # If b=21, 1/21 = 0.0476. rem_b = 0.0024.
                    # 1/c + 1/d = 0.0024. c ~ 416.
                    
                    # What if a=10?
                    # 1/10 = 0.1. rem_a = 0.061.
                    # 1/b < 0.061 => b > 16.
                    # If b=17, 1/17 = 0.0588. rem_b = 0.0023.
                    # c ~ 434.
                    
                    # It seems max denom will be in the hundreds or thousands.
                    # The current program (3-term) found [8, 28, 1736] max 1736.
                    # A 4-term solution should be better.
                    
                    if best_max_denom < 100:
                        # Very good solution found
                        break
                        # Maybe break outer loops too?
                        # For now, just break inner c loop.
                        
            # If we found a solution with d < 100, we can break b loop
            if best_solution and best_solution[-1] < 100:
                break
                
        # If we found a solution with d < 100, we can break a loop
        if best_solution and best_solution[-1] < 100:
            break
            
    if best_solution:
        return best_solution
    
    # Fallback: 3-term search (from original program) just in case 4-term fails
    # But 4-term should always find something if 3-term does, with smaller max?
    # Not necessarily, but likely.
    
    # Quick fallback: Greedy
    r = target
    values = []
    while r:
        if time.time() > deadline:
            break
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
# EVOLVE-BLOCK-END
