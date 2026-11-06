import math
import time

def construct():
    deadline = time.time() + 50  # 50 second safety margin (limit is 60s)
    answers = [None] * 254  # indices 0..253 correspond to n=3..256
    
    # Precompute divisors for numbers up to a reasonable limit? 
    # No, we need divisors of n*x which can be large.
    # Instead, we iterate x, compute A, B, then find divisors of B.
    
    # To speed up divisor finding, we can just loop d up to sqrt(B).
    # But B = n*x. Max n=256, x approx 3n/4 = 192. B approx 256*192 = 49152.
    # Wait, x can be larger?
    # 1/x + 1/y + 1/z = 4/n
    # If x < n/4, 1/x > 4/n, sum > 4/n. So x > n/4.
    # If x >= 3n/4, then 1/x <= 4/(3n).
    # 1/y + 1/z = 4/n - 1/x >= 4/n - 4/(3n) = 8/(3n).
    # Since y >= x >= 3n/4, 1/y <= 4/(3n), 1/z <= 4/(3n).
    # So 1/y + 1/z <= 8/(3n).
    # This implies y and z must be close to x.
    # Specifically, if x = 3n/4, 1/y + 1/z = 8/(3n).
    # Max sum for y=z=x is 2/x = 8/(3n). So y=z=x=3n/4 is a solution IF 3n/4 is integer.
    # But we need distinct denominators.
    # Generally, x is in [floor(n/4)+1, floor(3n/4)].
    # For n=256, x in [64, 192].
    # B = n*x <= 256*192 = 49152.
    # sqrt(49152) is about 221.
    # So finding divisors of B is very fast (loop up to 221).
    
    # Let's refine the bounds for x.
    # We need 1/x < 4/n => x > n/4.
    # We need 1/x + 1/y + 1/z = 4/n with x < y < z.
    # Max value of 1/y + 1/z is when y and z are as small as possible.
    # y > x, z > y.
    # So 1/y < 1/x, 1/z < 1/x.
    # 4/n - 1/x < 2/x => 4/n < 3/x => x < 3n/4.
    # So x is in [floor(n/4)+1, floor(3n/4)-1] or similar.
    # Actually, if x = 3n/4, then 4/n - 1/x = 4/n - 4/(3n) = 8/(3n).
    # We need 1/y + 1/z = 8/(3n) with y > x = 3n/4.
    # Max 1/y + 1/z is just under 2/x = 8/(3n).
    # So it's possible.
    # Let's just iterate x from floor(n/4)+1 to floor(3n/4).
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
            
        found = None
        start_x = n // 4 + 1
        end_x = (3 * n) // 4
        
        # We want distinct x, y, z.
        # To ensure distinctness and avoid duplicates in search, we can enforce x < y < z.
        # Then we return [x, y, z] or any permutation? The task says "three DISTINCT positive integer denominators".
        # Order doesn't matter for the set, but the output is a list of 3.
        # The evaluator checks 1/x+1/y+1/z=4/n.
        # So any permutation works.
        
        for x in range(start_x, end_x + 1):
            A = 4 * x - n
            B = n * x
            
            if A <= 0:
                continue
                
            # We need 1/y + 1/z = A/B
            # B(y+z) = A y z
            # A y z - B y - B z = 0
            # Multiply by A:
            # A^2 y z - A B y - A B z = 0
            # (A y - B)(A z - B) = B^2
            # Let U = A y - B, V = A z - B.
            # U V = B^2.
            # U < V since y < z.
            # Also y = (U+B)/A, z = (V+B)/A.
            # We need y and z to be integers.
            # So (U+B) % A == 0 and (V+B) % A == 0.
            # Also y > x => (U+B)/A > x => U+B > A x => U > A x - B = (4x-n)x - nx = 4x^2 - 2nx = 2x(2x-n).
            # And z > y => V > U.
            # And y, z distinct from x.
            
            # Iterate over divisors U of B^2 such that U < V.
            # B^2 can be large? B ~ 50000, B^2 ~ 2.5 * 10^9.
            # sqrt(B^2) = B.
            # So we iterate U from 1 to B.
            # This is O(B) per x.
            # Sum of B over x?
            # B = n*x.
            # Sum n*x for x in range ~ n * n^2/32 ~ n^3/32.
            # For n=256, 256^3/32 = 524288.
            # Total over all n: sum n^3 ~ 256^4/4 = 4e9. Too slow in Python?
            # 4e9 iterations is too slow.
            
            # Optimization:
            # We only need U such that (U+B) % A == 0.
            # U = k A - B.
            # We need U V = B^2 => (k A - B)(l A - B) = B^2.
            # This is equivalent to finding factors of B^2.
            
            # Alternative approach:
            # Iterate divisors d of B.
            # Wait, U and V are factors of B^2.
            # If we iterate divisors d of B, does that help?
            # Not directly.
            
            # Let's stick to iterating U up to B.
            # But we can break early if U > B (since U < V and UV=B^2, U <= B).
            # Actually U must be <= B.
            
            # Can we reduce the range?
            # U > 2x(2x-n).
            # For small x, this lower bound is small.
            
            # Let's try iterating U from max(1, 2*x*(2*x-n)+1) to B.
            # If B is small, this is fast.
            
            B_sq = B * B
            lower_U = max(1, 2 * x * (2 * x - n) + 1)
            
            # We need U to divide B_sq.
            # Instead of checking all U, we can find divisors of B_sq.
            # But finding divisors of B_sq is as hard as iterating up to B.
            
            # Is there a smarter way?
            # (A y - B)(A z - B) = B^2
            # Let g = gcd(A, B).
            # A = g A', B = g B'.
            # (g A' y - g B')(g A' z - g B') = g^2 B'^2
            # (A' y - B')(A' z - B') = B'^2
            # Let U' = A' y - B', V' = A' z - B'.
            # U' V' = B'^2.
            # y = (U' + B') / A'.
            # z = (V' + B') / A'.
            # We need U' V' = B'^2.
            # U' must be a divisor of B'^2.
            # Also U' < V'.
            # And (U' + B') % A' == 0.
            # And (V' + B') % A' == 0.
            
            # B' = B/g.
            # Iterating divisors of B'^2 is still O(B').
            
            # Let's just iterate U from lower_U to B.
            # Check if B_sq % U == 0.
            # If yes, V = B_sq // U.
            # Check if V > U.
            # Check if (U+B) % A == 0.
            # If yes, y = (U+B)//A.
            # Check if y > x.
            # Check if (V+B) % A == 0.
            # If yes, z = (V+B)//A.
            # Check if z > y.
            # Check if x, y, z are distinct (implied by x<y<z).
            # Check if x,y,z <= 10^12.
            
            # This loop is O(B) per x.
            # Average B is n * (n/2) = n^2/2.
            # Number of x is n/2.
            # Work per n is n^3/4.
            # Sum n^3 for n=3..256 is approx 256^4/4 = 4.2e9.
            # In Python, 4e9 simple ops is ~100-200 seconds.
            # This might TLE.
            
            # We need to optimize.
            # Notice that U must be a divisor of B^2.
            # The number of divisors of B^2 is small (usually < 1000).
            # So we should enumerate divisors of B^2 directly.
            
            # How to enumerate divisors of B^2 efficiently?
            # Factorize B.
            # B = n*x.
            # We can precompute prime factors for n and x?
            # Or just factorize B each time.
            # B < 50000. Factorization is fast.
            
            def get_divisors(num):
                divs = set()
                i = 1
                while i * i <= num:
                    if num % i == 0:
                        divs.add(i)
                        divs.add(num // i)
                    i += 1
                return divs
            
            # get_divisors(B_sq) takes O(sqrt(B_sq)) = O(B).
            # Same complexity.
            
            # However, we can skip many U by checking (U+B)%A==0 first?
            # No, we need U to be a divisor.
            
            # Let's try the direct iteration of U but with a tighter bound.
            # U < V => U < B.
            # Also y > x => U > 2x(2x-n).
            
            # Maybe we can iterate y directly?
            # 1/y + 1/z = A/B.
            # z = B y / (A y - B).
            # We need A y - B > 0 => y > B/A = n x / (4x-n).
            # Also z > y => B y / (A y - B) > y => B > A y - B => A y < 2B => y < 2B/A.
            # So y is in (B/A, 2B/A).
            # Range size is B/A.
            # A = 4x - n.
            # For x close to n/4, A is small, range is large.
            # For x close to 3n/4, A is large, range is small.
            
            # Let's iterate y in this range.
            # y_min = B // A + 1
            # y_max = 2 * B // A
            # Range size approx B/A.
            # Sum of B/A over x?
            # B/A = n x / (4x - n).
            # For x = n/4 + k, A = 4k.
            # B/A = n(n/4+k)/(4k) = n^2/16k + n/4.
            # Sum over k=1 to n/2.
            # Sum 1/k is log(n).
            # So total work per n is n^2 log n.
            # Sum n^2 log n for n=3..256.
            # 256^2 * log(256) * 256/3 approx 16000 * 5.5 * 85 = 7.5e6.
            # This is very fast!
            
            # Let's use this approach.
            
            A = 4 * x - n
            B = n * x
            
            if A <= 0:
                continue
                
            y_min = B // A + 1
            y_max = 2 * B // A
            
            # Ensure y > x
            if y_min <= x:
                y_min = x + 1
                
            if y_min > y_max:
                continue
                
            for y in range(y_min, y_max + 1):
                denom = A * y - B
                if denom <= 0:
                    continue
                if (B * y) % denom == 0:
                    z = (B * y) // denom
                    if z > y:
                        # Check distinctness and bounds
                        if x != y and y != z and x != z:
                            if x <= 10**12 and y <= 10**12 and z <= 10**12:
                                found = [x, y, z]
                                break
            if found:
                break
                
        answers[idx] = found
        
    return answers
