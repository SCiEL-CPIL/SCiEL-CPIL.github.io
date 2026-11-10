import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    LIMIT = 10**12
    answers = [None] * 254
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        # x must satisfy: n/4 < x < 3n/4 (approximately)
        # More precisely: 4x - n > 0 => x > n/4
        # And we need x < y < z, so 1/x + 1/y + 1/z < 3/x => 4/n < 3/x => x < 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4  # x < 3n/4, so x <= floor(3n/4) - 1 if 3n/4 is integer, but let's be safe
        # Actually x < 3n/4 means x <= ceil(3n/4) - 1
        # Let's use x_max = (3*n - 1) // 4 to be safe (x < 3n/4)
        
        for x in range(x_min, (3 * n) // 4 + 1):
            if time.time() > deadline:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            # We need (a*y - b)(a*z - b) = b²
            # Let d = a*y - b, then d divides b², and z = (b²//d + b) // a
            # y = (d + b) // a
            # Need: (d + b) % a == 0, (b²//d + b) % a == 0
            # And d > 0 (since y > x > 0, a*y - b = a*y - n*x, and a = 4x-n, so a*y - n*x = (4x-n)y - nx = 4xy - ny - nx. 
            # For y > x: 4xy - ny - nx > 4x² - nx - nx = 4x² - 2nx = 2x(2x-n). 
            # Since x > n/4, 2x > n/2, so 2x - n > -n/2. Hmm, this isn't necessarily positive.
            # Let's just require d > 0 and check.
            
            # Find all divisors of b²
            b2 = b * b
            # We need to iterate over divisors d of b2 such that d <= b2//d (to avoid duplicates, but we need both y and z)
            # Actually, we can iterate d from 1 to sqrt(b2), and for each divisor d, check both d and b2//d
            
            # But b2 can be huge (n up to 256, x up to ~192, so b = n*x up to ~49152, b2 up to ~2.4e9)
            # sqrt(b2) up to ~49152, which is manageable per x, but we have many x values.
            
            # Alternative: iterate d over divisors of b2. But factoring b2 for each x might be slow.
            
            # Let's try: for d in range(1, int(b2**0.5)+1): if b2 % d == 0: check d and b2//d
            # But b2 can be up to (256*192)^2 ≈ 2.4e9, sqrt ≈ 49152. For each x, we do up to 49152 iterations.
            # Number of x values per n: about 3n/4 - n/4 = n/2 ≈ 128 for n=256.
            # Total: 254 * 128 * 49152 ≈ 1.6e9 operations. Too slow!
            
            # Need a better approach. Let's think differently.
            
            # From (a*y - b)(a*z - b) = b², let u = a*y - b, v = a*z - b, uv = b².
            # y = (u + b)/a, z = (v + b)/a = (b²/u + b)/a
            # Need u | b², (u+b) % a == 0, (b²/u + b) % a == 0
            # And y > x, z > y (or just distinct)
            
            # u must be a divisor of b². The number of divisors of b² depends on the factorization of b.
            # b = n*x. For small n and x, this is manageable.
            
            # Let's factor b and enumerate divisors of b².
            
            # Actually, let's just try a limited range of u. Since y > x, u = a*y - b > a*x - b = a*x - n*x = x(a-n) = x(4x-n-n) = x(4x-2n) = 2x(2x-n)
            # Since x > n/4, 2x > n/2, so 2x - n > -n/2. If x > n/2, then 2x - n > 0, so u > 0.
            # If x <= n/2, then 2x - n <= 0, so u could be negative or positive.
            
            # Let's just require u > 0 and v > 0 (so y, z > 0), and check distinctness.
            
            # For efficiency, let's limit the number of x values we try per n, and use a smarter divisor search.
            
            # Actually, let's try: for each x, compute a, b, and then iterate u over divisors of b² that are <= some bound.
            # Since y = (u+b)/a and we want y not too large (y <= 10^12), u <= a*10^12 - b.
            # But that's still huge.
            
            # Let's just try the first few x values and hope we find a solution quickly.
            # The current program only tries x = n//4+1 and it gets score 1.0. Let's try more x values.
            
            # Let's limit: for each n, try at most 50 x values (from x_min upward), and for each x, 
            # iterate u over divisors of b² up to some limit.
            
            # Actually, let's just try: for d in a range of possible u values.
            # u must divide b². Let's factor b first.
            
            pass  # This approach is getting complex. Let me try a simpler improvement first.
        
        # Fallback: try the original approach with more x values
        if found is None:
            for x in range(x_min, min(x_min + 20, (3 * n) // 4 + 1)):
                if time.time() > deadline:
                    break
                a = 4 * x - n
                if a <= 0:
                    continue
                b = n * x
                # (a*y - b)(a*z - b) = b²
                # Let u = a*y - b. u must divide b².
                # y = (u + b) // a, z = (b²//u + b) // a
                # Need u | b², (u+b) % a == 0, (b²//u + b) % a == 0
                # And x, y, z distinct, all positive, all <= LIMIT
                
                # Iterate u over divisors of b²
                b2 = b * b
                # Find divisors of b2
                # For efficiency, only check u up to sqrt(b2), and also b2//u
                sqrt_b2 = int(b2 ** 0.5)
                for u in range(1, sqrt_b2 + 1):
                    if b2 % u != 0:
                        continue
                    # Check u
                    if (u + b) % a == 0:
                        y = (u + b) // a
                        v = b2 // u
                        if (v + b) % a == 0:
                            z = (v + b) // a
                            if y > 0 and z > 0 and y <= LIMIT and z <= LIMIT:
                                # Check distinctness
                                if x != y and x != z and y != z:
                                    found = [x, y, z]
                                    break
                    # Check b2//u (if different from u)
                    v = b2 // u
                    if v != u and (v + b) % a == 0:
                        y = (v + b) // a
                        u2 = u  # this is the other factor
                        if (u2 + b) % a == 0:
                            z = (u2 + b) // a
                            if y > 0 and z > 0 and y <= LIMIT and z <= LIMIT:
                                if x != y and x != z and y != z:
                                    found = [x, y, z]
                                    break
                if found:
                    break
        
        answers[idx] = found
    
    return answers
