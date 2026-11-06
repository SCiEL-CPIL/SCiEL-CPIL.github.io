def construct():
    answers = []
    
    for n in range(3, 257):
        found = None
        
        # Strategy 1: Try small x values using the divisor method
        # 1/x + 1/y + 1/z = 4/n
        # 1/y + 1/z = (4x-n)/(nx) = A/B
        # (Ay-B)(Az-B) = B^2
        # Let U = Ay-B, V = Az-B. UV = B^2.
        # y = (U+B)/A, z = (V+B)/A.
        
        # We iterate x. x must be > n/4.
        # Start x from floor(n/4) + 1
        start_x = (n // 4) + 1
        
        # We will try x up to a limit. 
        # If no solution found in small x, we might need larger x, 
        # but usually small x works.
        # Let's try x up to start_x + 1000 or 10^5, whichever is smaller.
        # Actually, let's just try a few hundred x values first.
        
        max_x_tries = 1000
        for x in range(start_x, start_x + max_x_tries):
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # Find divisors of B
            # We need U such that U divides B^2.
            # It is sufficient to check U = d^2 where d divides B.
            # Why? If U V = B^2, and U, V are integers, does U have to be a square?
            # No. U and V just multiply to B^2.
            # However, if we iterate all divisors of B^2, it's too many.
            # But note: y = (U+B)/A. 
            # If we pick U as any divisor of B^2, we get a candidate.
            # To reduce work, we can iterate d dividing B, and set U = d * (B/d) * something? No.
            
            # Let's just iterate d dividing B.
            # If we set U = d * k and V = (B^2) / (d*k), it's complex.
            
            # Better: Iterate d dividing B.
            # Let U = d * e. Then V = B^2 / (de).
            # This is still many.
            
            # Wait, if we set U = d^2 and V = (B/d)^2, we cover a subset.
            # Are all solutions covered by U being a square of a divisor?
            # No. Example: B=12. B^2=144.
            # Divisors of 12: 1, 2, 3, 4, 6, 12.
            # Squares: 1, 4, 9, 16, 36, 144.
            # 9 is not a divisor of 12, but 9 divides 144.
            # So iterating squares of divisors of B misses divisors of B^2 that are not squares of divisors of B?
            # Actually, any divisor of B^2 is of the form $\prod p_i^{2e_i}$? No.
            # If B = $\prod p_i^{a_i}$, $B^2 = \prod p_i^{2a_i}$.
            # A divisor U is $\prod p_i^{b_i}$ with $0 \le b_i \le 2a_i$.
            # If we only check $b_i$ even, we miss odd exponents.
            
            # So we must check all divisors of B^2?
            # Number of divisors of B^2 can be large.
            
            # Alternative: Iterate y?
            # The previous code iterated y in a window.
            # Let's stick to iterating x, but for each x, iterate y?
            # 1/z = A/B - 1/y = (Ay-B)/(By).
            # z = By / (Ay-B).
            # Let D = Ay - B. D must divide By.
            # D = Ay - B.
            # y = (D+B)/A.
            # For y to be integer, D+B must be divisible by A.
            # Also D must divide B(D+B)/A?
            # z = B(D+B)/(AD).
            # For z to be integer, AD must divide B(D+B).
            # Since gcd(D, D+B) = gcd(D, B), let g = gcd(D, B).
            # This is getting complicated.
            
            # Let's go back to iterating y in a reasonable range.
            # The previous code did this. Why did it score 1.0161?
            # Maybe it missed some n where the solution y is large.
            
            # Let's try a hybrid:
            # 1. Try the specific algebraic identity (fast).
            # 2. Try iterating x, and for each x, iterate y in a wider range or use factorization of B^2 if B is small.
            
            if B < 10**6:
                # Factor B^2 by finding divisors of B
                # Then divisors of B^2 are d1*d2 where d1|B, d2|B? No.
                # Just find all divisors of B. Let's call them D_B.
                # Then divisors of B^2 can be generated?
                # Simpler: Iterate y from ceil(B/A) + 1 to 2B/A.
                # Range size is B/A = nx/(4x-n).
                # For small x, this is large.
                
                # Let's use the property:
                # z = By / (Ay - B).
                # Let k = Ay - B. Then y = (k+B)/A.
                # z = B(k+B)/(Ak).
                # k must divide B(k+B)/A.
                # k | B^2/A ? No.
                
                # Let's just iterate y.
                min_y = B // A + 1
                if min_y <= x:
                    min_y = x + 1
                # Upper bound for y: z > y implies y < 2B/A.
                max_y = min(2 * B // A + 1, min_y + 1000) # Limit window
                
                for y in range(min_y, max_y):
                    denom = A * y - B
                    if denom <= 0:
                        continue
                    if (B * y) % denom != 0:
                        continue
                    z = (B * y) // denom
                    if z <= y:
                        continue
                    if x == y or y == z or x == z:
                        continue
                    if x > 10**12 or y > 10**12 or z > 10**12:
                        continue
                    lhs = n * (y * z + x * z + x * y)
                    rhs = 4 * x * y * z
                    if lhs == rhs:
                        found = [x, y, z]
                        break
                if found is not None:
                    break
            else:
                # If B is large, the window 2B/A is large.
                # But for large x, A is large, so B/A is small.
                # B/A = nx/(4x-n) approx n/4.
                # So the window size is approx n/4.
                # For n=256, window is 64.
                # So iterating y in a window of 1000 is safe.
                min_y = B // A + 1
                if min_y <= x:
                    min_y = x + 1
                max_y = min(2 * B // A + 1, min_y + 1000)
                
                for y in range(min_y, max_y):
                    denom = A * y - B
                    if denom <= 0:
                        continue
                    if (B * y) % denom != 0:
                        continue
                    z = (B * y) // denom
                    if z <= y:
                        continue
                    if x == y or y == z or x == z:
                        continue
                    if x > 10**12 or y > 10**12 or z > 10**12:
                        continue
                    lhs = n * (y * z + x * z + x * y)
                    rhs = 4 * x * y * z
                    if lhs == rhs:
                        found = [x, y, z]
                        break
                if found is not None:
                    break
        
        answers.append(found)
        
    return answers
