import time

def construct():
    # Deadline guard with safety margin to ensure return before 60s
    start_time = time.time()
    deadline = start_time + 55  # 55s to be safe against 60s limit
    
    answers = []
    MAX_Z = 10**12
    n_total = 254  # n from 3 to 256 inclusive
    
    for idx, n in enumerate(range(3, 257)):
        # Check global deadline
        if time.time() > deadline:
            # If we run out of time, append None for the rest
            # But we need to fill the list to the correct length
            # Since we are in a loop, we can just break and pad later
            # Or append None and continue? 
            # Better to break and pad at the end to avoid overhead
            break
            
        # Adaptive time budgeting
        elapsed = time.time() - start_time
        remaining = deadline - start_time - elapsed
        remaining_n = n_total - idx
        
        if remaining <= 0:
            break
            
        # Allocate time for this n
        # We want to finish all n's if possible.
        # Average time per n remaining.
        time_per_n = remaining / remaining_n
        
        # Cap the time per n to prevent a single hard n from eating too much,
        # but allow a bit more if there are few remaining.
        # Actually, just using the average is a good heuristic.
        # If an n is hard, it will use its budget and fail, moving to the next.
        # If an n is easy, it finishes fast, leaving more time for the rest.
        
        n_deadline = time.time() + time_per_n
        
        found = None
        
        # x must satisfy: x > n/4 and x <= 3n/4
        # 1/x < 4/n - 1/y - 1/z < 4/n
        # x > n/4
        # Also 1/x > 4/n - 2/1 = 4/n - 2 (since y,z >= 1, but distinct positive ints)
        # Actually, the lower bound for x comes from 1/x < 4/n (if y,z -> inf)
        # And 1/x > 4/n - 1/1 - 1/1 = 4/n - 2.
        # For n >= 3, 4/n <= 1.33, so 4/n - 2 < 0. So x > n/4 is the main constraint.
        # Upper bound: 1/x > 4/n - 1/y - 1/z.
        # Since y > x and z > y, 1/y < 1/x and 1/z < 1/x.
        # 1/x > 4/n - 2/x - 2/x => 4/x > 4/n => x < n.
        # Wait, the standard bound is x <= 3n/4.
        # 1/x + 1/y + 1/z = 4/n
        # 3/x < 4/n => x > 3n/4 is false.
        # 1/x < 4/n - 1/(x+1) - 1/(x+2) ... this is getting complicated.
        # Standard bound: x is in (n/4, 3n/4].
        # Let's verify x_max.
        # 1/x + 1/y + 1/z = 4/n
        # 1/y + 1/z = 4/n - 1/x = (4x-n)/nx
        # Since y > x and z > y, 1/y < 1/x and 1/z < 1/x.
        # So (4x-n)/nx < 2/x
        # (4x-n)/n < 2
        # 4x - n < 2n
        # 4x < 3n
        # x < 3n/4.
        # So x_max = (3n - 1) // 4 or similar.
        # Let's use x_max = (3 * n) // 4.
        
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        # Try x values. 
        # Smaller x leads to larger 1/x, leaving less for 1/y+1/z, 
        # which might make y,z larger? 
        # Actually, smaller x means A = 4x-n is smaller.
        # y_min = B/A + 1. B = nx. A = 4x-n.
        # For small x, A is small, B is small. 
        # Let's just iterate x from x_min to x_max.
        
        for x in range(x_min, x_max + 1):
            if time.time() > n_deadline:
                break
                
            denom_x = 4 * x - n
            if denom_x <= 0:
                continue
            
            # B = n * x
            # A = denom_x
            B = n * x
            A = denom_x
            
            # We need 1/y + 1/z = A/B
            # y > B/A
            # z > y
            # 1/z = A/B - 1/y = (Ay - B) / By
            # z = By / (Ay - B)
            # Condition z > y:
            # By / (Ay - B) > y
            # B / (Ay - B) > 1  (since y > 0)
            # B > Ay - B
            # 2B > Ay
            # y < 2B / A
            
            # Wait, Inspiration 1 said y < (n+B)/A is tighter.
            # Let's re-verify.
            # z = nxy / (4xy - ny - nx)
            # z > y => nxy > y(4xy - ny - nx) => nx > 4xy - ny - nx => 2nx + ny > 4xy => y(4x - n) < 2nx => y < 2nx / (4x - n).
            # This matches y < 2B/A.
            
            # Where did (n+B)/A come from?
            # z = ny / (Ay - B). This formula assumes 1/z = (Ay-B)/ny.
            # 1/z = 4/n - 1/x - 1/y = (4xy - ny - nx) / nxy.
            # My previous A/B was (4x-n)/nx.
            # 1/y + 1/z = (4x-n)/nx.
            # 1/z = (4x-n)/nx - 1/y = ((4x-n)y - nx) / nxy = (4xy - ny - nx) / nxy.
            # So z = nxy / (4xy - ny - nx).
            # The formula z = ny / (Ay - B) is WRONG unless x=1?
            # If x=1, A=4-n, B=n.
            # 1/z = (4-n)/n - 1/y = (4y - ny - n) / ny.
            # z = ny / (4y - ny - n).
            # The general formula is z = nxy / (4xy - ny - nx).
            
            # So the condition z > y is:
            # nxy / (4xy - ny - nx) > y
            # nx / (4xy - ny - nx) > 1
            # nx > 4xy - ny - nx
            # 2nx + ny > 4xy
            # y(4x - n) < 2nx
            # y < 2nx / (4x - n)
            
            # So y_max = (2 * B) // A is the correct tight bound for z > y.
            # Why did Inspiration 1 think (n+B)/A was tighter?
            # They derived y < (n+B)/A from z = ny/(Ay-B).
            # But that formula for z is incorrect for general x.
            # Let's check if (n+B)/A is ever tighter than 2B/A.
            # (n+B)/A < 2B/A  <=> n+B < 2B <=> n < B = nx <=> 1 < x.
            # So for x > 1, (n+B)/A IS smaller than 2B/A.
            # But is (n+B)/A a valid bound?
            # If y < (n+B)/A, does it guarantee z > y?
            # We need y < 2B/A for z > y.
            # If (n+B)/A < 2B/A, then y < (n+B)/A implies y < 2B/A.
            # So y < (n+B)/A is a SUBSET of the valid range.
            # It is a tighter bound, but does it exclude valid solutions?
            # Valid solutions require y < 2B/A.
            # If we restrict to y < (n+B)/A, we might miss solutions where (n+B)/A <= y < 2B/A.
            # Let's check if such solutions exist.
            # We need z integer.
            # z = nxy / (4xy - ny - nx).
            # Let's test n=3.
            # x=2. A=5, B=6.
            # y < 2*6/5 = 2.4. So y can be 1, 2.
            # But y > x=2. So no y.
            
            # Let's test n=4.
            # x=2. A=4, B=8.
            # y < 16/4 = 4.
            # y > 2. So y=3.
            # (n+B)/A = (4+8)/4 = 3.
            # So y < 3. y=2? No y>x.
            # So no solution for x=2?
            # 1/2 + 1/3 + 1/z = 1 => 5/6 + 1/z = 1 => 1/z = 1/6 => z=6.
            # x=2, y=3, z=6.
            # Check bound: y=3.
            # y < 2B/A = 4. 3 < 4. OK.
            # y < (n+B)/A = 3. 3 < 3 is FALSE.
            # So using y_max = (n+B)//A would EXCLUDE y=3.
            # So Inspiration 1's tighter bound is WRONG for general x.
            # It only holds if the formula z = ny/(Ay-B) is correct, which it isn't.
            
            # So I should stick with y_max = (2 * B) // A.
            
            y_min = B // A + 1
            y_max = (2 * B) // A
            
            # Ensure y > x for distinctness (assuming x < y < z)
            if y_min <= x:
                y_min = x + 1
                
            if y_max < y_min:
                continue
            
            # Iterate y
            for y in range(y_min, y_max + 1):
                # z = nxy / (4xy - ny - nx)
                # Denominator D = 4xy - ny - nx
                # Numerator N = nxy
                D = 4 * x * y - n * y - n * x
                if D <= 0:
                    continue
                
                N = n * x * y
                if N % D != 0:
                    continue
                    
                z = N // D
                
                # Check distinctness and bounds
                if z <= y:
                    continue
                if z > MAX_Z:
                    continue
                
                # x < y < z ensures distinctness
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers.append(found)
        
        # If we broke out of the x loop early due to time, 
        # found might be None. That's fine.
        
    # Pad with None if we didn't reach the end
    while len(answers) < n_total:
        answers.append(None)
        
    return answers
