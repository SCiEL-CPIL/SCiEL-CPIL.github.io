import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    LIMIT = 10**12
    answers = [None] * 254
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        # Try x values from x_min to x_max
        # We can limit the number of x's if we are running out of time, 
        # but for now let's try all and optimize the inner loop.
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            
            a = 4 * x - n
            if a <= 0:
                continue
            
            b = n * x
            b2 = b * b
            
            # u = a*y - b
            # y = (u + b) / a
            # z = (b2/u + b) / a
            # Conditions:
            # 1. u divides b2
            # 2. (u + b) % a == 0
            # 3. (b2/u + b) % a == 0
            # 4. y, z distinct from x and each other
            # 5. y, z <= LIMIT
            
            # Lower bound for u to ensure y > x:
            # y > x => (u+b)/a > x => u > ax - b
            # ax - b = x(4x-n) - nx = 4x^2 - 2nx
            u_min = max(1, 4 * x * x - 2 * n * x + 1)
            
            # Upper bound for u:
            # We need u < b to ensure y != z (if u=b, y=z).
            # Also u <= b2 (trivial).
            # If u > b, then v = b2/u < b, so z < y. 
            # We can just check u up to b-1. If we find a solution with u < b, we have y < z.
            # If we only find solutions with u > b, we have y > z.
            # Since the equation is symmetric in y and z, checking u < b is sufficient 
            # to find ALL pairs {y, z} where y != z, provided we don't care about order.
            # Wait, if u < b, then v > b, so y < z.
            # If u > b, then v < b, so y > z.
            # The set of solutions {y, z} is the same.
            # So we can restrict u to [u_min, b-1].
            
            u_max = b - 1
            
            if u_min > u_max:
                continue
                
            # Iterate u
            # Optimization: We can step by 1, but we can also skip if possible? No.
            for u in range(u_min, u_max + 1):
                if b2 % u != 0:
                    continue
                
                if (u + b) % a != 0:
                    continue
                    
                y = (u + b) // a
                
                # Check y bounds and distinctness
                if y <= 0 or y > LIMIT or y == x:
                    continue
                    
                v = b2 // u
                if (v + b) % a != 0:
                    continue
                    
                z = (v + b) // a
                
                if z <= 0 or z > LIMIT or z == x or z == y:
                    continue
                
                # Valid solution found
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers[idx] = found
        
    return answers
