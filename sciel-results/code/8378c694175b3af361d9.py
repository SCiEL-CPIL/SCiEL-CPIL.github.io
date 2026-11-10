import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80
    
    answers = []
    MAX_Z = 10**12
    
    # Pre-calculate or just loop
    for n in range(3, 257):
        # Check time periodically, not in inner loop
        if time.time() > deadline:
            answers.append(None)
            continue
            
        found = None
        
        # x must satisfy: x > n/4 and x <= 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            denom_x = 4 * x - n
            if denom_x <= 0:
                continue
            
            # B = n * x
            B = n * x
            A = denom_x
            
            # y > B/A and y <= 2B/A
            y_min = B // A + 1
            
            # Tighter bound for y derived from z > y:
            # z = B*y / (A*y - B) > y
            # B*y > A*y^2 - B*y
            # A*y^2 - 2B*y < 0
            # y(A*y - 2B) < 0
            # Since y>0, A*y - 2B < 0 => y < 2B/A.
            # Wait, my previous derivation:
            # z > y <=> n/(Ay-B) > 1 <=> n > Ay - B <=> Ay < n + B <=> y < (n+B)/A.
            # Let's re-verify.
            # 1/x + 1/y + 1/z = 4/n
            # 1/z = 4/n - 1/x - 1/y = (4xy - ny - nx) / nxy
            # z = nxy / (4xy - ny - nx)
            # We want z > y.
            # nxy / (4xy - ny - nx) > y
            # nxy > y(4xy - ny - nx)
            # nx > 4xy - ny - nx  (divide by y>0)
            # 2nx + ny > 4xy
            # y(4x - n) < 2nx
            # y < 2nx / (4x - n) = 2B/A.
            # So the bound y < 2B/A is the correct one for z > y.
            # My previous derivation "Ay < n+B" was wrong?
            # Let's re-evaluate "z > y" with z = ny / (Ay - B).
            # Note: 1/z = (Ay - B) / ny.
            # So z = ny / (Ay - B).
            # z > y <=> ny / (Ay - B) > y <=> n / (Ay - B) > 1 <=> n > Ay - B <=> Ay < n + B.
            # Why the discrepancy?
            # Ah, B = nx. A = 4x - n.
            # Ay - B = (4x-n)y - nx = 4xy - ny - nx.
            # So z = nxy / (4xy - ny - nx).
            # The formula z = ny / (Ay - B) is correct.
            # Condition z > y:
            # ny / (Ay - B) > y
            # n / (Ay - B) > 1
            # n > Ay - B
            # Ay < n + B
            # y < (n + B) / A.
            
            # Now check the other derivation:
            # z = nxy / (4xy - ny - nx).
            # z > y => nxy > y(4xy - ny - nx) => nx > 4xy - ny - nx => 2nx + ny > 4xy => y(4x - n) < 2nx => y < 2nx / (4x - n).
            
            # Are (n+B)/A and 2B/A the same?
            # (n + nx) / (4x - n) vs 2nx / (4x - n).
            # n(1+x) vs 2nx.
            # 1+x vs 2x => 1 vs x.
            # Since x >= 1, 2x >= x+1.
            # So 2nx >= n(x+1).
            # So 2B/A >= (n+B)/A.
            # The bound y < (n+B)/A is TIGHTER (smaller) than y < 2B/A.
            # So we should use y_max = (n + B) // A.
            
            y_max = (n + B) // A
            
            # Ensure y > x for distinctness (assuming x < y < z)
            if y_min <= x:
                y_min = x + 1
                
            if y_max < y_min:
                continue
            
            # Iterate y
            for y in range(y_min, y_max + 1):
                # z = B*y / (A*y - B)
                D = A * y - B
                if D <= 0:
                    continue
                
                if (B * y) % D != 0:
                    continue
                    
                z = (B * y) // D
                
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
        
    return answers
