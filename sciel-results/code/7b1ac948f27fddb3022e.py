import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    LIMIT = 10**12
    answers = [None] * 254
    
    for n in range(3, 257):
        if time.time() > deadline:
            break
        found = None
        
        # Try x in increasing order first
        x_min = n // 4 + 1
        x_max = 3 * n // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            B2 = B * B
            
            # y must satisfy: y > B/A, y < 2B/A, y > x
            # Y = Ay - B, so Y > 0 and Y < B
            # Actually Y = Ay - B, Z = B²/Y
            # y > x => Ay - B > Ax - B => Y > Ax - B = x(4x-n) - nx = 4x² - 2nx
            # z > y => Z > Y => B²/Y > Y => Y² < B² => Y < B (since Y > 0)
            # Also y < 2B/A => Ay - B < 2B => Y < 3B? No.
            # y < 2B/A => Ay < 2B => Y = Ay - B < B. So Y < B.
            # And Y > max(0, 4x² - 2nx)
            
            Y_low = max(1, 4*x*x - 2*n*x + 1)  # Y > 4x² - 2nx
            Y_high = B - 1  # Y < B
            
            if Y_low > Y_high:
                continue
            
            # Y must be congruent to -B mod A
            r = (-B) % A
            if Y_low % A != r:
                Y_low += (r - Y_low) % A
            
            if Y_low > Y_high:
                continue
            
            # Iterate Y in steps of A
            for Y in range(Y_low, Y_high + 1, A):
                if B2 % Y == 0:
                    Z = B2 // Y
                    if Z >= Y:  # z >= y
                        if (Z + B) % A == 0:
                            z = (Z + B) // A
                            y = (Y + B) // A
                            # Check distinctness and bounds
                            if x != y and y != z and x != z and z <= LIMIT:
                                # Verify: 1/x + 1/y + 1/z == 4/n
                                # Cross multiply: n(yz + xz + xy) == 4xyz
                                lhs = n * (y*z + x*z + x*y)
                                rhs = 4 * x * y * z
                                if lhs == rhs:
                                    found = [x, y, z]
                                    break
            if found:
                break
        
        if not found:
            # Fallback: try x in decreasing order with direct y iteration
            for x in range(x_max, x_min - 1, -1):
                if time.time() > deadline:
                    break
                A = 4 * x - n
                if A <= 0:
                    continue
                B = n * x
                
                y_min = max(x + 1, B // A + 1)
                y_max = (2 * B) // A - 1
                
                if y_min > y_max:
                    continue
                
                # Cap the range to avoid too many iterations
                if y_max - y_min > 100000:
                    continue
                    
                for y in range(y_min, y_max + 1):
                    denom = A * y - B
                    if denom <= 0:
                        continue
                    if (B * y) % denom == 0:
                        z = (B * y) // denom
                        if z > y and z <= LIMIT:
                            # Verify
                            lhs = n * (y*z + x*z + x*y)
                            rhs = 4 * x * y * z
                            if lhs == rhs:
                                found = [x, y, z]
                                break
                if found:
                    break
        
        answers[n - 3] = found

    return answers
