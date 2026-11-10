"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    LIMIT = 10**12
    answers = [None] * 254
    
    for n in range(3, 257):
        if time.time() > deadline:
            break
        found = None
        # x range: n/4 < x <= 3n/4
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
            
            # y_min, y_max logic
            y_min = max(x + 1, B // A + 1)
            
            # y < 2B/A
            if (2 * B) % A == 0:
                y_max = (2 * B) // A - 1
            else:
                y_max = (2 * B) // A
            
            if y_min > y_max:
                continue
            
            # Iterate Y = A*y - B
            # Y ranges from A*y_min - B to A*y_max - B
            # Step is A
            # We need B2 % Y == 0 and (B2//Y + B) % A == 0
            # Also y > x is already handled by y_min
            
            Y_start = A * y_min - B
            Y_end = A * y_max - B
            
            # Ensure Y_start is positive (should be, given y_min > B/A)
            if Y_start <= 0:
                # Adjust to first positive multiple congruent to -B mod A
                # Y = k*A - B. k > B/A.
                k_min = B // A + 1
                Y_start = k_min * A - B
                # Also need y > x => Y + B > A*x => Y > A*x - B
                # A*x - B = (4x-n)x - nx = 4x^2 - nx - nx = 4x^2 - 2nx
                Y_start = max(Y_start, 4*x*x - 2*n*x + 1)
                
                # Align Y_start to congruence class -B mod A
                # Y % A == -B % A
                r = (-B) % A
                if Y_start % A != r:
                    diff = (r - Y_start) % A
                    Y_start += diff
            
            if Y_start > Y_end:
                continue
            
            # Iterate Y in steps of A
            for Y in range(Y_start, Y_end + 1, A):
                if time.time() > deadline:
                    break
                if B2 % Y == 0:
                    Z = B2 // Y
                    # z = (Z + B) / A
                    if (Z + B) % A == 0:
                        z = (Z + B) // A
                        y = (Y + B) // A
                        # Check distinctness and bounds
                        if y > x and z > y and z <= LIMIT:
                            # x, y, z are distinct
                            found = [x, y, z]
                            break
            if found:
                break
        
        answers[n - 3] = found

    return answers
# EVOLVE-BLOCK-END
