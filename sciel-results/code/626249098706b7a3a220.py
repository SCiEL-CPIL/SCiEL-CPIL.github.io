import math

def construct():
    answers = []
    
    for n in range(3, 257):
        found = None
        start_x = n // 4 + 1
        # We search for x. 
        # For 4/n, it is known that solutions exist.
        # We will search x up to 10^7 to be safe.
        for x in range(start_x, 10**7):
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            B2 = B * B
            
            # m must satisfy:
            # U = m*A - B > 0  => m > B/A
            # V = B^2/U >= U   => U <= B => m*A - B <= B => m <= 2B/A
            m_min = B // A + 1
            m_max = (2 * B) // A
            
            for m in range(m_min, m_max + 1):
                U = m * A - B
                if U <= 0:
                    continue
                if B2 % U != 0:
                    continue
                V = B2 // U
                # Check if y, z are integers
                if (U + B) % A != 0:
                    continue
                if (V + B) % A != 0:
                    continue
                
                y = (U + B) // A
                z = (V + B) // A
                
                # Check distinctness
                if x == y or x == z or y == z:
                    continue
                
                # Check bounds
                if y > 10**12 or z > 10**12:
                    continue
                
                # Final verification
                if n * (y * z + x * z + x * y) == 4 * x * y * z:
                    found = [x, y, z]
                    break
            if found:
                break
        answers.append(found)
    return answers
