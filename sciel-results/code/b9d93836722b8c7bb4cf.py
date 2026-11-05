import math

def construct():
    answers = []
    for n in range(3, 257):
        found = None
        # x must be > n/4
        start_x = n // 4 + 1
        # We don't need to search too far, usually small x works
        for x in range(start_x, start_x + 50):
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            M = B * B
            
            # Find divisor u of M such that u % A == (-B) % A
            # And v = M/u such that v % A == (-B) % A
            target_rem = (-B) % A
            
            # Iterate through divisors of M
            # M can be up to (256*300)^2 ~ 5.8e10, sqrt ~ 2.4e5
            # This is fast enough for 254 iterations
            
            u = 1
            while u * u <= M:
                if M % u == 0:
                    v = M // u
                    if u % A == target_rem and v % A == target_rem:
                        y = (u + B) // A
                        z = (v + B) // A
                        # Check distinctness and bounds
                        if x != y and x != z and y != z:
                            if x <= 1000000000000 and y <= 1000000000000 and z <= 1000000000000:
                                found = [x, y, z]
                                break
                u += 1
            
            if found:
                break
        
        answers.append(found)
    return answers
