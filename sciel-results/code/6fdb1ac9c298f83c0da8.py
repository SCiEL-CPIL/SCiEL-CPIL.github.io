import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        M = 4 * n * n * n
        
        # Get all divisors of M
        divs = []
        sqrt_m = int(math.isqrt(M))
        for i in range(1, sqrt_m + 1):
            if M % i == 0:
                divs.append(i)
                if i != M // i:
                    divs.append(M // i)
        
        # Sort divisors to allow early termination in inner loop if needed, 
        # though we just iterate all.
        divs.sort()
        
        # Iterate through pairs (X, Y) such that X*Y*Z = M
        # We can just iterate X from divs, and for each X, iterate Y from divs
        # such that Y <= Z (i.e., Y <= sqrt(Rem))
        
        for X in divs:
            if (X + n) % 4 != 0:
                continue
            
            x = (X + n) // 4
            if x <= 0:
                continue
            
            Rem = M // X
            
            # We need Y such that Y divides Rem and Z = Rem/Y divides M (which it does if Y divides Rem)
            # And Y, Z satisfy modulo constraints.
            # We can iterate Y from divs.
            # To avoid duplicates and ensure Y <= Z, we can check Y <= sqrt(Rem)
            # But simply iterating all Y in divs and checking Y <= Z is fine, 
            # or just checking Y*Y <= Rem.
            
            for Y in divs:
                if Y * Y > Rem:
                    break
                
                if Rem % Y != 0:
                    continue
                
                Z = Rem // Y
                
                if (Y + n) % 4 != 0:
                    continue
                
                if (Z + n) % 4 != 0:
                    continue
                
                y = (Y + n) // 4
                z = (Z + n) // 4
                
                if y <= 0 or z <= 0:
                    continue
                
                # Check distinctness
                if x == y or x == z or y == z:
                    continue
                
                # Check bounds (though x,y,z are derived from divisors of 4n^3, 
                # they are bounded by M, which is << 10^12)
                if x > 10**12 or y > 10**12 or z > 10**12:
                    continue
                
                # Verify just in case
                # 1/x + 1/y + 1/z == 4/n
                # n(yz + xz + xy) == 4xyz
                if n * (y * z + x * z + x * y) == 4 * x * y * z:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
