import time
import math

def _find_solution(n, deadline):
    # x must be > n/4
    start_x = n // 4 + 1
    # x must be < 3n/4 for y,z to be >= x (assuming x is smallest)
    # Actually if x is not the smallest, we can still find a solution, 
    # but assuming x is smallest reduces search space.
    # Let's search x up to 2*n to be safe, but usually small x works.
    end_x = 2 * n + 10
    
    for x in range(start_x, end_x):
        if time.time() > deadline:
            return None
            
        A = 4 * x - n
        if A <= 0:
            continue
        B = n * x
        
        # We need (By - A)(Bz - A) = A^2
        # Let U = By - A. U must be a divisor of A^2.
        # U = B*y - A => y = (U + A) / B
        # V = A^2 / U => z = (V + A) / B
        
        # Iterate through divisors of A^2
        # To avoid duplicates and ensure y <= z, we can restrict U <= A (since U <= V)
        # But we just need any distinct solution.
        
        a2 = A * A
        # Get divisors of a2
        divisors = []
        for i in range(1, int(math.isqrt(a2)) + 1):
            if a2 % i == 0:
                divisors.append(i)
                if i != a2 // i:
                    divisors.append(a2 // i)
        
        for U in divisors:
            if time.time() > deadline:
                return None
                
            # Check if U + A is divisible by B
            if (U + A) % B != 0:
                continue
            y = (U + A) // B
            
            # Check if corresponding V + A is divisible by B
            V = a2 // U
            if (V + A) % B != 0:
                continue
            z = (V + A) // B
            
            # Check distinctness and bounds
            if x == y or x == z or y == z:
                continue
            if y > 10**12 or z > 10**12:
                continue
            if y < 1 or z < 1:
                continue
                
            # Verify the equation just in case
            if x * y * z == 4 * n * (y * z + x * z + x * y):
                return [x, y, z]
                
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # 45s deadline to be safe within 60s
    
    answers = []
    for n in range(3, 257):
        sol = _find_solution(n, deadline)
        answers.append(sol)
        
    return answers
