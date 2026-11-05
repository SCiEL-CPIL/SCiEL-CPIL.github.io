import time
import math
from math import gcd

def construct():
    """
    Find three DISTINCT positive integer denominators x, y, z <= 10^12
    such that 1/x + 1/y + 1/z = 4/n for n in range(3, 257).
    Returns a list of 254 entries, each None or [x, y, z].
    """
    deadline = time.time() + 45.0  # Safety margin before 60s limit
    answers = [None] * 254
    LIMIT = 10**12

    # For each n, try to find a solution
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        
        # We need 1/x + 1/y + 1/z = 4/n
        # Assume x <= y <= z. Then 3/x >= 4/n => x <= 3n/4
        # Also 1/x < 4/n => x > n/4
        # So n/4 < x <= 3n/4
        
        # To ensure distinctness, we'll enforce x < y < z.
        # 1/x + 1/y + 1/z = 4/n
        # 1/y + 1/z = 4/n - 1/x = (4x - n) / (nx)
        # Let A = 4x - n, B = nx
        # 1/y + 1/z = A/B
        # (y+z)/yz = A/B => Ayz = B(y+z) => Ayz - By - Bz = 0
        # Multiply by A: A^2 yz - AB y - AB z = 0
        # (Ay - B)(Az - B) = B^2
        # Let u = Ay - B, v = Az - B
        # Then uv = B^2
        # y = (u+B)/A, z = (v+B)/A
        # We need y, z to be integers, so (u+B) % A == 0 and (v+B) % A == 0
        # Also y > x and z > y
        
        found = False
        # x range: n//4 + 1 to 3*n//4 (approximately)
        # Actually, since 1/x < 4/n is required for y,z to be larger than x?
        # If x = n/4, then 1/x = 4/n, so 1/y+1/z=0, impossible.
        # So x > n/4.
        # Max x: if x is too large, 1/x is small, 1/y+1/z must be close to 4/n.
        # Since y >= x+1, 1/y <= 1/(x+1).
        # 2/x + 1/x ... roughly 3/x >= 4/n => x <= 3n/4.
        
        x_start = n // 4 + 1
        x_end = (3 * n) // 4 + 2  # A bit of buffer
        
        # Iterate x
        for x in range(x_start, x_end + 1):
            if time.time() > deadline:
                break
            
            A = 4 * x - n
            if A <= 0:
                continue
            
            B = n * x
            B_sq = B * B
            
            # We need divisors u of B_sq such that:
            # 1. (u + B) % A == 0
            # 2. y = (u + B) // A
            # 3. y > x
            # 4. v = B_sq // u
            # 5. (v + B) % A == 0
            # 6. z = (v + B) // A
            # 7. z > y
            # 8. z <= LIMIT
            
            # Instead of iterating all divisors of B_sq (which can be huge),
            # we can iterate u up to sqrt(B_sq) = B.
            # But B can be large.
            # Note that u and v are symmetric. We can assume u <= v => u <= B.
            # Also y < z => u < v => u < B.
            # So we only need to check u < B.
            
            # Optimization:
            # y = (u+B)/A. Since y > x, (u+B)/A > x => u+B > Ax => u > Ax - B = A(x) - B.
            # But A = 4x-n, so Ax - B = x(4x-n) - nx = 4x^2 - nx - nx = 4x^2 - 2nx.
            # This lower bound might not be very helpful if it's negative.
            
            # Let's just iterate u from 1 to B? No, B can be ~ n^2/4. For n=256, B~16000.
            # Iterating to 16000 is fine. For n=3, B~3.
            # Wait, B = n*x. x is ~ n. So B ~ n^2.
            # For n=256, x~192, B ~ 49000.
            # Iterating u from 1 to B is 50k iterations.
            # 254 values of n. 254 * 50k = 12.7 million iterations.
            # In Python, this might be slow if done naively.
            # We need to be smarter.
            
            # We require u | B^2.
            # And (u+B) % A == 0 => u ≡ -B (mod A).
            # Let u = k*A - B.
            # We need u > 0 => k*A > B => k > B/A.
            # Also u < B (for y < z) => k*A - B < B => k*A < 2B => k < 2B/A.
            # So k is in range (B/A, 2B/A).
            # The number of values for k is roughly B/A.
            # A = 4x - n. For x ~ n, A ~ 3n. B ~ n^2.
            # So B/A ~ n/3.
            # For n=256, B/A ~ 85.
            # This is much smaller!
            
            # So we iterate k.
            # k_min = floor(B/A) + 1
            # k_max = ceil(2*B/A) - 1  (strictly less than 2B/A for u < B? If u=B, y=z, not distinct)
            # Actually if u=B, y=(2B)/A. z=(2B)/A. y=z. We need y<z so u<v => u<B.
            # So u < B.
            
            k_min = B // A + 1
            # u = k*A - B < B => k*A < 2B => k < 2B/A
            k_max = (2 * B - 1) // A  # Largest integer k such that k*A < 2B
            
            for k in range(k_min, k_max + 1):
                u = k * A - B
                # Check if u divides B^2
                if B_sq % u != 0:
                    continue
                
                v = B_sq // u
                
                # Check distinctness and order
                # y = (u+B)/A
                # z = (v+B)/A
                # We need (u+B) divisible by A (by construction u = kA - B, so u+B = kA, yes)
                # We need (v+B) divisible by A
                if (v + B) % A != 0:
                    continue
                
                y = (u + B) // A
                z = (v + B) // A
                
                # Check conditions
                if y > x and z > y and z <= LIMIT:
                    # Verify solution just in case
                    if (z * y * n == 4 * y * z - 4 * x * z - 4 * x * y): 
                        # 1/x + 1/y + 1/z = 4/n
                        # yz + xz + xy = 4xyz/n
                        # n(yz + xz + xy) = 4xyz
                        # Let's check n(yz+xz+xy) == 4xyz
                        if n * (y * z + x * z + x * y) == 4 * x * y * z:
                            answers[idx] = [x, y, z]
                            found = True
                            break
            
            if found:
                break
        
        # If not found with x < y < z, we might have missed some due to ordering?
        # The derivation assumed x is the smallest denominator.
        # If a solution exists, one of the denominators is the smallest.
        # By symmetry, we can assume x is the smallest.
        # So this search covers all cases where x is the smallest distinct denominator.
        
        # If still not found, try a broader search?
        # The current search is quite efficient.
        # If it fails, it might be that no solution exists with x as smallest, or we hit time limit.
        
    return answers

# Quick test
if __name__ == "__main__":
    res = construct()
    solved = sum(1 for r in res if r is not None)
    print(f"Solved {solved}/254")
