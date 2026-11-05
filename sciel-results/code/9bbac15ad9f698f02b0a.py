"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

# Precompute solutions for all n in [3, 256] at module load time.
# This ensures construct() is O(1) per lookup and well within time limits.

def _solve_all():
    """
    Find distinct positive integers x, y, z for each n in 3..256
    such that 1/x + 1/y + 1/z = 4/n.
    Returns a dictionary mapping n to [x, y, z] or None.
    """
    solutions = {}
    
    # We assume x < y < z.
    # Then 3/x > 4/n => x < 3n/4.
    # And 1/x < 4/n => x > n/4.
    # So x is in (n/4, 3n/4).
    
    # For each n, iterate x.
    # For a given x, let A = 4x - n, B = nx.
    # 1/y + 1/z = A/B
    # (Ay - B)(Az - B) = B^2
    # Let d = Ay - B. Then d | B^2.
    # y = (d + B) / A
    # z = (B^2/d + B) / A
    # We need y, z integers, x < y < z.
    # y > x => (d+B)/A > x => d > Ax - B.
    # z > y => B^2/d > d => d < B.
    # So we iterate d from max(1, Ax - B + 1) to B-1.
    # Check if B^2 % d == 0.
    
    for n in range(3, 257):
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        found = None
        for x in range(x_min, x_max + 1):
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            B_sq = B * B
            
            # d range: (Ax - B, B)
            # Ax - B = x(4x - n) - nx = 4x^2 - 2nx
            d_start = 4 * x * x - 2 * n * x + 1
            if d_start < 1:
                d_start = 1
            d_end = B - 1  # d < B for z > y
            
            # Optimization: If d_start > d_end, skip
            if d_start > d_end:
                continue
                
            for d in range(d_start, d_end + 1):
                if B_sq % d == 0:
                    # Check y
                    num_y = d + B
                    if num_y % A != 0:
                        continue
                    y = num_y // A
                    
                    # Check z
                    d_prime = B_sq // d
                    num_z = d_prime + B
                    if num_z % A != 0:
                        continue
                    z = num_z // A
                    
                    # Verify distinctness and order
                    if x < y < z:
                        # Final verification
                        if n * (y * z + x * z + x * y) == 4 * x * y * z:
                            found = [x, y, z]
                            break
            if found:
                break
        solutions[n] = found
        
    return solutions

# Compute the table once at import time
_SOLUTIONS = _solve_all()

def construct():
    """
    Returns 254 rows for n=3..256.
    Each row is None or three DISTINCT positive integer denominators x,y,z
    with 1/x+1/y+1/z=4/n.
    """
    answers = []
    for n in range(3, 257):
        sol = _SOLUTIONS.get(n)
        if sol is not None:
            # Ensure z <= 10^12
            if sol[2] <= 1000000000000:
                answers.append(sol)
            else:
                answers.append(None)
        else:
            answers.append(None)
    return answers

# EVOLVE-BLOCK-END
