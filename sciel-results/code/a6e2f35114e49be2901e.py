import random
import math
from collections import defaultdict

def count_4_lines(points):
    """
    Returns the number of lines containing exactly 4 points.
    Returns -1 if any line contains 5 or more points.
    """
    n = len(points)
    if n != 16:
        return -1
    
    # Check distinctness
    if len(set(map(tuple, points))) != 16:
        return -1
        
    # Check bounds
    for p in points:
        if abs(p[0]) > 1000 or abs(p[1]) > 1000:
            return -1

    lines = defaultdict(list)
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            # Line: A*x + B*y = C
            # A = dy, B = -dx
            # C = A*x1 + B*y1
            A = dy
            B = -dx
            C = A*x1 + B*y1
            
            # Normalize the line equation to avoid duplicates
            # Find gcd of A, B, C
            if A == 0 and B == 0:
                continue # Should not happen for distinct points
            
            g_line = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g_line == 0:
                g_line = 1
            A_n = A // g_line
            B_n = B // g_line
            C_n = C // g_line
            
            # Ensure canonical form: A_n > 0 or (A_n == 0 and B_n > 0)
            if A_n < 0 or (A_n == 0 and B_n < 0):
                A_n = -A_n
                B_n = -B_n
                C_n = -C_n
                
            key = (A_n, B_n, C_n)
            lines[key].append(i)
            
    count_4 = 0
    for key, pts in lines.items():
        if len(pts) > 4:
            return -1
        if len(pts) == 4:
            count_4 += 1
            
    return count_4

def construct():
    # GF(16) construction
    # Primitive polynomial: x^4 + x + 1
    # Elements are represented as integers 0..15 (binary representation of polynomials)
    # Addition is XOR
    # Multiplication is defined by the primitive polynomial
    
    # Precompute multiplication table for GF(16)
    def gf16_mult(a, b):
        if a == 0 or b == 0:
            return 0
        result = 0
        while b:
            if b & 1:
                result ^= a
            b >>= 1
            a <<= 1
            # Reduce by x^4 + x + 1 (0b10011 = 19)
            if a & 0b10000:  # if degree >= 4
                a ^= 0b10011
        return result & 0b1111
    
    # Map GF(16) elements to 4x4 grid coordinates
    # We want to map each element to a unique (x, y) in {0,1,2,3}x{0,1,2,3}
    # Let's use the binary representation: element = b3 b2 b1 b0
    # x = b3 * 2 + b2, y = b1 * 2 + b0
    # This maps 0..15 to 16 distinct points in 4x4 grid
    
    gf16_to_grid = {}
    for elem in range(16):
        b3 = (elem >> 3) & 1
        b2 = (elem >> 2) & 1
        b1 = (elem >> 1) & 1
        b0 = elem & 1
        x = b3 * 2 + b2
        y = b1 * 2 + b0
        gf16_to_grid[elem] = (x, y)
    
    # In GF(16), every line has exactly 4 points
    # We need to map these to integer coordinates while preserving the line structure
    # The key insight: if we use a field isomorphism to integers, we can maintain collinearity
    # But we need integer coordinates, so we use an affine transformation
    
    # First, get the 16 points in GF(16) grid representation
    gf16_points = [gf16_to_grid[i] for i in range(16)]
    
    # Now we need to map these to integer coordinates
    # The problem is that the GF(16) lines don't directly correspond to integer lines
    # We need to find a mapping that preserves the 4-collinearity structure
    
    # Alternative approach: Use the fact that in GF(16), we can represent points as (x, y) 
    # where x, y are in GF(16), but that gives 256 points, not 16.
    
    # Let me reconsider. The 16 points in GF(16) can be thought of as the affine plane AG(2, 4)
    # but that has 16 points and lines of size 4.
    # Wait, AG(2, 4) has 4^2 = 16 points, and each line has 4 points.
    # The number of lines in AG(2, 4) is (16*15/2) / (4*3/2) = 120/6 = 20 lines.
    # But we need to map this to integer coordinates.
    
    # The issue is that AG(2, 4) is not a subset of the integer plane in a way that preserves lines.
    # We need to find an embedding into Z^2 that preserves the 4-collinearity.
    
    # Let's try a different approach: use the 4x4 grid but with a non-linear mapping
    # that breaks the 5-collinearities while preserving 4-collinearities.
    
    # Actually, the standard 4x4 grid has many 5-collinearities (diagonals, etc.)
    # We need to perturb it.
    
    # Let's use simulated annealing starting from the 4x4 grid, but with a better strategy
    
    points = [[x, y] for x in range(4) for y in range(4)]
    
    best_score = count_4_lines(points)
    best_points = [p[:] for p in points]
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.99
    
    import time
    start_time = time.time()
    
    while T > T_min and (time.time() - start_time) < 110:
        # Pick a random point
        i = random.randint(0, 15)
        
        # Try multiple perturbation strategies
        strategy = random.randint(0, 2)
        
        if strategy == 0:
            # Small step
            delta_x = random.randint(-2, 2)
            delta_y = random.randint(-2, 2)
        elif strategy == 1:
            # Medium step
            delta_x = random.randint(-5, 5)
            delta_y = random.randint(-5, 5)
        else:
            # Large step
            delta_x = random.randint(-20, 20)
            delta_y = random.randint(-20, 20)
        
        new_points = [p[:] for p in points]
        new_points[i][0] += delta_x
        new_points[i][1] += delta_y
        
        new_score = count_4_lines(new_points)
        
        if new_score < 0:
            # Invalid, reject
            continue
            
        delta = new_score - best_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            points = new_points
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
                
        T *= cooling
        
    # Final check
    if count_4_lines(best_points) < 0:
        # Fallback to grid
        return [[x, y] for x in range(4) for y in range(4)]
        
    return best_points
