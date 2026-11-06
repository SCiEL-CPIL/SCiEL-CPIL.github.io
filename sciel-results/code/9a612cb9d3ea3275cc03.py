import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140 seconds limit
    
    # Helper to check if a set of points has 4 collinear
    def has_four_collinear(points):
        n = len(points)
        # Map slope to count? No, just check all pairs and count.
        # For speed, we can use a dictionary of lines.
        # Line representation: (A, B, C) for Ax + By + C = 0, normalized.
        lines = {}
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                A = y1 - y2
                B = x2 - x1
                C = x1 * y2 - x2 * y1
                # Normalize
                g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
                if g != 0:
                    A //= g
                    B //= g
                    C //= g
                # Ensure consistent sign
                if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                    A = -A
                    B = -B
                    C = -C
                key = (A, B, C)
                if key in lines:
                    lines[key].append(i)
                else:
                    lines[key] = [i, j]
        
        for key, pts in lines.items():
            if len(pts) >= 4:
                return True
        return False

    # Helper to count collinear triples
    def count_triples(points):
        n = len(points)
        lines = {}
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                A = y1 - y2
                B = x2 - x1
                C = x1 * y2 - x2 * y1
                g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
                if g != 0:
                    A //= g
                    B //= g
                    C //= g
                if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                    A = -A
                    B = -B
                    C = -C
                key = (A, B, C)
                if key in lines:
                    lines[key].append(i)
                else:
                    lines[key] = [i, j]
        
        count = 0
        for key, pts in lines.items():
            if len(pts) >= 3:
                # Number of triples is C(k, 3)
                k = len(pts)
                count += k * (k - 1) * (k - 2) // 6
        return count

    # Generate a random valid point
    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]

    # Initialize points
    # We want to start with a configuration that is likely to have high density of triples
    # but is valid.
    # A grid is bad because of 4-collinear.
    # Let's just start random and let local search do the work.
    
    best_points = None
    best_triples = -1
    
    # Try a few random restarts
    for restart in range(5):
        points = []
        for _ in range(20):
            p = random_point()
            points.append(p)
        
        # Ensure no duplicates and no 4 collinear
        # Simple rejection sampling for initialization might be slow if random is sparse.
        # Let's try to build it greedily.
        
        points = []
        while len(points) < 20:
            p = random_point()
            # Check duplicates
            if p in points:
                continue
            # Check 4 collinear with new point
            new_points = points + [p]
            if not has_four_collinear(new_points):
                points.append(p)
        
        # Local search
        current_triples = count_triples(points)
        if current_triples > best_triples:
            best_triples = current_triples
            best_points = points.copy()
        
        # Simulated Annealing / Local Search
        T = 1.0
        for i in range(10000): # Fixed iterations per restart, or time based
            if time.time() > deadline:
                break
            
            idx = random.randint(0, 19)
            old_p = points[idx]
            new_p = random_point()
            
            if new_p == old_p:
                continue
            if new_p in points:
                continue
                
            # Check validity of new configuration
            # Temporarily replace
            points[idx] = new_p
            if has_four_collinear(points):
                points[idx] = old_p
                continue
            else:
                new_triples = count_triples(points)
                delta = new_triples - current_triples
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_triples = new_triples
                    if current_triples > best_triples:
                        best_triples = current_triples
                        best_points = points.copy()
                else:
                    points[idx] = old_p
            
            T *= 0.999 # Cooling
            
        if time.time() > deadline:
            break
            
    return best_points
