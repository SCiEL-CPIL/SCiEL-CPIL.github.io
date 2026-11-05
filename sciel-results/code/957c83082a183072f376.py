# EVOLVE-BLOCK-START
import itertools
from math import gcd
import random
import sys

def count_two_point_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    line_to_points = {}
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            g = gcd(abs(A), gcd(abs(B), abs(C)))
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            if key not in line_to_points:
                line_to_points[key] = set()
            line_to_points[key].add(i)
            line_to_points[key].add(j)
    
    two_point_lines = sum(1 for pts in line_to_points.values() if len(pts) == 2)
    return two_point_lines

def all_collinear(points):
    if len(points) < 3:
        return True
    x1, y1 = points[0]
    x2, y2 = points[1]
    for i in range(2, len(points)):
        x3, y3 = points[i]
        if (x2-x1)*(y3-y1) != (y2-y1)*(x3-x1):
            return False
    return True

def construct():
    """
    Construct 12 points minimizing the number of 2-point lines.
    Uses a backtracking search with heuristics to find a configuration
    with 6 ordinary lines (the theoretical minimum).
    """
    N = 12
    best_score = float('inf')
    best_config = None
    
    # We search in a small grid to find the combinatorial structure,
    # then we can use those coordinates directly (as they are integers).
    # Grid size 0..10 is sufficient for most small configurations.
    
    def solve():
        points = []
        lines = {} # key: (A,B,C), value: set of point indices
        
        def add_point(p):
            idx = len(points)
            points.append(p)
            # Update lines
            for j in range(idx):
                x1, y1 = points[j]
                x2, y2 = p
                A = y2 - y1
                B = x1 - x2
                C = x2 * y1 - x1 * y2
                g = gcd(abs(A), gcd(abs(B), abs(C)))
                if g != 0:
                    A //= g
                    B //= g
                    C //= g
                if A < 0 or (A == 0 and B < 0):
                    A = -A
                    B = -B
                    C = -C
                key = (A, B, C)
                if key not in lines:
                    lines[key] = set()
                lines[key].add(idx)
                lines[key].add(j)
            return idx

        def remove_point(idx):
            # Rebuild lines from scratch for simplicity in backtracking
            # This is O(N^2) but N is small (12)
            pass # We'll just recalculate score from scratch at leaves
            
        def backtrack(depth):
            nonlocal best_score, best_config
            
            if best_score == 6: # Can't do better than 6
                return True
                
            if depth == N:
                score = count_two_point_lines(points)
                if score < best_score:
                    best_score = score
                    best_config = [list(p) for p in points]
                return best_score == 6
            
            # Heuristic: Try to place points that extend existing lines
            # Candidates: 
            # 1. Points on existing lines with 2 points (to make 3)
            # 2. Points that form a new 3-point line with 2 existing points
            # 3. Random points from a small grid
            
            cands = set()
            
            # 1. Extend lines
            for key, pts in lines.items():
                if len(pts) == 2:
                    i, j = list(pts)
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    # Find integer points on this line in a small range
                    # Line: A x + B y + C = 0
                    A, B, C = key
                    # We want to find (x,y) such that Ax+By+C=0
                    # And (x,y) not in points
                    # Search in range -10 to 10
                    for x in range(-10, 11):
                        val = -A*x - C
                        if B == 0:
                            if val == 0:
                                # x is fixed, y is free? No, if B=0, A!=0, x is fixed.
                                # We need a specific y.
                                # Actually if B=0, the line is vertical x = -C/A.
                                # We can pick any y not used.
                                for y in range(-10, 11):
                                    p = (x, y)
                                    if p not in points:
                                        cands.add(p)
                        else:
                            if val % B == 0:
                                y = val // B
                                if -10 <= y <= 10:
                                    p = (x, y)
                                    if p not in points:
                                        cands.add(p)
            
            # 2. Form new 3-point lines
            # For any pair of existing points, find a 3rd point that makes a line
            # This is similar to 1, but we also consider lines with 1 point? No.
            # If we have points i, j, and we place k, it forms a line.
            # We want to maximize 3-point lines.
            
            # 3. Random candidates to escape local minima
            for _ in range(20):
                x = random.randint(-10, 10)
                y = random.randint(-10, 10)
                p = (x, y)
                if p not in points:
                    cands.add(p)
            
            # Sort candidates by "potential"
            # Potential: number of existing points on the line defined by (p, existing)
            def potential(p):
                pot = 0
                for j in range(len(points)):
                    x1, y1 = points[j]
                    x2, y2 = p
                    A = y2 - y1
                    B = x1 - x2
                    C = x2 * y1 - x1 * y2
                    g = gcd(abs(A), gcd(abs(B), abs(C)))
                    if g != 0:
                        A //= g
                        B //= g
                        C //= g
                    if A < 0 or (A == 0 and B < 0):
                        A = -A
                        B = -B
                        C = -C
                    key = (A, B, C)
                    if key in lines:
                        pot += len(lines[key])
                return pot
            
            sorted_cands = sorted(cands, key=potential, reverse=True)
            
            for p in sorted_cands:
                if p in points:
                    continue
                # Check if adding p keeps the set non-collinear (if N>2)
                # And distinct
                if p in points:
                    continue
                
                # Add p
                old_lines = {k: set(v) for k, v in lines.items()}
                idx = add_point(p)
                
                # Pruning: If current ordinary lines > best_score, skip?
                # Hard to calculate incrementally.
                
                if backtrack(depth + 1):
                    return True
                
                # Undo
                points.pop()
                lines = old_lines
                
            return False
        
        # Initialize with 3 points to break symmetry
        # (0,0), (1,0), (0,1)
        add_point((0,0))
        add_point((1,0))
        add_point((0,1))
        
        # Run backtracking
        # To ensure we finish in time, we limit the search or use random restarts
        # Since we want to find 6, we try a few random seeds
        for seed in range(10):
            random.seed(seed)
            # Reset state
            points = []
            lines = {}
            add_point((0,0))
            add_point((1,0))
            add_point((0,1))
            
            if backtrack(3):
                if best_score == 6:
                    break
        
        return best_config

    config = solve()
    if config is None:
        # Fallback to 3x3 grid + 3 points
        config = [[x, y] for x in range(3) for y in range(3)]
        config.append([3, 0])
        config.append([3, 1])
        config.append([3, 2])
        
    return config

# EVOLVE-BLOCK-END
