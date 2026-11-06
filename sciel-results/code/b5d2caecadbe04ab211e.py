"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def _collinear(p1, p2, p3):
    """Check if three points are collinear."""
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

def _has_four_collinear(points):
    """Check if any four points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear(points[i], points[j], points[k]):
                    # Check if any other point is also on this line
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    dx = x2 - x1
                    dy = y2 - y1
                    for l in range(k+1, n):
                        if _collinear(points[i], points[j], points[l]):
                            return True
    return False

def _find_collinear_triples(points):
    """Find all collinear triples."""
    n = len(points)
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear(points[i], points[j], points[k]):
                    triples.append((i, j, k))
    return triples

def _max_independent_no_three(points):
    """
    Find the largest subset of points with no three collinear.
    Equivalent to: minimum hitting set of collinear triples.
    Score = n - (minimum number of points to remove to break all triples)
    """
    n = len(points)
    triples = _find_collinear_triples(points)
    
    if not triples:
        return n
    
    # Branch and bound: try to find minimum hitting set
    # A hitting set is a set of points that intersects every triple.
    # We want the minimum hitting set size.
    
    best = [n]  # worst case: remove all
    best_set = [set(range(n))]
    
    triples_set = set(triples)
    
    # Sort triples by "difficulty" - not critical for 20 points
    def bnb(remaining_triples, removed, depth):
        if not remaining_triples:
            if len(removed) < best[0]:
                best[0] = len(removed)
                best_set[0] = set(removed)
            return
        
        # Upper bound: if current removed + remaining needed >= best, prune
        # Lower bound: need at least 1 more if remaining_triples non-empty
        if len(removed) >= best[0]:
            return
        
        # Pick a triple with fewest "available" points (not already removed)
        # Actually, pick a triple and try removing each of its points
        triple = remaining_triples[0]
        rest = remaining_triples[1:]
        
        for pt in triple:
            if pt not in removed:
                new_removed = removed | {pt}
                # Remove all triples containing this point
                new_remaining = [t for t in rest if pt not in t]
                bnb(new_remaining, new_removed, depth+1)
    
    bnb(triples, set(), 0)
    
    return n - best[0]

def construct():
    deadline = time.time() + 140  # safety margin before 150s
    
    # Strategy: search for configurations with many collinear triples
    # but no four collinear.
    
    best_score = 20
    best_points = None
    
    # Start with a structured base and perturb
    random.seed(42)
    
    # Generate candidate configurations
    def generate_candidate():
        """Generate a random configuration of 20 points."""
        points = []
        used = set()
        while len(points) < 20:
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            if (x, y) not in used:
                used.add((x, y))
                points.append((x, y))
        return points
    
    # Try structured approaches first
    # Approach 1: Points on a small grid with perturbations
    def generate_grid_candidate():
        """Generate points from a grid structure."""
        points = []
        used = set()
        # Use a 5x4 grid region
        for x in range(-4, 5):
            for y in range(-3, 4):
                if len(points) >= 20:
                    break
                pt = (x, y)
                if pt not in used:
                    used.add(pt)
                    points.append(pt)
        return points[:20]
    
    # Check the grid candidate
    grid_pts = generate_grid_candidate()
    if not _has_four_collinear(grid_pts):
        score = _max_independent_no_three(grid_pts)
        if score < best_score:
            best_score = score
            best_points = grid_pts
    
    # Try random configurations with many collinear triples
    # Bias toward creating collinear triples
    def generate_biased_candidate():
        """Generate points biased toward collinear triples."""
        points = []
        used = set()
        
        # Place some points on a line (but only 3 max)
        # Then place others to create more triples
        for _ in range(20):
            if random.random() < 0.3 and len(points) >= 2:
                # Try to place a point to form a triple with two existing points
                for i in range(len(points)):
                    for j in range(i+1, len(points)):
                        x1, y1 = points[i]
                        x2, y2 = points[j]
                        # Extend the line
                        dx = x2 - x1
                        dy = y2 - y1
                        for t in range(1, 4):
                            nx = x2 + dx * t
                            ny = y2 + dy * t
                            if -100 <= nx <= 100 and -100 <= ny <= 100:
                                if (nx, ny) not in used:
                                    used.add((nx, ny))
                                    points.append((nx, ny))
                                    break
                            if len(points) >= 20:
                                break
                        if len(points) >= 20:
                            break
                    if len(points) >= 20:
                        break
            if len(points) >= 20:
                break
            # Otherwise random
            while len(points) < 20:
                x = random.randint(-50, 50)
                y = random.randint(-50, 50)
                if (x, y) not in used:
                    used.add((x, y))
                    points.append((x, y))
                    break
        
        return points[:20]
    
    # Try many random/biased configurations
    attempts = 0
    while time.time() < deadline and attempts < 2000:
        attempts += 1
        pts = generate_biased_candidate()
        if len(pts) != 20:
            continue
        if _has_four_collinear(pts):
            continue
        score = _max_independent_no_three(pts)
        if score < best_score:
            best_score = score
            best_points = pts
            if best_score <= 10:
                break
    
    # If no improvement, try a different structured approach
    if best_score == 20:
        # Try points on a modular curve: y = x^2 mod p, mapped to integers
        # This creates some collinear structure
        p = 23
        pts = []
        used = set()
        for x in range(-11, 12):
            y = (x * x) % p
            # Map to a range
            pt = (x, y)
            if pt not in used:
                used.add(pt)
                pts.append(pt)
        
        if len(pts) == 20 and not _has_four_collinear(pts):
            score = _max_independent_no_three(pts)
            if score < best_score:
                best_score = score
                best_points = pts
    
    if best_points is None:
        # Fallback: parabola (score 20)
        return [[x, x * x] for x in range(-10, 10)]
    
    return [[x, y] for x, y in best_points]

# EVOLVE-BLOCK-END
