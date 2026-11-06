"""Improved program for no-isosceles set task using greedy + local search."""
import numpy as np
import random
import time

N = 64
NUM_POINTS = N * N
DEADLINE = time.time() + 280  # Leave 20s safety margin

# Pre-compute all grid points as indices
# Index i corresponds to point (i % N, i // N)
points = np.array([(x, y) for y in range(N) for x in range(N)], dtype=np.int32)

def idx_to_point(i):
    return (i % N, i // N)

def point_to_idx(x, y):
    return y * N + x

def squared_dist(i1, i2):
    """Squared distance between two grid point indices."""
    x1, y1 = i1 % N, i1 // N
    x2, y2 = i2 % N, i2 // N
    return (x1 - x2)**2 + (y1 - y2)**2

def is_isosceles(p1, p2, p3):
    """Check if three points form an isosceles triangle (including degenerate)."""
    d12 = squared_dist(p1, p2)
    d13 = squared_dist(p1, p3)
    d23 = squared_dist(p2, p3)
    # Isosceles if any two distances are equal (and not all three equal which would be equilateral, also isosceles)
    # But we must exclude degenerate cases where points are collinear and one is between the other two?
    # Actually, the problem says "no three points forming an isosceles triangle". 
    # Typically, a triangle requires non-collinear points. But let's check the standard definition.
    # If points are collinear, they don't form a triangle. So we should only consider non-collinear triples.
    # However, many formulations of this problem include degenerate isosceles triangles (collinear with one point in the middle).
    # Let's assume the standard geometric definition: a triangle is isosceles if it has at least two equal sides.
    # For collinear points, no triangle is formed. So we need to check collinearity.
    
    # Check collinearity: cross product of (p2-p1) and (p3-p1) is 0
    x1, y1 = p1 % N, p1 // N
    x2, y2 = p2 % N, p2 // N
    x3, y3 = p3 % N, p3 // N
    cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    if cross == 0:
        return False  # Collinear, not a triangle
    
    return d12 == d13 or d12 == d23 or d13 == d23

def count_conflicts(S):
    """Count the number of isosceles triangles in set S."""
    conflicts = 0
    K = len(S)
    for i in range(K):
        for j in range(i+1, K):
            for k in range(j+1, K):
                if is_isosceles(S[i], S[j], S[k]):
                    conflicts += 1
    return conflicts

def greedy_construct():
    """Greedy construction: add points one by one if they don't create an isosceles triangle."""
    S = []
    # Order points by some heuristic, e.g., random or by distance from center
    indices = list(range(NUM_POINTS))
    random.shuffle(indices)
    
    for idx in indices:
        # Check if adding idx to S creates any isosceles triangle
        valid = True
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                if is_isosceles(S[i], S[j], idx):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            S.append(idx)
    return S

def local_search(S, time_limit):
    """Improve S using local search: try to swap or add/remove points."""
    S = list(S)
    current_conflicts = count_conflicts(S)
    
    while time.time() < time_limit:
        improved = False
        # Try adding a random point
        for _ in range(1000):
            candidate = random.randint(0, NUM_POINTS-1)
            if candidate not in S:
                new_S = S + [candidate]
                new_conflicts = count_conflicts(new_S)
                if new_conflicts < current_conflicts:
                    S = new_S
                    current_conflicts = new_conflicts
                    improved = True
                    break
        
        # Try removing a point
        if not improved and current_conflicts > 0:
            for _ in range(1000):
                if not S:
                    break
                idx_to_remove = random.randint(0, len(S)-1)
                candidate_idx = S[idx_to_remove]
                new_S = S[:idx_to_remove] + S[idx_to_remove+1:]
                new_conflicts = count_conflicts(new_S)
                if new_conflicts < current_conflicts:
                    S = new_S
                    current_conflicts = new_conflicts
                    improved = True
                    break
        
        # Try swapping
        if not improved and current_conflicts > 0:
            for _ in range(1000):
                if len(S) < 2:
                    break
                idx_in = random.randint(0, len(S)-1)
                candidate_out = S[idx_in]
                candidate_in = random.randint(0, NUM_POINTS-1)
                if candidate_in not in S:
                    new_S = S[:idx_in] + [candidate_in] + S[idx_in+1:]
                    new_conflicts = count_conflicts(new_S)
                    if new_conflicts < current_conflicts:
                        S = new_S
                        current_conflicts = new_conflicts
                        improved = True
                        break
        
        if not improved:
            # Stagnation, try a more aggressive move
            if current_conflicts == 0:
                break
            # Random restart with perturbation
            S = S[:]
            for _ in range(5):
                if S:
                    idx_to_remove = random.randint(0, len(S)-1)
                    S.pop(idx_to_remove)
            current_conflicts = count_conflicts(S)
    
    return S

def construct():
    best_S = []
    best_size = 0
    
    start_time = time.time()
    
    # Try multiple greedy constructions with different orderings
    for attempt in range(50):
        if time.time() > DEADLINE - 30:
            break
        
        # Different greedy strategies
        indices = list(range(NUM_POINTS))
        if attempt % 3 == 0:
            random.shuffle(indices)
        elif attempt % 3 == 1:
            # Sort by distance from center
            indices.sort(key=lambda i: (i % N - N//2)**2 + (i // N - N//2)**2)
        else:
            # Sort by x then y
            indices.sort(key=lambda i: (i // N, i % N))
        
        S = []
        for idx in indices:
            if time.time() > DEADLINE - 30:
                break
            valid = True
            for i in range(len(S)):
                for j in range(i+1, len(S)):
                    if is_isosceles(S[i], S[j], idx):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S.append(idx)
        
        if len(S) > best_size:
            best_S = S[:]
            best_size = len(S)
        
        # Local search to improve
        if time.time() < DEADLINE - 10:
            S_improved = local_search(S, DEADLINE - 10)
            if len(S_improved) > best_size:
                best_S = S_improved
                best_size = len(S_improved)
    
    # Convert indices to (x, y) tuples
    return [idx_to_point(i) for i in best_S]
