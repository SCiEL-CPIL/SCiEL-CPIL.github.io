import numpy as np
import random
import time
from collections import defaultdict

N = 64
GRID_SIZE = N * N
ALL_POINTS = [(x, y) for x in range(N) for y in range(N)]

# Pre-compute squared distances between all pairs of grid points
# DIST[i][j] = squared distance between point i and point j
# Using a list of lists for faster random access in Python loops
DIST = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]

def precompute_distances():
    for i in range(GRID_SIZE):
        x1, y1 = ALL_POINTS[i]
        for j in range(i + 1, GRID_SIZE):
            x2, y2 = ALL_POINTS[j]
            dx = x1 - x2
            dy = y1 - y2
            d_sq = dx * dx + dy * dy
            DIST[i][j] = d_sq
            DIST[j][i] = d_sq

# Run pre-computation at module load time
precompute_distances()

def construct():
    start_time = time.time()
    deadline = start_time + 270.0
    
    best_set = []
    best_size = 0
    
    # Group points by y-coordinate
    y_groups = defaultdict(list)
    for i in range(GRID_SIZE):
        x, y = ALL_POINTS[i]
        y_groups[y].append(i)
    
    # Pre-compute squared distances for y-axis (same y)
    # For points with same y, distance is (x1-x2)^2
    
    indices = list(range(GRID_SIZE))
    
    # We will perform multiple random restarts with greedy + local search
    while time.time() < deadline:
        random.shuffle(indices)
        current_set = []
        current_set_set = set()
        
        for k in indices:
            if time.time() > deadline:
                break
            
            m = len(current_set)
            
            if m == 0:
                current_set.append(k)
                current_set_set.add(k)
                continue
            
            # Check if point k can be added to current_set
            valid = True
            
            # Collect distances from k to all points in current_set
            dists_from_k = {}
            for a in current_set:
                dists_from_k[a] = DIST[k][a]
            
            # Check 1: Duplicate distances from k
            seen_dists = set()
            for a in current_set:
                d_ka = dists_from_k[a]
                if d_ka in seen_dists:
                    valid = False
                    break
                seen_dists.add(d_ka)
            
            if not valid:
                continue
            
            # Check 2 and 3: For each pair (a, b) in current_set
            for i in range(m):
                a = current_set[i]
                d_ka = dists_from_k[a]
                for j in range(i + 1, m):
                    b = current_set[j]
                    d_ab = DIST[a][b]
                    d_kb = dists_from_k[b]
                    
                    if d_ka == d_ab or d_kb == d_ab:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                current_set.append(k)
                current_set_set.add(k)
        
        # Local search: try to swap points to improve the set
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
        
        # Try local search on current_set if it's competitive
        if len(current_set) >= best_size - 2 and time.time() < deadline - 5:
            # Try to swap each point with a candidate not in the set
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for i in range(len(current_set)):
                    if time.time() > deadline - 5:
                        break
                    removed = current_set[i]
                    # Try to add a point not in current_set
                    candidates = [k for k in indices if k not in current_set_set and k != removed]
                    random.shuffle(candidates)
                    for k in candidates[:50]:  # Limit candidates to speed up
                        if time.time() > deadline - 5:
                            break
                        # Check if k can replace removed
                        test_set = [p for j, p in enumerate(current_set) if j != i]
                        valid = True
                        dists_from_k = {a: DIST[k][a] for a in test_set}
                        
                        # Check 1: Duplicate distances
                        seen = set()
                        for a in test_set:
                            d = dists_from_k[a]
                            if d in seen:
                                valid = False
                                break
                            seen.add(d)
                        
                        if not valid:
                            continue
                        
                        # Check 2 and 3
                        for ii in range(len(test_set)):
                            a = test_set[ii]
                            d_ka = dists_from_k[a]
                            for jj in range(ii + 1, len(test_set)):
                                b = test_set[jj]
                                d_ab = DIST[a][b]
                                d_kb = dists_from_k[b]
                                if d_ka == d_ab or d_kb == d_ab:
                                    valid = False
                                    break
                            if not valid:
                                break
                        
                        if valid:
                            current_set[i] = k
                            current_set_set.discard(removed)
                            current_set_set.add(k)
                            improved = True
                            if len(current_set) > best_size:
                                best_size = len(current_set)
                                best_set = current_set[:]
                            break
    
    return [ALL_POINTS[i] for i in best_set]
