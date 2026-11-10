import time
import random
import math
import numpy as np

N = 64
NUM_POINTS = N * N

# Precompute coordinates and distances
coords = np.array([(i // N, i % N) for i in range(NUM_POINTS)], dtype=np.int32)
dx = coords[:, 0:1] - coords[0:1, :]
dy = coords[:, 1:2] - coords[0:1, :]
dist = dx**2 + dy**2

def construct():
    start_time = time.time()
    deadline = start_time + 290.0
    
    rng = random.Random(42)
    best_size = 0
    best_selected = set()
    
    # Run multiple trials
    # Each trial: Greedy with random order, then Local Search
    for trial in range(20):
        if time.time() > deadline:
            break
            
        selected = set()
        dist_sets = {}
        
        # Greedy Phase
        candidates = list(range(NUM_POINTS))
        rng.shuffle(candidates)
        
        for p in candidates:
            if time.time() > deadline:
                break
            if len(selected) == 0:
                selected.add(p)
                dist_sets[p] = set()
                continue
                
            # Check validity
            valid = True
            seen_dists = set()
            for q in selected:
                d = dist[p, q]
                if d in seen_dists:
                    valid = False
                    break
                seen_dists.add(d)
                if d in dist_sets[q]:
                    valid = False
                    break
                    
            if valid:
                selected.add(p)
                dist_sets[p] = set()
                for q in selected:
                    if q != p:
                        d = dist[p, q]
                        dist_sets[p].add(d)
                        dist_sets[q].add(d)
        
        # Local Search Phase
        # Try to add random points, remove if stuck
        current_size = len(selected)
        if current_size > best_size:
            best_size = current_size
            best_selected = selected.copy()
            
        stuck_count = 0
        while time.time() < deadline and stuck_count < 50:
            # Try to add a random point
            added = False
            for _ in range(100):
                if time.time() > deadline:
                    break
                p = rng.randint(0, NUM_POINTS - 1)
                if p in selected:
                    continue
                    
                valid = True
                seen_dists = set()
                for q in selected:
                    d = dist[p, q]
                    if d in seen_dists:
                        valid = False
                        break
                    seen_dists.add(d)
                    if d in dist_sets[q]:
                        valid = False
                        break
                        
                if valid:
                    selected.add(p)
                    dist_sets[p] = set()
                    for q in selected:
                        if q != p:
                            d = dist[p, q]
                            dist_sets[p].add(d)
                            dist_sets[q].add(d)
                    added = True
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_selected = selected.copy()
                    break
            if not added:
                stuck_count += 1
                # Remove a random point
                if selected:
                    p_to_remove = rng.choice(list(selected))
                    selected.remove(p_to_remove)
                    for q in selected:
                        dist_sets[q].remove(dist[p_to_remove, q])
                    del dist_sets[p_to_remove]
                    current_size -= 1

    result = [(i // N, i % N) for i in best_selected]
    return result
