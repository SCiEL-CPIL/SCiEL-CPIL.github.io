"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100.0  # 100s safety margin
    
    # Define the lattice neighborhood
    # (a, b) neighbors: (a+1, b), (a-1, b), (a, b+1), (a, b-1), (a+1, b-1), (a-1, b+1)
    
    # Bounding box for candidates
    # 32 points fit easily in a 10x10 box
    L = 10
    candidates = [(a, b) for a in range(-L, L+1) for b in range(-L, L+1)]
    cand_set = set(candidates)
    
    # Pre-compute neighbors for each candidate within the candidate set
    neighbors = {}
    for p in candidates:
        a, b = p
        nbs = []
        for da, db in [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]:
            q = (a+da, b+db)
            if q in cand_set:
                nbs.append(q)
        neighbors[p] = nbs

    # Initial seed: Hexagon of radius 3 minus 5 corners
    # This gives 75 edges.
    hex_points = [(a, b) for a in range(-3, 4) for b in range(-3, 4) if max(abs(a), abs(b), abs(a+b)) <= 3]
    to_remove = [(3,0), (0,3), (-3,3), (-3,0), (0,-3)]
    initial_points = set(p for p in hex_points if p not in to_remove)
    
    # If initial is not size 32, fallback to random
    if len(initial_points) != 32:
        initial_points = set(random.sample(candidates, 32))
        
    points = initial_points
    
    # Calculate edge count
    def count_edges(pts):
        count = 0
        for p in pts:
            for q in neighbors[p]:
                if q in pts:
                    count += 1
        return count // 2

    current_edges = count_edges(points)
    best_points = set(points)
    best_edges = current_edges
    
    # Local Search: Swap moves
    # We try to swap a point in 'points' with a point outside 'points'
    # to increase edge count.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle points to avoid getting stuck in local minima due to order
        point_list = list(points)
        random.shuffle(point_list)
        
        for p in point_list:
            if time.time() > deadline:
                break
            # Edges lost by removing p
            lost = 0
            for q in neighbors[p]:
                if q in points:
                    lost += 1
            
            # Try to find a replacement q
            # Candidates are outside 'points'
            best_gain = -1
            best_q = None
            
            for q in candidates:
                if q in points:
                    continue
                gained = 0
                for r in neighbors[q]:
                    if r in points:
                        gained += 1
                
                delta = gained - lost
                if delta > best_gain:
                    best_gain = delta
                    best_q = q
            
            if best_gain > 0:
                # Perform swap
                points.remove(p)
                points.add(best_q)
                current_edges += best_gain
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_points = set(points)
                improved = True
                # Restart loop after a swap to re-evaluate all points
                break
        
    # Return the best found
    return [[a, b] for a, b in best_points]
# EVOLVE-BLOCK-END
