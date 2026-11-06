"""Triangular lattice packing: maximize edges for 32 points."""
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Precompute neighbors for a given point
    def neighbors(a, b):
        return [(a+1, b), (a-1, b), (a, b+1), (a, b-1), (a+1, b-1), (a-1, b+1)]
    
    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da*da + da*db + db*db
    
    def edge_count(S):
        """Count pairs at distance exactly 1."""
        Sset = set(S)
        cnt = 0
        for p in S:
            a, b = p
            for q in neighbors(a, b):
                if q in Sset and (a, b) < q:
                    cnt += 1
        return cnt
    
    def valid(S):
        if len(S) != 32:
            return False
        if len(set(S)) != 32:
            return False
        for p in S:
            if abs(p[0]) > 64 or abs(p[1]) > 64:
                return False
        # Check min distance >= 1
        Sset = set(S)
        for p in S:
            a, b = p
            for q in S:
                if p == q:
                    continue
                d2 = dist_sq(p, q)
                if d2 < 1:
                    return False
        return True
    
    # Start with a spiral-like configuration
    def spiral_config(n):
        """Generate a spiral configuration of n points on triangular lattice."""
        points = []
        # Use a known good pattern: hexagonal spiral
        # Center at (0,0), then expand in rings
        points.append((0, 0))
        if n <= 1:
            return points[:n]
        
        # Hexagonal spiral directions
        # Ring k has 6k points
        k = 1
        while len(points) < n:
            # Start of ring k
            # For triangular lattice, ring k starts at (k, 0)
            # and goes around
            # Actually, let's use a different approach
            # Generate all points in ring k
            ring_points = []
            # Ring k: points at "hexagonal distance" k from center
            # In triangular lattice coordinates, this is complex
            # Let's just do a simple spiral by trying all points in increasing distance
            pass
        
        # Simpler approach: just use a compact block
        # 16x2 strip gives 32 points with many edges
        # Let's count edges in 16x2 strip
        # Points: (a, b) for a in 0..15, b in 0..1
        # Edges: horizontal: 15*2 = 30
        # Vertical: 16 (between (a,0) and (a,1))
        # Diagonal: (a,b) to (a+1,b-1) and (a-1,b+1)
        # For b=0: (a,0) to (a+1,-1) - not in set
        # For b=1: (a,1) to (a+1,0) - in set if a+1 <= 15, so 15
        # (a,0) to (a-1,1) - in set if a-1 >= 0, so 15
        # Total: 30 + 16 + 15 + 15 = 76
        # That's exactly Harborth's bound!
        
        return [[a, b] for a in range(16) for b in range(2)]
    
    best_config = spiral_config(32)
    best_edges = edge_count(best_config)
    
    # Try to improve with local search
    random.seed(42)
    
    def try_improve(config, max_iter=5000):
        """Local search: try moving points to increase edge count."""
        config = [list(p) for p in config]
        Sset = set(tuple(p) for p in config)
        
        current_edges = edge_count(config)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Pick a random point
            idx = random.randint(0, 31)
            a, b = config[idx]
            
            # Try all nearby positions
            improved = False
            for da in range(-3, 4):
                for db in range(-3, 4):
                    if da == 0 and db == 0:
                        continue
                    na, nb = a + da, b + db
                    if abs(na) > 64 or abs(nb) > 64:
                        continue
                    new_pt = (na, nb)
                    
                    # Check if position is free
                    if new_pt in Sset:
                        continue
                    
                    # Check min distance constraint
                    ok = True
                    for j, p in enumerate(config):
                        if j == idx:
                            continue
                        if dist_sq(new_pt, tuple(p)) < 1:
                            ok = False
                            break
                    if not ok:
                        continue
                    
                    # Try the move
                    config[idx] = [na, nb]
                    Sset.remove((a, b))
                    Sset.add(new_pt)
                    
                    new_edges = edge_count(config)
                    
                    if new_edges > current_edges:
                        current_edges = new_edges
                        improved = True
                        break
                    else:
                        # Revert
                        config[idx] = [a, b]
                        Sset.remove(new_pt)
                        Sset.add((a, b))
            
            if improved:
                # Reset iteration counter on improvement
                max_iter = it + 5000
        
        return config, current_edges
    
    # Run local search
    improved_config, improved_edges = try_improve(best_config)
    
    if improved_edges > best_edges:
        best_config = improved_config
        best_edges = improved_edges
    
    # Try multiple random restarts
    for restart in range(5):
        if time.time() > deadline:
            break
        
        # Generate a random valid configuration
        random_config = _random_valid_config()
        if random_config is None:
            continue
        
        rc_edges = edge_count(random_config)
        if rc_edges > best_edges:
            best_config = random_config
            best_edges = rc_edges
        
        # Local search from random config
        improved_config, improved_edges = try_improve(random_config, max_iter=2000)
        if improved_edges > best_edges:
            best_config = improved_config
            best_edges = improved_edges
    
    return [[a, b] for a, b in best_config]


def _random_valid_config():
    """Generate a random valid configuration of 32 points."""
    import random
    
    # Start with the 16x2 strip and perturb
    base = [(a, b) for a in range(16) for b in range(2)]
    config = [list(p) for p in base]
    
    # Try to make it valid by ensuring min distance
    # The base is already valid, so just return it with some random shifts
    # Actually, let's generate a truly random one
    points = []
    attempts = 0
    while len(points) < 32 and attempts < 10000:
        attempts += 1
        a = random.randint(-20, 20)
        b = random.randint(-20, 20)
        p = (a, b)
        
        # Check if it's too close to existing points
        ok = True
        for q in points:
            da = a - q[0]
            db = b - q[1]
            if da*da + da*db + db*db < 1:
                ok = False
                break
        if ok:
            points.append(p)
    
    if len(points) != 32:
        return None
    
    return points
