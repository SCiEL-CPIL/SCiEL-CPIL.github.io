"""Triangular lattice packing: maximize edges for 32 points."""
import time
import random

def _neighbors(p):
    a, b = p
    return ((a + 1, b), (a - 1, b), (a, b + 1),
            (a, b - 1), (a + 1, b - 1), (a - 1, b + 1))

def _dist_sq(p, q):
    """Squared distance between two lattice points a+b*omega."""
    da = p[0] - q[0]
    db = p[1] - q[1]
    # |a + b*omega|^2 = a^2 + ab + b^2 for omega = exp(i*pi/3)
    return da * da + da * db + db * db

def _edge_count(Sset):
    """Count pairs at distance exactly 1."""
    cnt = 0
    for p in Sset:
        for q in _neighbors(p):
            if q in Sset and p < q:
                cnt += 1
    return cnt

def _valid(Sset):
    """Check validity: 32 distinct points, |a|,|b| <= 64, min distance >= 1."""
    if len(Sset) != 32:
        return False
    if len(set(Sset)) != 32:
        return False
    for p in Sset:
        if abs(p[0]) > 64 or abs(p[1]) > 64:
            return False
    for i, p in enumerate(Sset):
        for j in range(i + 1, len(Sset)):
            if _dist_sq(p, Sset[j]) < 1:
                return False
    return True

def _hex_spiral(n):
    """Generate points in a hexagonal spiral pattern centered at origin."""
    points = [(0, 0)]
    if n <= 1:
        return points[:n]
    
    # Hex spiral: layer k has 6*k points
    # Layer 0: center
    # Layer 1: 6 points
    # Layer 2: 12 points
    # Layer k: 6*k points
    layer = 1
    while len(points) < n:
        # Add 6*layer points in layer 'layer'
        # The hexagonal spiral goes around the hexagon
        # For layer k, the points are at distance k from center
        # Starting from (k, 0) and going counterclockwise
        
        # Six sides of the hexagon at layer k
        # Side 1: (k, 0) to (0, k) - direction (-1, 1)
        # Side 2: (0, k) to (-k, 0) - direction (-1, -1)  
        # Side 3: (-k, 0) to (-k, -k) - direction (0, -1)
        # Side 4: (-k, -k) to (0, -k) - direction (1, 0)
        # Side 5: (0, -k) to (k, 0) - direction (1, 1)
        # Side 6: (k, 0) to (k, k) - direction (0, 1)
        
        # Actually, let me use a simpler approach
        # For layer k, generate all points at hex-distance k
        # and add them in spiral order
        
        # The six directions from center for layer k:
        # (k, 0), (0, k), (-k, k), (-k, 0), (0, -k), (k, -k)
        
        # Let me just generate all points with max(|a|,|b|,|a-b|) = k
        # and sort them in spiral order
        
        # Simpler: just add points in a systematic way
        # Layer k consists of points where the hex distance is exactly k
        # Hex distance of (a,b) = max(|a|, |b|, |a-b|) ... no
        # Actually hex distance = (|a| + |b| + |a+b|) / 2 ... let me think
        
        # For triangular lattice with omega = exp(i*pi/3):
        # Point (a,b) represents a + b*omega
        # The hex distance from origin is max(|a|, |b|, |a+b|) ... 
        # Let me just use: for layer k, points are those with
        # max(|a|, |b|, |a-b|) ... hmm
        
        # Let me use a different approach: generate all points in a bounding
        # hexagon and sort by distance
        
        # Actually, simplest: just use a rectangular grid that's large enough
        # and pick the n closest to origin
        pass
    
    # Let me use a cleaner approach
    return points

def _generate_hexagonal(n):
    """Generate n points in a hexagonal arrangement."""
    if n <= 1:
        return [(0, 0)] if n == 1 else []
    
    # Use layers: layer 0 has 1 point, layer k has 6k points
    # Total up to layer L: 1 + 6*(1+2+...+L) = 1 + 3L(L+1)
    # For n=32: 1 + 3L(L+1) >= 32 => 3L(L+1) >= 31 => L(L+1) >= 10.33 => L >= 3
    # Layer 0: 1, Layer 1: 6, Layer 2: 12, Layer 3: 18
    # Cumulative: 1, 7, 19, 37
    # So for n=32, we need layers 0-2 (19 points) + 13 from layer 3
    
    points = [(0, 0)]
    
    # Add layers in order
    layer = 1
    while len(points) < n:
        # Generate 6*layer points for this layer
        # Hexagonal spiral layer k:
        # Start at (k, 0)
        # Go to (0, k): direction (-1, 1), k steps
        # Go to (-k, 0): direction (-1, -1), k steps  
        # Go to (-k, -k): direction (0, -1), k steps
        # Go to (0, -k): direction (1, 0), k steps
        # Go to (k, 0): direction (1, 1), k steps
        # Go to (k, k): direction (0, 1), k steps... wait that's not right
        
        # Let me trace a hexagonal spiral properly
        # Layer 1 (6 points): (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        # Layer 2 (12 points): 
        #   (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0),
        #   (-2,-1), (-1,-2), (0,-2), (1,-2), (2,-1), (2,0)... 
        #   wait, (2,0) is already counted. Let me be more careful.
        
        # Standard hex spiral:
        # For layer k, starting from (k, 0), we traverse 6 sides:
        # Side 1: (k,0) -> (k-k, 0+k) = (0, k), step (-1, 1), k points (excluding start)
        # Side 2: (0,k) -> (-k, 0), step (-1, -1), k points
        # Side 3: (-k,0) -> (-k, -k), step (0, -1), k points
        # Side 4: (-k,-k) -> (0, -k), step (1, 0), k points
        # Side 5: (0,-k) -> (k, 0), step (1, 1), k points
        # Side 6: (k,0) -> (k, k), step (0, 1), k points... 
        # But that gives 6k points total for the layer, which is correct.
        
        # However, I need to be careful about which points to include
        # to avoid duplicates and maintain the spiral order.
        
        # Let me generate all 6k points for layer k
        layer_points = []
        a, b = k, 0  # Start at (k, 0)
        
        for side in range(6):
            da, db = [(-1, 1), (-1, -1), (0, -1), (1, 0), (1, 1), (0, 1)][side]
            for step in range(k):
                a += da
                b += db
                layer_points.append((a, b))
        
        # Add as many as needed
        for p in layer_points:
            if len(points) >= n:
                break
            if p not in points:
                points.append(p)
        
        layer += 1
    
    return points[:n]

def _local_search(points, max_iter=50000, deadline=None):
    """Local search to maximize edge count."""
    points = list(points)
    Sset = set(points)
    current_edges = _edge_count(Sset)
    best_edges = current_edges
    best_points = list(points)
    
    # Precompute neighbor offsets
    offsets = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    
    start_time = time.time()
    
    for iteration in range(max_iter):
        if deadline and (time.time() - start_time) > deadline:
            break
        
        # Pick a random point
        i = random.randint(0, len(points) - 1)
        a, b = points[i]
        
        # Try moving to a random nearby position
        # Generate candidate positions in a small neighborhood
        candidates = []
        for r in range(1, 4):
            for da in range(-r, r + 1):
                for db in range(-r, r + 1):
                    if da == 0 and db == 0:
                        continue
                    if abs(da) > r or abs(db) > r:
                        continue
                    na, nb = a + da, b + db
                    if abs(na) <= 64 and abs(nb) <= 64:
                        candidates.append((na, nb))
        
        random.shuffle(candidates)
        
        improved = False
        for na, nb in candidates:
            new_pos = (na, nb)
            
            # Check if position is already occupied
            if new_pos in Sset:
                continue
            
            # Check minimum distance constraint
            valid = True
            for j, (pa, pb) in enumerate(points):
                if j == i:
                    continue
                if _dist_sq(new_pos, (pa, pb)) < 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Compute change in edge count
            # Remove edges from old position
            old_edges = 0
            for off in offsets:
                if (a + off[0], b + off[1]) in Sset:
                    old_edges += 1
            
            # Add edges from new position
            new_edges = 0
            for off in offsets:
                if (na + off[0], nb + off[1]) in Sset:
                    new_edges += 1
            
            delta = new_edges - old_edges
            
            if delta > 0:
                # Accept the move
                Sset.discard((a, b))
                Sset.add(new_pos)
                points[i] = new_pos
                current_edges += delta
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_points = list(points)
                improved = True
                break
        
        if not improved:
            # Try a random restart-like perturbation occasionally
            if iteration % 1000 == 0 and iteration > 0:
                # Randomly move 2-3 points
                for _ in range(random.randint(2, 3)):
                    i = random.randint(0, len(points) - 1)
                    a, b = points[i]
                    for _ in range(50):
                        na = a + random.randint(-3, 3)
                        nb = b + random.randint(-3, 3)
                        new_pos = (na, nb)
                        if abs(na) > 64 or abs(nb) > 64:
                            continue
                        if new_pos in Sset:
                            continue
                        valid = True
                        for j, (pa, pb) in enumerate(points):
                            if j == i:
                                continue
                            if _dist_sq(new_pos, (pa, pb)) < 1:
                                valid = False
                                break
                        if valid:
                            Sset.discard((a, b))
                            Sset.add(new_pos)
                            points[i] = new_pos
                            break
    
    return best_points, best_edges

def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s limit
    
    # Start with hexagonal arrangement
    points = _generate_hexagonal(32)
    Sset = set(points)
    
    # Verify validity
    if not _valid(Sset):
        # Fallback: use a simple valid configuration
        points = [(a, b) for a in range(8) for b in range(4)]
        Sset = set(points)
    
    best_points = list(points)
    best_edges = _edge_count(Sset)
    
    # Multiple restarts with local search
    random.seed(42)
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        # Run local search
        result_points, result_edges = _local_search(
            best_points, 
            max_iter=20000,
            deadline=deadline - elapsed
        )
        
        if result_edges > best_edges:
            best_points = result_points
            best_edges = result_edges
        
        # Check if we reached the theoretical optimum
        if best_edges >= 76:
            break
        
        # Try a different random perturbation for next restart
        perturbed = list(best_points)
        for _ in range(random.randint(3, 8)):
            i = random.randint(0, len(perturbed) - 1)
            a, b = perturbed[i]
            for _ in range(100):
                na = a + random.randint(-4, 4)
                nb = b + random.randint(-4, 4)
                if abs(na) > 64 or abs(nb) > 64:
                    continue
                new_pos = (na, nb)
                # Check it's not occupied
                if new_pos in set(perturbed):
                    continue
                # Check distance constraint
                valid = True
                for j, (pa, pb) in enumerate(perturbed):
                    if j == i:
                        continue
                    if _dist_sq(new_pos, (pa, pb)) < 1:
                        valid = False
                        break
                if valid:
                    perturbed[i] = new_pos
                    break
        
        # Use perturbed configuration for next local search
        # But keep best_points as the reference
        if time.time() - start_time > deadline - 5:
            break
    
    # Convert to list of lists
    return [[p[0], p[1]] for p in best_points]
