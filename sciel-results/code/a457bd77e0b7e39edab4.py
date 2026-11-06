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
    for a, b in Sset:
        if abs(a) > 64 or abs(b) > 64:
            return False
    # Check mutual distance at least 1 (squared distance >= 1)
    pts = list(Sset)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if _dist_sq(pts[i], pts[j]) < 1:
                return False
    return True

def _hexagon_config():
    """Generate a hexagonal configuration of 32 points on triangular lattice.
    
    Centered hexagon with side length 3 gives 1+6+12+18 = 37 points (too many).
    Side length 2 gives 1+6+12 = 19 points.
    We need 32, so use a hexagon of side 3 (37 pts) and remove 5 points from the outer ring.
    
    Actually, let's think differently. A hexagon of radius r (distance from center) has:
    - r=0: 1 point
    - r=1: 6 points (total 7)
    - r=2: 12 points (total 19)
    - r=3: 18 points (total 37)
    
    So we take all points with hex distance <= 2 (19 points) plus 13 from the r=3 ring.
    """
    points = []
    # Add center
    points.append((0, 0))
    # Add hex ring 1 (6 points)
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    points.extend(ring1)
    # Add hex ring 2 (12 points)
    ring2 = [
        (2, 0), (2, -1), (2, -2),
        (1, -2), (0, -2), (-1, -2),
        (-2, -1), (-2, 0), (-2, 1),
        (-1, 2), (0, 2), (1, 2)
    ]
    points.extend(ring2)
    # Now we have 19 points. Need 13 more from ring 3.
    # Ring 3 has 18 points. Pick 13 that maximize edges.
    # Ring 3 points (hex distance exactly 3):
    ring3 = [
        (3, 0), (3, -1), (3, -2), (3, -3),
        (2, -3), (1, -3), (0, -3), (-1, -3),
        (-2, -2), (-3, -1), (-3, 0), (-3, 1),
        (-2, 3), (-1, 3), (0, 3), (1, 3),
        (2, 3), (3, 2), (3, 1)  # Wait, let me recompute
    ]
    # Let me properly generate ring 3
    ring3 = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if a == 0 and b == 0:
                continue
            # Hex distance in axial coordinates: max(|a|, |b|, |a+b|) ... 
            # Actually for triangular lattice with basis (1,0) and (0,1) representing 1 and omega:
            # The hex distance is max(|a|, |b|, |a-b|) ... let me just use the proper formula
            # For omega = exp(i*pi/3), the six neighbors are (±1,0), (0,±1), (1,-1), (-1,1)
            # Hex distance: we need to find shortest path using those 6 moves
            # This is: max(|a|, |b|, |a+b|) ... no
            # Actually the hex distance for this coordinate system is:
            # Let's use: d = max(|a|, |b|, |a-b|) ... hmm
            # The six directions are (1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)
            # Hex distance = max(|a|, |b|, |a+b|) ... let me verify:
            # (1,0): max(1,0,1)=1 ✓
            # (1,-1): max(1,1,0)=1 ✓  
            # (0,1): max(0,1,1)=1 ✓
            # (2,0): max(2,0,2)=2 ✓
            # (2,-1): max(2,1,1)=2 ✓
            # (1,1): max(1,1,2)=2 ✓ (needs 2 steps: (1,0)+(0,1))
            # Yes! Hex distance = max(|a|, |b|, |a+b|)
            hd = max(abs(a), abs(b), abs(a + b))
            if hd == 3:
                ring3.append((a, b))
    
    # ring3 should have 18 points
    # Pick 13 of them. To maximize edges, we want points adjacent to existing points.
    # Each ring-3 point is adjacent to 2 or 3 ring-2 points.
    # Let's just pick 13 consecutive ones from the ring for a good start.
    # Sort ring3 by angle and take 13
    import math
    def angle(p):
        # Convert to Cartesian: x = a + b*cos(pi/3) = a + b/2, y = b*sin(pi/3) = b*sqrt(3)/2
        x = p[0] + p[1] * 0.5
        y = p[1] * (3 ** 0.5) / 2
        return math.atan2(y, x)
    
    ring3.sort(key=angle)
    # Take 13 points
    selected = ring3[:13]
    points.extend(selected)
    
    return points

def _local_search(Sset, max_iters=50000, deadline=None):
    """Local search: try moving points to improve edge count while maintaining validity."""
    current_edges = _edge_count(Sset)
    best_edges = current_edges
    best_set = set(Sset)
    
    # Candidate positions to try
    # Use a bounded region
    candidates = []
    for a in range(-6, 7):
        for b in range(-6, 7):
            candidates.append((a, b))
    
    rng = random.Random(42)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random point to move
        idx = rng.randrange(len(Sset))
        old_pos = Sset[idx]
        
        # Try a few random candidate positions
        tried = 0
        while tried < 8:
            tried += 1
            new_pos = rng.choice(candidates)
            if new_pos == old_pos:
                continue
            if new_pos in Sset:
                continue
            
            # Check if new position is valid (distance >= 1 from all others)
            valid = True
            for other in Sset:
                if other == old_pos:
                    continue
                if _dist_sq(new_pos, other) < 1:
                    valid = False
                    break
            if not valid:
                continue
            
            # Compute new edge count
            Sset[idx] = new_pos
            new_edges = _edge_count(Sset)
            
            if new_edges > current_edges:
                current_edges = new_edges
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_set = set(Sset)
            else:
                # Revert
                Sset[idx] = old_pos
        
        # Also try removing and re-adding (swap with a candidate)
        if rng.random() < 0.3:
            idx = rng.randrange(len(Sset))
            old_pos = Sset[idx]
            Sset[idx] = None  # temporarily remove
            for _ in range(5):
                new_pos = rng.choice(candidates)
                if new_pos in Sset:
                    continue
                valid = True
                for other in Sset:
                    if other is None:
                        continue
                    if _dist_sq(new_pos, other) < 1:
                        valid = False
                        break
                if not valid:
                    continue
                Sset[idx] = new_pos
                new_edges = _edge_count(Sset)
                if new_edges > current_edges:
                    current_edges = new_edges
                    if new_edges > best_edges:
                        best_edges = new_edges
                        best_set = set(Sset)
                else:
                    Sset[idx] = old_pos
                    break
            if Sset[idx] is None:
                Sset[idx] = old_pos
    
    return list(best_set)

def _simulated_annealing(Sset, deadline):
    """Simulated annealing for better exploration."""
    current_edges = _edge_count(Sset)
    best_edges = current_edges
    best_set = set(Sset)
    
    candidates = []
    for a in range(-8, 9):
        for b in range(-8, 9):
            candidates.append((a, b))
    
    rng = random.Random(123)
    T = 1.0
    cooling = 0.9999
    
    while time.time() < deadline and T > 0.01:
        idx = rng.randrange(len(Sset))
        old_pos = Sset[idx]
        new_pos = rng.choice(candidates)
        
        if new_pos == old_pos or new_pos in Sset:
            T *= cooling
            continue
        
        # Check validity
        valid = True
        for other in Sset:
            if other == old_pos:
                continue
            if _dist_sq(new_pos, other) < 1:
                valid = False
                break
        if not valid:
            T *= cooling
            continue
        
        Sset[idx] = new_pos
        new_edges = _edge_count(Sset)
        delta = new_edges - current_edges
        
        if delta > 0 or rng.random() < (2.718281828 ** (delta / T)):
            current_edges = new_edges
            if new_edges > best_edges:
                best_edges = new_edges
                best_set = set(Sset)
        else:
            Sset[idx] = old_pos
        
        T *= cooling
    
    return list(best_set)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with hexagonal configuration
    Sset = _hexagon_config()
    
    # Ensure we have exactly 32 distinct valid points
    Sset = list(set(Sset))
    if len(Sset) != 32:
        # Fallback: generate a valid configuration
        Sset = [(a, b) for a in range(8) for b in range(4)]
    
    # Local search
    result = _local_search(Sset, max_iters=100000, deadline=deadline)
    
    # Simulated annealing for further improvement
    if time.time() < deadline - 5:
        result2 = _simulated_annealing(result, deadline - 2)
        if _edge_count(result2) > _edge_count(result):
            result = result2
    
    # Final verification
    result = list(set(result))
    if len(result) != 32:
        # Fallback
        result = [(a, b) for a in range(8) for b in range(4)]
    
    return [[a, b] for a, b in result]
