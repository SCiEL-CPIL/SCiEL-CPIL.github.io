import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    N = 11
    sqrt3 = math.sqrt(3)
    
    def axial_to_cartesian(q, r):
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        return (x, y)
    
    def generate_hex_grid(radius):
        positions = []
        for q in range(-radius, radius + 1):
            for r in range(-radius, radius + 1):
                if abs(q + r) <= radius:
                    positions.append((q, r))
        return positions
    
    def compute_min_enclosing_hexagon(cart_positions):
        min_S = 0.0
        for (px, py) in cart_positions:
            for i in range(6):
                d = math.radians(30 + i * 60)
                proj = px * math.cos(d) + py * math.sin(d)
                S_candidate = 1.0 + (2.0 / sqrt3) * proj
                if S_candidate > min_S:
                    min_S = S_candidate
        return min_S
    
    def center_positions(cart_positions):
        n = len(cart_positions)
        cx = sum(p[0] for p in cart_positions) / n
        cy = sum(p[1] for p in cart_positions) / n
        return [(x - cx, y - cy) for (x, y) in cart_positions]
    
    best_score = float('inf')
    best_inner = None
    
    # Try hexagonal arrangements of various radii, taking N positions
    for radius in range(1, 6):
        positions = generate_hex_grid(radius)
        if len(positions) >= N:
            # Try taking the first N positions
            arr = positions[:N]
            cart = [axial_to_cartesian(q, r) for (q, r) in arr]
            cart_centered = center_positions(cart)
            S = compute_min_enclosing_hexagon(cart_centered)
            if S < best_score:
                best_score = S
                best_inner = [(x, y, 0.0) for (x, y) in cart_centered]
            
            # Also try a more centered subset: sort by distance from origin
            dists = [(q*q + r*r, q, r) for (q, r) in positions]
            dists.sort()
            arr2 = [(q, r) for (_, q, r) in dists[:N]]
            cart2 = [axial_to_cartesian(q, r) for (q, r) in arr2]
            cart2_centered = center_positions(cart2)
            S2 = compute_min_enclosing_hexagon(cart2_centered)
            if S2 < best_score:
                best_score = S2
                best_inner = [(x, y, 0.0) for (x, y) in cart2_centered]
        
        if time.time() > deadline:
            break
    
    if best_inner is not None:
        return best_inner, (0.0, 0.0), best_score, 0.0
    
    # Fallback
    inner = [(4.0 * k, 0.0, 0.0) for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
