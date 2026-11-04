"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    d = math.sqrt(3)  # center-to-center distance for same-orientation unit hexagons
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    def hex_to_cart(q, r):
        x = d * (q + r / 2)
        y = d * (math.sqrt(3) / 2) * r
        return (x, y)
    
    def evaluate(cart):
        nonlocal best_side, best_inner, best_center, best_angle
        xs = [p[0] for p in cart]
        ys = [p[1] for p in cart]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        total_w = x_max - x_min + 2
        total_h = y_max - y_min + math.sqrt(3)
        
        side = max(total_w / 2, total_h / math.sqrt(3))
        
        if side < best_side:
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            inner = [[x - cx, y - cy, 0.0] for x, y in cart]
            best_side = side
            best_inner = inner
            best_center = (0.0, 0.0)
            best_angle = 0.0
    
    # Try hexagonal clusters
    for r_max in range(1, 8):
        positions = []
        for q in range(-r_max, r_max + 1):
            for r in range(-r_max, r_max + 1):
                if max(abs(q), abs(r), abs(-q - r)) <= r_max:
                    positions.append((q, r))
        
        if len(positions) >= N:
            positions_sorted = sorted(positions, key=lambda p: p[0]**2 + p[1]**2 + p[0]*p[1])
            subset = positions_sorted[:N]
            cart = [hex_to_cart(q, r) for q, r in subset]
            evaluate(cart)
    
    # Try linear arrangements
    for start in range(-N, N):
        cart = [(d * (start + k), 0.0) for k in range(N)]
        evaluate(cart)
    
    # Try two-row arrangements
    for row1_count in range(1, N):
        row2_count = N - row1_count
        cart = []
        for k in range(row1_count):
            cart.append((d * k, 0.0))
        for k in range(row2_count):
            cart.append((d * (k + 0.5), d * math.sqrt(3) / 2))
        evaluate(cart)
    
    # Try three-row arrangements
    for row1_count in range(1, N - 1):
        for row2_count in range(1, N - row1_count):
            row3_count = N - row1_count - row2_count
            if row3_count < 1:
                continue
            cart = []
            for k in range(row1_count):
                cart.append((d * k, 0.0))
            for k in range(row2_count):
                cart.append((d * (k + 0.5), d * math.sqrt(3) / 2))
            for k in range(row3_count):
                cart.append((d * k, d * math.sqrt(3)))
            evaluate(cart)
    
    if best_inner is None:
        inner = [[k * d, 0.0, 0.0] for k in range(N)]
        return inner, (0.0, 0.0), N * d + 2, 0.0
    
    return best_inner, best_center, best_side, best_angle
# EVOLVE-BLOCK-END
