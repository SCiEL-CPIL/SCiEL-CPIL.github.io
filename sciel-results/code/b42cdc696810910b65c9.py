"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_sum = -np.inf
    best_result = None

    def evaluate(xs, ys, rs, w, h):
        """Check validity and return sum of radii if valid."""
        if len(xs) != N:
            return -np.inf
        for i in range(N):
            if rs[i] <= 0 or xs[i] < rs[i] - 1e-8 or xs[i] > w - rs[i] + 1e-8:
                return -np.inf
            if ys[i] < rs[i] - 1e-8 or ys[i] > h - rs[i] + 1e-8:
                return -np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                if d < rs[i] + rs[j] - 1e-8:
                    return -np.inf
        return np.sum(rs)

    def optimize_pack(w, h, seed=42, max_iter=3000):
        """Pack N circles in w x h rectangle."""
        rng = np.random.RandomState(seed)
        
        # Grid init
        cols = 5
        rows = int(np.ceil(N / cols))
        xs = np.zeros(N)
        ys = np.zeros(N)
        rs = np.zeros(N)
        
        for i in range(N):
            c = i % cols
            r = i // cols
            xs[i] = (c + 0.5) * w / cols
            ys[i] = (r + 0.5) * h / rows
            rs[i] = 0.08
        
        # Iterative: grow radii, resolve overlaps
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Grow radii slightly
            for i in range(N):
                max_r = min(xs[i], w - xs[i], ys[i], h - ys[i])
                if rs[i] < max_r * 0.999:
                    rs[i] = min(rs[i] * 1.0005, max_r * 0.999)
            
            # Resolve overlaps - multiple passes
            for _ in range(30):
                moved = False
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = xs[j] - xs[i]
                        dy = ys[j] - ys[i]
                        dist = np.sqrt(dx*dx + dy*dy)
                        min_d = rs[i] + rs[j]
                        if dist < min_d - 1e-10:
                            if dist < 1e-14:
                                dx, dy = 0.001, 0.001
                                dist = np.sqrt(dx*dx + dy*dy)
                            push = (min_d - dist) / 2.0 + 1e-6
                            nx, ny = dx/dist, dy/dist
                            xs[i] -= nx * push
                            ys[i] -= ny * push
                            xs[j] += nx * push
                            ys[j] += ny * push
                            moved = True
                
                # Clamp
                for i in range(N):
                    xs[i] = np.clip(xs[i], rs[i], w - rs[i])
                    ys[i] = np.clip(ys[i], rs[i], h - ys[i])
                
                if not moved:
                    break
        
        # Final: maximize radii given positions
        for i in range(N):
            rs[i] = min(xs[i], w - xs[i], ys[i], h - ys[i])
        
        # Verify and fix
        for _ in range(100):
            overlap_found = False
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                    if d < rs[i] + rs[j] - 1e-10:
                        # Reduce smaller radius
                        if rs[i] < rs[j]:
                            rs[i] = max(0.01, (d - 1e-6) / 2.0)
                        else:
                            rs[j] = max(0.01, (d - 1e-6) / 2.0)
                        overlap_found = True
            if not overlap_found:
                break
        
        return xs, ys, rs

    # Try various aspect ratios
    aspects = [1.0, 0.9, 1.1, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0, 0.4, 2.5, 0.4, 3.0]
    
    for ar in aspects:
        if time.time() > deadline - 5.0:
            break
        
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)
        
        for seed in [42, 123, 7]:
            if time.time() > deadline - 3.0:
                break
            
            xs, ys, rs = optimize_pack(w, h, seed=seed, max_iter=2000)
            
            s = evaluate(xs, ys, rs, w, h)
            if s > best_sum:
                best_sum = s
                best_result = np.column_stack([xs, ys, rs])
    
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
