import time
import random

DEADLINE = 115  # seconds, safety margin before 150s kill

def _time_left():
    return time.time() - _START_TIME < DEADLINE

_START_TIME = time.time()

def construct():
    global _START_TIME
    _START_TIME = time.time()
    random.seed(42)

    # ---- Build the n=32 seed colouring (Greenwood-Gleason doubled) ----
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)

    n_seed = 32
    colors_seed = []
    for i in range(n_seed):
        for j in range(i + 1, n_seed):
            if i // 16 == j // 16:
                colors_seed.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                colors_seed.append(3)

    def _validate(colors, n):
        """Check no monochromatic triangle."""
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        for c in range(4):
            for i in range(n):
                neighbors = adj[c][i]
                while neighbors:
                    lsb = neighbors & (-neighbors)
                    j = lsb.bit_length() - 1
                    neighbors ^= lsb
                    common = adj[c][i] & adj[c][j]
                    if common:
                        return False
        return True

    if not _validate(colors_seed, n_seed):
        return {'n': 1, 'colors': []}

    best_n = n_seed
    best_colors = colors_seed[:]

    def extend_by_one(current_colors, current_n):
        """Try to extend coloring from current_n to current_n+1."""
        if not _time_left():
            return None
        
        new_n = current_n + 1
        if new_n > 62:
            return None
        
        # New vertex is index new_n-1
        # Edges: (new_n-1, i) for i in 0..new_n-2
        # We need to assign colors to these new_n-1 edges
        
        # Build lookup for existing edge colors
        edge_color = [[-1] * current_n for _ in range(current_n)]
        idx = 0
        for i in range(current_n):
            for j in range(i + 1, current_n):
                edge_color[i][j] = current_colors[idx]
                edge_color[j][i] = current_colors[idx]
                idx += 1
        
        def compute_cost(x):
            cost = 0
            for i in range(current_n):
                for j in range(i + 1, current_n):
                    if x[i] == x[j]:
                        c = x[i]
                        if edge_color[i][j] == c:
                            cost += 1
            return cost
        
        def compute_cost_delta(x, i, old_c, new_c):
            delta = 0
            for j in range(current_n):
                if j == i:
                    continue
                if old_c == x[j] and edge_color[i][j] == old_c:
                    delta -= 1
                if new_c == x[j] and edge_color[i][j] == new_c:
                    delta += 1
            return delta
        
        for restart in range(50):
            if not _time_left():
                return None
            
            x = [random.randint(0, 3) for _ in range(current_n)]
            current_cost = compute_cost(x)
            
            if current_cost == 0:
                new_colors = current_colors[:]
                for i in range(current_n):
                    new_colors.append(x[i])
                return new_colors
            
            for step in range(1000):
                if not _time_left():
                    return None
                
                if current_cost == 0:
                    new_colors = current_colors[:]
                    for i in range(current_n):
                        new_colors.append(x[i])
                    return new_colors
                
                i = random.randint(0, current_n - 1)
                old_c = x[i]
                new_c = random.randint(0, 3)
                if new_c == old_c:
                    continue
                
                delta = compute_cost_delta(x, i, old_c, new_c)
                
                if delta <= 0:
                    x[i] = new_c
                    current_cost += delta
                else:
                    if random.random() < 0.05:
                        x[i] = new_c
                        current_cost += delta
        
        return None
    
    current_colors = colors_seed[:]
    current_n = n_seed
    
    for target in range(current_n + 1, 51):
        if not _time_left():
            break
        
        new_colors = extend_by_one(current_colors, current_n)
        if new_colors is not None:
            if _validate(new_colors, current_n + 1):
                current_colors = new_colors
                current_n += 1
                best_n = current_n
                best_colors = current_colors[:]
            else:
                break
        else:
            break
    
    return {'n': best_n, 'colors': best_colors}
