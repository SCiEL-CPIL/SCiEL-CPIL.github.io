import time
import random
import numpy as np

def _grid_adj(a1, b1, a2, b2):
    da = abs(a1 - a2); da = min(da, 5 - da)
    db = abs(b1 - b2); db = min(db, 10 - db)
    if a1 == a2: return db == 1
    if b1 == b2: return da == 1
    return False

def _neighbors_grid(N):
    adj = [[] for _ in range(N)]
    for a in range(5):
        for b in range(10):
            u = a * 10 + b
            for a2 in range(5):
                for b2 in range(10):
                    if (a, b) != (a2, b2) and _grid_adj(a, b, a2, b2):
                        adj[u].append(a2 * 10 + b2)
    return adj

def _proper_4color(adj, N, deadline, seed=0):
    rng = random.Random(seed)
    order = list(range(N))
    order.sort(key=lambda v: -len(adj[v]))
    # Randomize tie-breakers
    for i in range(N-1, 0, -1):
        j = rng.randint(0, i)
        order[i], order[j] = order[j], order[i]
        
    color = [-1] * N
    
    def assign(pos):
        if time.time() > deadline:
            return False
        if pos == N:
            return True
        v = order[pos]
        used = set()
        for nb in adj[v]:
            if color[nb] != -1:
                used.add(color[nb])
        # Try colours in random order to escape local minima
        colors_to_try = list(range(4))
        rng.shuffle(colors_to_try)
        for c in colors_to_try:
            if c not in used:
                color[v] = c
                if assign(pos + 1):
                    return True
                color[v] = -1
        return False
        
    if assign(0):
        return color
    
    # Fallback: local search with conflict minimization
    return _local_search_4color(adj, N, deadline, seed + 1000)

def _local_search_4color(adj, N, deadline, seed=0):
    """Minimize monochromatic edges via local search."""
    rng = random.Random(seed)
    
    # Initialize with random coloring
    color = [rng.randint(0, 3) for _ in range(N)]
    
    # Precompute adjacency as sets for fast lookup
    adj_set = [set(nei) for nei in adj]
    
    best_conflicts = float('inf')
    best_color = None
    no_improve = 0
    
    while time.time() < deadline and no_improve < 500:
        # Count conflicts (monochromatic edges)
        conflicts = 0
        for i in range(N):
            for j in adj_set[i]:
                if j > i and color[i] == color[j]:
                    conflicts += 1
        
        if conflicts == 0:
            return color
        
        if conflicts < best_conflicts:
            best_conflicts = conflicts
            best_color = color[:]
            no_improve = 0
        else:
            no_improve += 1
        
        # Find vertex with most conflicts and try to recolor
        vertex_conflicts = [0] * N
        for i in range(N):
            for j in adj_set[i]:
                if j > i and color[i] == color[j]:
                    vertex_conflicts[i] += 1
                    vertex_conflicts[j] += 1
        
        # Pick vertex with most conflicts
        v = max(range(N), key=lambda x: vertex_conflicts[x])
        
        # Try each color, pick the one that reduces conflicts most
        best_c = color[v]
        best_delta = 0
        for c in range(4):
            if c == color[v]:
                continue
            delta = 0
            for nb in adj_set[v]:
                if color[nb] == c:
                    delta -= 1  # Creates new conflict
                elif color[nb] == color[v]:
                    delta += 1  # Removes existing conflict
            if delta > best_delta:
                best_delta = delta
                best_c = c
        
        if best_delta > 0:
            color[v] = best_c
        else:
            # No improvement possible, random perturbation
            v = rng.randint(0, N-1)
            color[v] = rng.randint(0, 3)
    
    return best_color if best_conflicts == 0 else None

def _build():
    N = 50
    adj = _neighbors_grid(N)
    # Use sets for faster membership check
    adj_set = [set(nei) for nei in adj]
    comp_adj = []
    for i in range(N):
        comp_nei = [j for j in range(N) if j != i and j not in adj_set[i]]
        comp_adj.append(comp_nei)
        
    deadline = time.time() + 100.0 # Leave time for verification
    
    # Try multiple random seeds for robustness
    for seed in range(50):
        if time.time() > deadline:
            break
        cg = _proper_4color(adj, N, deadline, seed)
        if cg is None:
            continue
        cc = _proper_4color(comp_adj, N, deadline, seed + 100)
        if cc is None:
            continue
            
        # Construct edge colours
        colors = []
        for i in range(N):
            for j in range(i + 1, N):
                if j in adj_set[i]:
                    colors.append(cg[i])
                else:
                    colors.append(cc[i])
        
        # Verify
        if _verify(N, colors):
            return N, colors
            
    return None

def _verify(n, colors):
    """Vectorized verification using numpy for speed."""
    N = n
    num_edges = N * (N - 1) // 2
    if len(colors) != num_edges:
        return False
    
    # Build color matrix
    col = np.zeros((N, N), dtype=np.int8)
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            col[i, j] = col[j, i] = colors[k]
            k += 1
    
    # Check for monochromatic triangles using vectorized approach
    # For each pair (i,j), check if any k > j has col[i,k] == col[j,k] == col[i,j]
    for i in range(N):
        for j in range(i + 1, N):
            c = col[i, j]
            if c == 0:
                continue
            # Check if there exists k > j with col[i,k] == c and col[j,k] == c
            mask = (col[i, j+1:] == c) & (col[j, j+1:] == c)
            if np.any(mask):
                return False
    return True

def _seed32():
    # Fallback to 32 vertices if 50 fails (should not happen)
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
    c = []
    for i in range(32):
        for j in range(i + 1, 32):
            c.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': c}

def construct():
    deadline = time.time() + 140.0
    r = _build()
    if r is not None and time.time() < deadline:
        n, colors = r
        return {'n': n, 'colors': colors}
    return _seed32()
