import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# Precompute edge index
def edge_idx(i, j):
    if i > j:
        i, j = j, i
    # Index for pair (i, j) where i < j
    # Number of edges from vertex 0 to j: j
    # Number of edges from vertex 1 to j: j-1
    # ...
    # For vertex i, edges to vertices i+1..j-1: (j-1) - i = j-1-i
    # Total before (i,j): sum_{k=0}^{i-1} (N-1-k) + (j-1-i)
    # = i*(N-1) - i*(i-1)//2 + j-1-i
    # Let's use a simpler mapping
    return i * N - i*(i+1)//2 + (j - i - 1)

# Precompute all 4-cycles
def precompute_cycles():
    cycles = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Three 4-cycles on these 4 vertices
        # Cycle 1: a-b-c-d-a
        cycles.append([(a,b), (b,c), (c,d), (d,a)])
        # Cycle 2: a-b-d-c-a
        cycles.append([(a,b), (b,d), (d,c), (c,a)])
        # Cycle 3: a-c-b-d-a
        cycles.append([(a,c), (c,b), (b,d), (d,a)])
    return cycles

CYCLES = precompute_cycles()

def count_colors(colors):
    return len(set(colors))

def is_valid(colors):
    for cycle in CYCLES:
        edge_colors = []
        for (u, v) in cycle:
            idx = edge_idx(u, v)
            edge_colors.append(colors[idx])
        if len(set(edge_colors)) == 4:
            return False
    return True

def check_edge_change(colors, edge_i, edge_j, new_color):
    """Check if changing edge (i,j) to new_color creates any rainbow 4-cycle."""
    # Only need to check cycles that include this edge
    # For efficiency, we can check all cycles but that's slow
    # Let's just check all cycles for now, but we can optimize later
    old_color = colors[edge_idx(i, j)] if i < j else colors[edge_idx(j, i)]
    idx = edge_idx(i, j)
    colors[idx] = new_color
    
    valid = True
    for cycle in CYCLES:
        edge_colors = []
        for (u, v) in cycle:
            eidx = edge_idx(u, v)
            edge_colors.append(colors[eidx])
        if len(set(edge_colors)) == 4:
            valid = False
            break
    
    colors[idx] = old_color
    return valid

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    best_colors = [0] * NUM_EDGES
    best_count = 0
    
    # Try multiple random seeds
    for seed in range(1000):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        
        # Start with all zeros (valid)
        colors = [0] * NUM_EDGES
        
        # Try to introduce new colors greedily
        used_colors = {0}
        
        # Order edges randomly
        edge_order = list(range(NUM_EDGES))
        random.shuffle(edge_order)
        
        # For each edge, try to assign a new color
        for _ in range(50):  # Multiple passes
            if time.time() > deadline:
                break
            
            edge_order = list(range(NUM_EDGES))
            random.shuffle(edge_order)
            
            for eidx in edge_order:
                if time.time() > deadline:
                    break
                
                # Find which edge this index corresponds to
                # We need to reverse the edge_idx mapping
                # Let's precompute this
                pass
        
        # Simpler approach: just try random edge-color changes
        colors = [0] * NUM_EDGES
        used = {0}
        
        for iteration in range(5000):
            if time.time() > deadline:
                break
            
            # Pick a random edge
            eidx = random.randint(0, NUM_EDGES - 1)
            
            # Try to assign a new color
            new_color = random.randint(0, 119)
            if new_color in used and random.random() < 0.7:
                continue  # Prefer new colors
            
            old_color = colors[eidx]
            colors[eidx] = new_color
            
            if is_valid(colors):
                used.add(new_color)
            else:
                colors[eidx] = old_color
        
        cur_count = count_colors(colors)
        if cur_count > best_count:
            best_count = cur_count
            best_colors = colors[:]
    
    return best_colors

# Precompute edge index to (i,j) mapping
EDGE_TO_PAIR = []
for i in range(N):
    for j in range(i+1, N):
        EDGE_TO_PAIR.append((i, j))

# Rebuild with proper edge index handling
def edge_idx(i, j):
    if i > j:
        i, j = j, i
    # Build lookup
    return EDGE_TO_PAIR.index((i, j))

# This is too slow, let's use a dictionary
EDGE_LOOKUP = {}
for idx, (i, j) in enumerate(EDGE_TO_PAIR):
    EDGE_LOOKUP[(i, j)] = idx
    EDGE_LOOKUP[(j, i)] = idx

def edge_idx(i, j):
    return EDGE_LOOKUP[(i, j)]

# Rebuild cycles with indices
CYCLES_IDX = []
for cycle in CYCLES:
    CYCLES_IDX.append([edge_idx(u, v) for (u, v) in cycle])

def is_valid(colors):
    for cycle in CYCLES_IDX:
        if len(set(colors[e] for e in cycle)) == 4:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    best_colors = [0] * NUM_EDGES
    best_count = 0
    
    for seed in range(200):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        colors = [0] * NUM_EDGES
        used = {0}
        
        for iteration in range(3000):
            if time.time() > deadline:
                break
            
            eidx = random.randint(0, NUM_EDGES - 1)
            new_color = random.randint(0, 119)
            
            if new_color in used and random.random() < 0.8:
                continue
            
            old_color = colors[eidx]
            colors[eidx] = new_color
            
            if is_valid(colors):
                used.add(new_color)
            else:
                colors[eidx] = old_color
        
        cur_count = count_colors(colors)
        if cur_count > best_count:
            best_count = cur_count
            best_colors = colors[:]
    
    return best_colors
