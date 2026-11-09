"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# Precompute all 4-cycles (all 4-vertex subsets and their 3 cycles)
# Each cycle is a tuple of 4 edge indices
# Edge index for (i,j) with i<j: i*N - i*(i+1)//2 + j - i - 1
# Actually simpler: edge_idx[i][j]

def build_edge_index():
    idx = [[-1]*N for _ in range(N)]
    counter = 0
    for i in range(N):
        for j in range(i+1, N):
            idx[i][j] = counter
            idx[j][i] = counter
            counter += 1
    return idx

EDGE_IDX = build_edge_index()

# Build all 4-cycles: for each 4-set {a,b,c,d} (a<b<c<d),
# the three cycles are:
# (a,b,c,d): edges (a,b), (b,c), (c,d), (d,a)
# (a,b,d,c): edges (a,b), (b,d), (d,c), (c,a)
# (a,c,b,d): edges (a,c), (c,b), (b,d), (d,a)
def build_cycles():
    cycles = []
    for a in range(N-3):
        for b in range(a+1, N-2):
            for c in range(b+1, N-1):
                for d in range(c+1, N):
                    # Cycle 1: a-b-c-d-a
                    cycles.append((EDGE_IDX[a][b], EDGE_IDX[b][c], EDGE_IDX[c][d], EDGE_IDX[d][a]))
                    # Cycle 2: a-b-d-c-a
                    cycles.append((EDGE_IDX[a][b], EDGE_IDX[b][d], EDGE_IDX[d][c], EDGE_IDX[c][a]))
                    # Cycle 3: a-c-b-d-a
                    cycles.append((EDGE_IDX[a][c], EDGE_IDX[c][b], EDGE_IDX[b][d], EDGE_IDX[d][a]))
    return cycles

CYCLES = build_cycles()
NUM_CYCLES = len(CYCLES)  # C(16,4)*3 = 1820*3 = 5460

# Precompute for each edge, which cycles it belongs to
EDGE_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci, cycle in enumerate(CYCLES):
    for e in cycle:
        EDGE_CYCLES[e].append(ci)

def can_take_color(colors, edge, new_color):
    """Check if assigning new_color to edge keeps all cycles valid."""
    for ci in EDGE_CYCLES[edge]:
        cycle = CYCLES[ci]
        # Count distinct colors in this cycle after change
        cnt = 0
        seen = set()
        for e in cycle:
            c = new_color if e == edge else colors[e]
            if c not in seen:
                seen.add(c)
                cnt += 1
        if cnt > 3:
            return False
    return True

def count_distinct_colors(colors):
    return len(set(colors))

def local_search(colors, deadline):
    """Try to increase distinct colors by recoloring edges."""
    colors = list(colors)
    used_colors = set(colors)
    next_color = 120  # We can use colors 0..119
    # Actually, we should track which colors are available
    # Colors 0..119 are available
    
    # Find next available color
    all_colors = set(range(120))
    available = all_colors - used_colors
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try each edge in random order
        edges = list(range(NUM_EDGES))
        random.shuffle(edges)
        for e in edges:
            if time.time() >= deadline:
                break
            # Try to assign a new color from available
            new_colors = list(available)
            random.shuffle(new_colors)
            for nc in new_colors:
                if can_take_color(colors, e, nc):
                    colors[e] = nc
                    used_colors.add(nc)
                    available.remove(nc)
                    improved = True
                    break
        # Also try to recolor edges that currently have colors that are "wasted"
        # If a color is used by only one edge, and we can recolor that edge to free up the color,
        # but that doesn't help unless we then use the freed color elsewhere
        # For now, just keep adding new colors
    return colors

def construct():
    deadline = time.time() + 50  # 50 second deadline for safety
    
    best_colors = [0] * NUM_EDGES
    best_score = 0
    
    # Try multiple random permutations
    for perm_iter in range(20):
        if time.time() >= deadline:
            break
        
        perm = list(range(N))
        random.shuffle(perm)
        
        # Base coloring: color of edge (i,j) = perm[i] if perm[i] < perm[j] else perm[j]
        # Actually, let's use: color = min(perm[i], perm[j])
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i+1, N):
                e = EDGE_IDX[i][j]
                colors[e] = min(perm[i], perm[j])
        
        # Verify base coloring is valid
        valid = True
        for cycle in CYCLES:
            cs = set(colors[e] for e in cycle)
            if len(cs) > 3:
                valid = False
                break
        
        if not valid:
            continue
        
        # Run local search
        improved = local_search(colors, deadline)
        score = count_distinct_colors(improved)
        
        if score > best_score:
            best_score = score
            best_colors = improved
    
    # Final check
    if best_score == 0:
        # Fallback: all zeros
        return [0] * NUM_EDGES
    
    # Verify final result
    for cycle in CYCLES:
        cs = set(best_colors[e] for e in cycle)
        if len(cs) > 3:
            # Should not happen, but fallback
            return [0] * NUM_EDGES
    
    return best_colors

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
