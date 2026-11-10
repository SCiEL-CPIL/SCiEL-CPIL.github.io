"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 120

# Precompute edge indices for fast lookup
# edge_index[i][j] = index of edge (i,j) in the flat array, for i < j
def _build_edge_index():
    idx = [[-1] * N for _ in range(N)]
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            idx[i][j] = k
            idx[j][i] = k
            k += 1
    return idx

EDGE_IDX = _build_edge_index()

# Precompute for each edge (i,j), the list of (x, y) pairs such that
# i-j-x-y-i is a 4-cycle, i.e., x,y are distinct, not i,j, and x != y
# The cycle edges are: (i,j), (j,x), (x,y), (y,i)
def _build_cycle_partners():
    partners = {}
    for i in range(N):
        for j in range(i + 1, N):
            ej = EDGE_IDX[i][j]
            plist = []
            for x in range(N):
                if x == i or x == j:
                    continue
                for y in range(N):
                    if y == i or y == j or y == x:
                        continue
                    # Cycle: i-j-x-y-i
                    # Edges: (i,j), (j,x), (x,y), (y,i)
                    e_jx = EDGE_IDX[j][x]
                    e_xy = EDGE_IDX[x][y]
                    e_yi = EDGE_IDX[y][i]
                    plist.append((e_jx, e_xy, e_yi))
            partners[ej] = plist
    return partners

CYCLE_PARTNERS = _build_cycle_partners()

def _is_valid_color(colors, edge_idx, new_color):
    """Check if assigning new_color to edge_idx creates any rainbow 4-cycle."""
    i = -1
    j = -1
    # Find which edge this is
    for a in range(N):
        for b in range(a + 1, N):
            if EDGE_IDX[a][b] == edge_idx:
                i, j = a, b
                break
        if i != -1:
            break
    
    for (e_jx, e_xy, e_yi) in CYCLE_PARTNERS[edge_idx]:
        c1 = colors[e_jx]
        c2 = colors[e_xy]
        c3 = colors[e_yi]
        if c1 >= 0 and c2 >= 0 and c3 >= 0:
            # Check if all four colors are distinct
            if len({new_color, c1, c2, c3}) == 4:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline, 10s margin
    
    # Greedy: assign edges in lex order, pick lowest valid color
    colors = [-1] * NUM_EDGES
    used_colors = set()
    
    for i in range(N):
        for j in range(i + 1, N):
            ej = EDGE_IDX[i][j]
            if time.time() > deadline:
                # Fill remaining with 0 (safe fallback)
                for k in range(NUM_EDGES):
                    if colors[k] == -1:
                        colors[k] = 0
                return colors
            
            # Try colors from 0 upward
            found = False
            for c in range(MAX_COLOR):
                if _is_valid_color(colors, ej, c):
                    colors[ej] = c
                    used_colors.add(c)
                    found = True
                    break
            if not found:
                # Should not happen with 120 colors, but fallback
                colors[ej] = 0
                used_colors.add(0)
    
    # Verify and return
    for k in range(NUM_EDGES):
        if colors[k] == -1:
            colors[k] = 0
    
    return colors
