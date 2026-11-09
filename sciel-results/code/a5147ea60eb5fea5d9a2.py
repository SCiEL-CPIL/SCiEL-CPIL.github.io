import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

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

def build_cycles():
    cycles = []
    for a in range(N-3):
        for b in range(a+1, N-2):
            for c in range(b+1, N-1):
                for d in range(c+1, N):
                    cycles.append((EDGE_IDX[a][b], EDGE_IDX[b][c], EDGE_IDX[c][d], EDGE_IDX[d][a]))
                    cycles.append((EDGE_IDX[a][b], EDGE_IDX[b][d], EDGE_IDX[d][c], EDGE_IDX[c][a]))
                    cycles.append((EDGE_IDX[a][c], EDGE_IDX[c][b], EDGE_IDX[b][d], EDGE_IDX[d][a]))
    return cycles

CYCLES = build_cycles()
NUM_CYCLES = len(CYCLES)

EDGE_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci, cycle in enumerate(CYCLES):
    for e in cycle:
        EDGE_CYCLES[e].append(ci)

def is_valid(colors):
    for cycle in CYCLES:
        if len(set(colors[e] for e in cycle)) > 3:
            return False
    return True

def can_take_color(colors, edge, new_color):
    for ci in EDGE_CYCLES[edge]:
        cycle = CYCLES[ci]
        seen = set()
        for e in cycle:
            c = new_color if e == edge else colors[e]
            if c in seen:
                continue
            seen.add(c)
            if len(seen) > 3:
                return False
    return True

def local_search(colors, deadline):
    colors = list(colors)
    used_colors = set(colors)
    best_score = len(used_colors)
    best_colors = list(colors)
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Kick: randomly recolor a few edges to break out of local optima
            for _ in range(3):
                e = random.randint(0, NUM_EDGES - 1)
                # Try random colors until we find a valid one
                for _ in range(30):
                    nc = random.randint(0, 119)
                    if can_take_color(colors, e, nc):
                        colors[e] = nc
                        break
            
            used_colors = set(colors)
            no_improve_count = 0
            max_no_improve = random.randint(500, 2000)
            continue

        # Try to improve by adding new colors
        improved = False
        edges = list(range(NUM_EDGES))
        random.shuffle(edges)
        
        for e in edges:
            if time.time() >= deadline:
                break
            
            # Prioritize unused colors
            all_colors = set(range(120))
            available = list(all_colors - used_colors)
            random.shuffle(available)
            
            for nc in available:
                if can_take_color(colors, e, nc):
                    colors[e] = nc
                    used_colors.add(nc)
                    improved = True
                    break
            
            if improved:
                no_improve_count = 0
                if len(used_colors) > best_score:
                    best_score = len(used_colors)
                    best_colors = list(colors)
            else:
                no_improve_count += 1
        
        if not improved:
            no_improve_count += 50

    return best_colors, best_score

def construct():
    deadline = time.time() + 55  # 55s safety margin
    
    best_colors = [0] * NUM_EDGES
    best_score = 0
    
    # Generate diverse initial colorings using different strategies
    for restart in range(100):
        if time.time() >= deadline:
            break
        
        # Different initialization strategies
        strategy = restart % 4
        
        if strategy == 0:
            # Random permutation-based coloring
            perm = list(range(N))
            random.shuffle(perm)
            colors = [0] * NUM_EDGES
            for i in range(N):
                for j in range(i+1, N):
                    e = EDGE_IDX[i][j]
                    colors[e] = min(perm[i], perm[j])
        
        elif strategy == 1:
            # Block-based coloring
            colors = [0] * NUM_EDGES
            for i in range(N):
                for j in range(i+1, N):
                    e = EDGE_IDX[i][j]
                    colors[e] = (i // 4) * 10 + (j // 4)
        
        elif strategy == 2:
            # Random sparse coloring
            colors = [0] * NUM_EDGES
            for e in range(NUM_EDGES):
                if random.random() < 0.3:
                    colors[e] = random.randint(0, 119)
        
        else:
            # Structured coloring based on distance
            colors = [0] * NUM_EDGES
            for i in range(N):
                for j in range(i+1, N):
                    e = EDGE_IDX[i][j]
                    dist = j - i
                    colors[e] = (dist * 7) % 120
        
        # Verify and repair if needed
        if not is_valid(colors):
            # Try to fix invalid cycles
            fixed = False
            for _ in range(10):
                valid = True
                for cycle in CYCLES:
                    if len(set(colors[e] for e in cycle)) > 3:
                        valid = False
                        # Pick one edge to recolor
                        e = random.choice(cycle)
                        for _ in range(20):
                            nc = random.randint(0, 119)
                            if can_take_color(colors, e, nc):
                                colors[e] = nc
                                break
                        break
                if valid:
                    fixed = True
                    break
            if not is_valid(colors):
                continue
        
        # Run local search
        if restart < 80:
            ls_deadline = time.time() + 0.3
        else:
            ls_deadline = deadline
        
        improved_colors, score = local_search(colors, ls_deadline)
        
        if score > best_score:
            best_score = score
            best_colors = improved_colors
        
        if best_score >= 20:
            break

    # Final verification
    if not is_valid(best_colors):
        return [0] * NUM_EDGES
        
    return best_colors

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
