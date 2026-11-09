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
    """Check if assigning new_color to edge is valid (no 4-cycle gets 4 distinct colors)."""
    for ci in EDGE_CYCLES[edge]:
        cycle = CYCLES[ci]
        # Count distinct colors in this cycle with the new assignment
        c0 = new_color if cycle[0] == edge else colors[cycle[0]]
        c1 = new_color if cycle[1] == edge else colors[cycle[1]]
        c2 = new_color if cycle[2] == edge else colors[cycle[2]]
        c3 = new_color if cycle[3] == edge else colors[cycle[3]]
        # Check if all four are distinct
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
    return True

def local_search(colors, deadline):
    colors = list(colors)
    used_colors = set(colors)
    best_score = len(used_colors)
    best_colors = list(colors)
    
    no_improve_count = 0
    max_no_improve = 500
    consecutive_no_improve = 0
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Kick: randomly recolor several edges to escape local optimum
            num_kicks = random.randint(3, 8)
            for _ in range(num_kicks):
                e = random.randint(0, NUM_EDGES - 1)
                # Try to find a valid color, prefer unused colors first
                all_colors = list(range(120))
                random.shuffle(all_colors)
                for nc in all_colors:
                    if can_take_color(colors, e, nc):
                        colors[e] = nc
                        break
            
            used_colors = set(colors)
            no_improve_count = 0
            max_no_improve = random.randint(300, 1500)
            consecutive_no_improve = 0
            continue

        # Try to improve: find an edge that can take an unused color
        improved = False
        edges = list(range(NUM_EDGES))
        random.shuffle(edges)
        
        for e in edges:
            if time.time() >= deadline:
                break
            
            # Try unused colors first (shuffled for diversity)
            unused = [c for c in range(120) if c not in used_colors]
            random.shuffle(unused)
            
            found = False
            for nc in unused:
                if can_take_color(colors, e, nc):
                    colors[e] = nc
                    used_colors.add(nc)
                    improved = True
                    found = True
                    break
            
            if found:
                no_improve_count = 0
                consecutive_no_improve = 0
                if len(used_colors) > best_score:
                    best_score = len(used_colors)
                    best_colors = list(colors)
            else:
                no_improve_count += 1
        
        if not improved:
            consecutive_no_improve += 1
            # If we've had many consecutive failed passes, do a more aggressive kick
            if consecutive_no_improve >= 5:
                # Aggressive kick: recolor more edges, including some used-color changes
                num_aggressive = random.randint(5, 12)
                for _ in range(num_aggressive):
                    e = random.randint(0, NUM_EDGES - 1)
                    all_colors = list(range(120))
                    random.shuffle(all_colors)
                    for nc in all_colors:
                        if can_take_color(colors, e, nc):
                            colors[e] = nc
                            break
                used_colors = set(colors)
                no_improve_count = 0
                consecutive_no_improve = 0

    return best_colors, best_score

def construct():
    deadline = time.time() + 55  # 55s safety
    
    best_colors = [0] * NUM_EDGES
    best_score = 0
    
    # Generate diverse starting colorings
    strategies = [
        "min_perm", "max_perm", "random", "structured"
    ]
    
    for strat_iter in range(60):
        if time.time() >= deadline:
            break
        
        strategy = strategies[strat_iter % len(strategies)]
        
        if strategy == "min_perm":
            perm = list(range(N))
            random.shuffle(perm)
            colors = [0] * NUM_EDGES
            for i in range(N):
                for j in range(i+1, N):
                    e = EDGE_IDX[i][j]
                    colors[e] = min(perm[i], perm[j])
        elif strategy == "max_perm":
            perm = list(range(N))
            random.shuffle(perm)
            colors = [0] * NUM_EDGES
            for i in range(N):
                for j in range(i+1, N):
                    e = EDGE_IDX[i][j]
                    colors[e] = max(perm[i], perm[j])
        elif strategy == "random":
            colors = [random.randint(0, 119) for _ in range(NUM_EDGES)]
        else:  # structured
            # Use a pattern based on vertex indices
            colors = [0] * NUM_EDGES
            for i in range(N):
                for j in range(i+1, N):
                    e = EDGE_IDX[i][j]
                    colors[e] = (i * 7 + j * 3) % 120
        
        if not is_valid(colors):
            # Try to fix by recoloring invalid edges
            fixed = False
            for e in range(NUM_EDGES):
                if time.time() >= deadline:
                    break
                for nc in range(120):
                    if can_take_color(colors, e, nc):
                        colors[e] = nc
                        fixed = True
                        break
            if not is_valid(colors):
                continue
        
        # Run local search
        # Use more time for later iterations
        if strat_iter < 45:
            ls_deadline = time.time() + 0.8
        else:
            ls_deadline = deadline
            
        improved_colors, score = local_search(colors, ls_deadline)
        
        if score > best_score:
            best_score = score
            best_colors = improved_colors
            
        # Early stop if we reach a high score
        if best_score >= 30:
            break

    # Final verification
    if not is_valid(best_colors):
        return [0] * NUM_EDGES
        
    return best_colors

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
