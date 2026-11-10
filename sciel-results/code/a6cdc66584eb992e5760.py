import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

# Precompute edges in lexicographic order
EDGES = [(i, j) for i in range(N) for j in range(i + 1, N)]
EDGE_IDX = {}
for idx, (i, j) in enumerate(EDGES):
    EDGE_IDX[(i, j)] = idx
    EDGE_IDX[(j, i)] = idx

# Precompute all 4-cycles and their edge indices
CYCLES = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                # Cycle 1: a-b-c-d-a
                e1 = EDGE_IDX[(a, b)]
                e2 = EDGE_IDX[(b, c)]
                e3 = EDGE_IDX[(c, d)]
                e4 = EDGE_IDX[(d, a)]
                CYCLES.append((e1, e2, e3, e4))
                # Cycle 2: a-b-d-c-a
                e1 = EDGE_IDX[(a, b)]
                e2 = EDGE_IDX[(b, d)]
                e3 = EDGE_IDX[(d, c)]
                e4 = EDGE_IDX[(c, a)]
                CYCLES.append((e1, e2, e3, e4))
                # Cycle 3: a-c-b-d-a
                e1 = EDGE_IDX[(a, c)]
                e2 = EDGE_IDX[(c, b)]
                e3 = EDGE_IDX[(b, d)]
                e4 = EDGE_IDX[(d, a)]
                CYCLES.append((e1, e2, e3, e4))

NUM_CYCLES = len(CYCLES)  # 5460

# For each edge, which cycles does it belong to?
OTHER_EDGES = [[] for _ in range(NUM_EDGES)]
for cyc in CYCLES:
    e1, e2, e3, e4 = cyc
    OTHER_EDGES[e1].append((e2, e3, e4))
    OTHER_EDGES[e2].append((e1, e3, e4))
    OTHER_EDGES[e3].append((e1, e2, e4))
    OTHER_EDGES[e4].append((e1, e2, e3))


def count_violations(colors):
    """Count number of 4-cycles with 4 distinct edge colors."""
    violations = 0
    for e1, e2, e3, e4 in CYCLES:
        c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
        if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
            violations += 1
    return violations


def check_edge_change(colors, edge_idx, new_color):
    """
    Check if changing colors[edge_idx] to new_color creates a rainbow cycle.
    Returns True if valid, False otherwise.
    """
    if colors[edge_idx] == new_color:
        return True
    
    for o1, o2, o3 in OTHER_EDGES[edge_idx]:
        c1 = colors[o1]
        c2 = colors[o2]
        c3 = colors[o3]
        
        if c1 == c2 or c2 == c3 or c1 == c3:
            continue
            
        if new_color == c1 or new_color == c2 or new_color == c3:
            continue
            
        return False
        
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 58.0
    rng = random.Random(42)
    
    best_colors = [0] * NUM_EDGES
    best_score = 1
    
    # Strategy: Multi-start with different initial colorings, then local search
    # to maximize distinct colors while maintaining validity.
    
    num_starts = 8
    per_start_time = (deadline - start_time) / num_starts
    
    for start_i in range(num_starts):
        if time.time() >= deadline - 2.0:
            break
        
        # Different initial strategies
        if start_i == 0:
            # Start with all zeros
            colors = [0] * NUM_EDGES
        elif start_i == 1:
            # Start with a few colors pre-assigned
            colors = [0] * NUM_EDGES
            for i in range(0, NUM_EDGES, 7):
                colors[i] = i % 5
        elif start_i == 2:
            # Random initial valid coloring: assign colors greedily
            colors = [0] * NUM_EDGES
            for e in range(NUM_EDGES):
                # Try to assign a random color
                for attempt in range(20):
                    c = rng.randint(0, 30)
                    if check_edge_change(colors, e, c):
                        colors[e] = c
                        break
        elif start_i == 3:
            # Structured: use vertex-based coloring
            colors = [0] * NUM_EDGES
            for e, (i, j) in enumerate(EDGES):
                colors[e] = (i + j) % 8
        else:
            # Random perturbation of previous best
            colors = list(best_colors)
            for _ in range(rng.randint(5, 20)):
                e = rng.randint(0, NUM_EDGES - 1)
                c = rng.randint(0, MAX_COLOR)
                if check_edge_change(colors, e, c):
                    colors[e] = c
        
        # Verify initial coloring is valid
        if count_violations(colors) != 0:
            # Fix violations by resetting to all zeros
            colors = [0] * NUM_EDGES
        
        # Local search: iteratively try to improve
        start_local = time.time()
        local_deadline = start_local + per_start_time - 1.0
        
        current_score = len(set(colors))
        stuck = 0
        
        while time.time() < local_deadline:
            if current_score == MAX_COLOR + 1:
                break
            
            # Try to add a new color
            used = set(colors)
            if len(used) < MAX_COLOR + 1:
                unused = [c for c in range(MAX_COLOR + 1) if c not in used]
                if unused:
                    # Try multiple random edges with new colors
                    improved = False
                    for _ in range(50):
                        e = rng.randint(0, NUM_EDGES - 1)
                        new_c = rng.choice(unused)
                        if check_edge_change(colors, e, new_c):
                            colors[e] = new_c
                            improved = True
                            break
                    
                    if improved:
                        current_score = len(set(colors))
                        stuck = 0
                        if current_score > best_score:
                            best_score = current_score
                            best_colors = list(colors)
                        continue
            
            # Perturbation: recolor edges to existing or new colors
            e = rng.randint(0, NUM_EDGES - 1)
            old_c = colors[e]
            
            # Try several candidate colors
            candidates = []
            if len(used) > 1:
                candidates.extend([c for c in used if c != old_c][:10])
            # Also try some new colors
            if len(used) < MAX_COLOR + 1:
                candidates.extend(rng.sample(range(MAX_COLOR + 1), min(10, MAX_COLOR + 1 - len(used))))
            
            if candidates:
                new_c = rng.choice(candidates)
                if check_edge_change(colors, e, new_c):
                    colors[e] = new_c
                    new_score = len(set(colors))
                    if new_score > current_score:
                        current_score = new_score
                        stuck = 0
                        if current_score > best_score:
                            best_score = current_score
                            best_colors = list(colors)
                    else:
                        stuck += 1
                        # If stuck too long, do a bigger perturbation
                        if stuck > 200:
                            for _ in range(rng.randint(10, 30)):
                                e2 = rng.randint(0, NUM_EDGES - 1)
                                c2 = rng.randint(0, MAX_COLOR)
                                if check_edge_change(colors, e2, c2):
                                    colors[e2] = c2
                            current_score = len(set(colors))
                            stuck = 0
    
    # Final verification
    if count_violations(best_colors) != 0:
        best_colors = [0] * NUM_EDGES
        best_score = 1
    
    return list(best_colors)
