"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# EVOLVE-BLOCK-START
def construct():
    """
    Construct edge coloring of K_16 with no rainbow 4-cycle.
    Maximize number of distinct colors used.
    Uses algebraic base coloring + local search.
    """
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s kill

    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * N - i*(i+1)//2 + (j - i - 1)

    # Precompute all 4-cycles
    all_4sets = list(itertools.combinations(range(N), 4))
    constraints = []  # list of tuples of 4 edge indices
    for a, b, c, d in all_4sets:
        # Cycle 1: a-b-c-d-a
        constraints.append((edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)))
        # Cycle 2: a-b-d-c-a
        constraints.append((edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a)))
        # Cycle 3: a-c-b-d-a
        constraints.append((edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a)))

    num_constraints = len(constraints)
    
    # Build adjacency: for each edge, which constraints it belongs to
    edge_to_constraints = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(constraints):
        for e in cyc:
            edge_to_constraints[e].append(ci)

    def count_distinct_colors(colors):
        return len(set(colors))

    def is_valid(colors):
        """Check if coloring has no rainbow 4-cycle."""
        for cyc in constraints:
            if len(set(colors[e] for e in cyc)) == 4:
                return False
        return True

    def check_edge_assignment(colors, e, new_col):
        """Check if assigning new_col to edge e creates any rainbow cycle."""
        old_col = colors[e]
        colors[e] = new_col
        valid = True
        for ci in edge_to_constraints[e]:
            cyc = constraints[ci]
            if len(set(colors[ce] for ce in cyc)) == 4:
                valid = False
                break
        colors[e] = old_col
        return valid

    # Strategy 1: Try algebraic colorings f(i,j) = (i+j) % K
    best_result = None
    best_count = 0

    for K in range(1, 120):
        if time.time() > deadline:
            break
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i+1, N):
                colors[edge_idx(i,j)] = (i + j) % K
        cnt = count_distinct_colors(colors)
        if cnt > best_count and is_valid(colors):
            best_count = cnt
            best_result = colors[:]

    # Strategy 2: Try f(i,j) = (i-j) % K
    for K in range(2, 120):
        if time.time() > deadline:
            break
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i+1, N):
                colors[edge_idx(i,j)] = (i - j) % K
        cnt = count_distinct_colors(colors)
        if cnt > best_count and is_valid(colors):
            best_count = cnt
            best_result = colors[:]

    # Strategy 3: Try f(i,j) = (i XOR j) % K for various K
    for K in range(1, 120):
        if time.time() > deadline:
            break
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i+1, N):
                colors[edge_idx(i,j)] = (i ^ j) % K
        cnt = count_distinct_colors(colors)
        if cnt > best_count and is_valid(colors):
            best_count = cnt
            best_result = colors[:]

    # Strategy 4: Try f(i,j) = (a*i + b*j) % K for small a,b
    for K in range(1, 120):
        if time.time() > deadline:
            break
        for a in range(1, 5):
            for b in range(1, 5):
                if a == b:
                    continue
                colors = [0] * NUM_EDGES
                for i in range(N):
                    for j in range(i+1, N):
                        colors[edge_idx(i,j)] = (a*i + b*j) % K
                cnt = count_distinct_colors(colors)
                if cnt > best_count and is_valid(colors):
                    best_count = cnt
                    best_result = colors[:]

    # If no valid coloring found yet, use all zeros
    if best_result is None:
        best_result = [0] * NUM_EDGES
        best_count = 1

    # Strategy 5: Local search to improve - try to change edges to new colors
    # to increase distinct color count while maintaining validity
    if time.time() < deadline - 5:
        current = best_result[:]
        current_count = best_count
        
        for trial in range(200):
            if time.time() > deadline - 2:
                break
            
            # Try random edge-color changes to introduce new colors
            improved = False
            for _ in range(50):
                if time.time() > deadline - 2:
                    break
                e = random.randint(0, NUM_EDGES - 1)
                # Try a new color not currently used
                used = set(current)
                new_col = random.randint(0, 119)
                if new_col in used:
                    continue
                if check_edge_assignment(current, e, new_col):
                    current[e] = new_col
                    current_count += 1
                    improved = True
                    break
            
            if not improved:
                # Try to swap colors to free up space or reorganize
                # Try changing an edge to a different existing color that allows
                # another edge to take a new color
                for _ in range(20):
                    if time.time() > deadline - 2:
                        break
                    e1 = random.randint(0, NUM_EDGES - 1)
                    e2 = random.randint(0, NUM_EDGES - 1)
                    if e1 == e2:
                        continue
                    old1, old2 = current[e1], current[e2]
                    # Swap
                    current[e1], current[e2] = old2, old1
                    if is_valid(current):
                        # Now try to assign a new color to some edge
                        for e3 in range(NUM_EDGES):
                            new_col = random.randint(0, 119)
                            if new_col in set(current):
                                continue
                            if check_edge_assignment(current, e3, new_col):
                                current[e3] = new_col
                                current_count += 1
                                improved = True
                                break
                    # Undo swap
                    current[e1], current[e2] = old1, old2
                    if improved:
                        break
            
            if current_count > best_count:
                best_count = current_count
                best_result = current[:]

    return best_result
# EVOLVE-BLOCK-END
