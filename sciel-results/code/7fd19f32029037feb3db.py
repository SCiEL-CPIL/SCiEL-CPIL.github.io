import time
import random
import sys
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
# A 4-cycle is a set of 4 edges forming a cycle.
# In K4, there are 3 cycles.
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
# OTHER_EDGES[e] = list of tuples (o1, o2, o3) representing the other 3 edges in the cycle
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
        
        # If any two of the other colors are the same, it can't be a rainbow cycle
        if c1 == c2 or c2 == c3 or c1 == c3:
            continue
            
        # If the new color matches any of the other colors, it's not a rainbow cycle
        if new_color == c1 or new_color == c2 or new_color == c3:
            continue
            
        # If we get here, all 4 colors are distinct
        return False
        
    return True

def check_swap(colors, e1, e2):
    """
    Check if swapping colors of e1 and e2 creates a rainbow cycle.
    Returns True if valid, False otherwise.
    """
    if e1 == e2:
        return True
        
    c1 = colors[e1]
    c2 = colors[e2]
    
    if c1 == c2:
        return True
        
    # We need to check all cycles that contain e1 or e2
    # To avoid double checking cycles containing both, we can just check all unique cycles.
    # However, checking duplicates is fine for correctness, just slightly slower.
    # Given small N, we can just iterate.
    
    # Optimize: Only check cycles that involve e1 or e2.
    # Set of cycles to check
    cycles_to_check = set()
    for other in OTHER_EDGES[e1]:
        cycles_to_check.add((e1, other[0], other[1], other[2]))
    for other in OTHER_EDGES[e2]:
        cycles_to_check.add((e2, other[0], other[1], other[2]))
        
    # Note: The tuples in OTHER_EDGES are unordered sets of 3 edges.
    # The cycle is {e1, o1, o2, o3}.
    # We need to check if the set of colors {c_new_e1, c_o1, c_o2, c_o3} are all distinct.
    
    for cyc_tuple in cycles_to_check:
        # cyc_tuple is (e_a, e_b, e_c, e_d) where one is e1 or e2
        # Actually, the loop above adds tuples like (e1, o1, o2, o3).
        # Let's just extract the 4 edge indices.
        # The structure of OTHER_EDGES[e1] is list of (o1, o2, o3).
        # So the cycle edges are e1, o1, o2, o3.
        
        # Wait, the set logic above is a bit messy.
        # Let's just iterate through OTHER_EDGES[e1] and OTHER_EDGES[e2] separately.
        
        pass # Placeholder, logic below

    # Simpler implementation:
    # Check cycles containing e1
    for o1, o2, o3 in OTHER_EDGES[e1]:
        # Cycle edges: e1, o1, o2, o3
        # New colors:
        # e1 gets c2
        # o1, o2, o3 keep their colors
        c_new_e1 = c2
        c_o1 = colors[o1]
        c_o2 = colors[o2]
        c_o3 = colors[o3]
        
        if c_new_e1 != c_o1 and c_new_e1 != c_o2 and c_new_e1 != c_o3 and \
           c_o1 != c_o2 and c_o1 != c_o3 and c_o2 != c_o3:
            return False
            
    # Check cycles containing e2
    for o1, o2, o3 in OTHER_EDGES[e2]:
        # Cycle edges: e2, o1, o2, o3
        # New colors:
        # e2 gets c1
        # o1, o2, o3 keep their colors
        c_new_e2 = c1
        c_o1 = colors[o1]
        c_o2 = colors[o2]
        c_o3 = colors[o3]
        
        if c_new_e2 != c_o1 and c_new_e2 != c_o2 and c_new_e2 != c_o3 and \
           c_o1 != c_o2 and c_o1 != c_o3 and c_o2 != c_o3:
            return False
            
    return True

def greedy_initialization(rng):
    """
    Initialize with a valid coloring by greedily assigning new colors to shuffled edges.
    """
    colors = [0] * NUM_EDGES
    used_colors = {0}
    
    edges = list(range(NUM_EDGES))
    rng.shuffle(edges)
    
    for e in edges:
        if len(used_colors) == MAX_COLOR + 1:
            break
        # Try to assign a new unused color
        # Pick a random unused color
        unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
        if not unused:
            break
        # Try a few candidates to increase success rate
        candidates = rng.sample(unused, min(10, len(unused)))
        for c in candidates:
            if check_edge_change(colors, e, c):
                colors[e] = c
                used_colors.add(c)
                break
                
    return colors, used_colors

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_colors = [0] * NUM_EDGES
    best_score = 1
    best_used = {0}
    
    # Try multiple random seeds/restarts
    for seed in range(50):
        if time.time() > deadline - 5.0:
            break
            
        rng = random.Random(42 + seed)
        
        # Phase 1: Greedy Initialization
        colors, used_colors = greedy_initialization(rng)
        current_score = len(used_colors)
        
        if current_score > best_score:
            best_score = current_score
            best_colors = list(colors)
            best_used = set(used_colors)
            
        # Phase 2: Local Search
        stuck_counter = 0
        max_stuck = 200
        
        while time.time() < deadline - 5.0:
            if current_score == MAX_COLOR + 1:
                break
            
            # 1. Try to add a new unused color
            if len(used_colors) < MAX_COLOR + 1:
                e = rng.randint(0, NUM_EDGES - 1)
                unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
                if unused:
                    new_c = rng.choice(unused)
                    if check_edge_change(colors, e, new_c):
                        colors[e] = new_c
                        used_colors.add(new_c)
                        current_score = len(used_colors)
                        stuck_counter = 0
                        if current_score > best_score:
                            best_score = current_score
                            best_colors = list(colors)
                            best_used = set(used_colors)
                        continue
            
            # 2. Perturbation: Recolor an edge to an EXISTING color
            # or Swap two edges
            move_type = rng.random()
            if move_type < 0.5:
                # Recolor
                e = rng.randint(0, NUM_EDGES - 1)
                old_c = colors[e]
                
                if len(used_colors) > 1:
                    candidates = [c for c in used_colors if c != old_c]
                    if candidates:
                        new_c = rng.choice(candidates)
                        if check_edge_change(colors, e, new_c):
                            colors[e] = new_c
                            if old_c not in colors:
                                used_colors.discard(old_c)
                            
                            stuck_counter = 0
                            max_stuck = int(max_stuck * 1.05)
                        else:
                            stuck_counter += 1
                else:
                    stuck_counter += 1
            else:
                # Swap
                e1 = rng.randint(0, NUM_EDGES - 1)
                e2 = rng.randint(0, NUM_EDGES - 1)
                if e1 != e2:
                    if check_swap(colors, e1, e2):
                        # Perform swap
                        temp = colors[e1]
                        colors[e1] = colors[e2]
                        colors[e2] = temp
                        stuck_counter = 0
                        max_stuck = int(max_stuck * 1.05)
                    else:
                        stuck_counter += 1
            
            # If stuck for too long, force a bigger perturbation
            if stuck_counter > max_stuck:
                # Recolor 5 random edges to random existing colors
                for _ in range(5):
                    p_e = rng.randint(0, NUM_EDGES - 1)
                    p_old = colors[p_e]
                    if len(used_colors) > 1:
                        p_cands = [c for c in used_colors if c != p_old]
                        if p_cands:
                            p_new = rng.choice(p_cands)
                            if check_edge_change(colors, p_e, p_new):
                                colors[p_e] = p_new
                                if p_old not in colors:
                                    used_colors.discard(p_old)
                stuck_counter = 0
                max_stuck = int(max_stuck * 1.1)

    # Final safety check
    if count_violations(best_colors) != 0:
        best_colors = [0] * NUM_EDGES
        best_score = 1

    return list(best_colors)
