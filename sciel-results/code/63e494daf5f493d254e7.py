import time
import random
import sys

N = 16
NUM_PAIRS = N * (N - 1) // 2
MAX_COLOR = 119

def pair_index(i, j):
    if i > j: i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

# Precompute cycles and edge-cycle relationships
ALL_CYCLES = []
EDGE_CYCLES = [[] for _ in range(NUM_PAIRS)]
OTHER_EDGES = [None] * NUM_PAIRS # Will store list of tuples of 3 other edge indices for each cycle containing the edge

def build_cycles():
    from itertools import combinations
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        e1 = [pair_index(a,b), pair_index(b,c), pair_index(c,d), pair_index(d,a)]
        # Cycle 2: a-b-d-c-a
        e2 = [pair_index(a,b), pair_index(b,d), pair_index(d,c), pair_index(c,a)]
        # Cycle 3: a-c-b-d-a
        e3 = [pair_index(a,c), pair_index(c,b), pair_index(b,d), pair_index(d,a)]
        
        for cyc in [e1, e2, e3]:
            idx = len(ALL_CYCLES)
            ALL_CYCLES.append(tuple(cyc))
            for e in cyc:
                EDGE_CYCLES[e].append(idx)

build_cycles()

# Precompute other edges for fast lookup
# For each edge e, OTHER_EDGES[e] is a list of tuples (o1, o2, o3) 
# representing the other 3 edges in each cycle that contains e.
for e in range(NUM_PAIRS):
    OTHER_EDGES[e] = []
    for cyc_idx in EDGE_CYCLES[e]:
        cyc = ALL_CYCLES[cyc_idx]
        others = [x for x in cyc if x != e]
        OTHER_EDGES[e].append(tuple(others))

def check_edge_change(coloring, edge_idx, new_color):
    """
    Check if changing coloring[edge_idx] to new_color creates a rainbow cycle.
    Returns True if valid, False otherwise.
    """
    if coloring[edge_idx] == new_color:
        return True
        
    others_list = OTHER_EDGES[edge_idx]
    for others in others_list:
        c1 = coloring[others[0]]
        c2 = coloring[others[1]]
        c3 = coloring[others[2]]
        
        # If any two of the other colors are the same, it can't be a rainbow cycle
        if c1 == c2 or c2 == c3 or c1 == c3:
            continue
            
        # If the new color matches any of the other colors, it's not a rainbow cycle
        if new_color == c1 or new_color == c2 or new_color == c3:
            continue
            
        # If we get here, all 4 colors are distinct
        return False
        
    return True

def is_valid(coloring):
    for cyc in ALL_CYCLES:
        c0, c1, c2, c3 = coloring[cyc[0]], coloring[cyc[1]], coloring[cyc[2]], coloring[cyc[3]]
        if len({c0, c1, c2, c3}) == 4:
            return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    rng = random.Random(42)
    
    best_score = 0
    best_coloring = None
    
    def greedy_construct():
        """
        Greedily assign colors to edges to maximize distinct colors while maintaining validity.
        """
        coloring = [0] * NUM_PAIRS
        used_colors = {0}
        
        # Shuffle edges to break symmetry
        edges = list(range(NUM_PAIRS))
        rng.shuffle(edges)
        
        for e in edges:
            # Try to assign a new color
            unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
            
            if not unused:
                continue
                
            # Try a few random candidates
            # We want to maximize distinct colors, so try unused colors first
            # But we don't know which unused colors are valid without checking
            # Strategy: Pick a random unused color, check if valid.
            # If valid, assign it.
            
            # To be efficient, we can try a few random unused colors
            # If we run out of time, we stop.
            
            # Generate a list of candidate colors
            # We prioritize new colors
            candidates = rng.sample(unused, min(5, len(unused)))
            
            found = False
            for c in candidates:
                if check_edge_change(coloring, e, c):
                    coloring[e] = c
                    used_colors.add(c)
                    found = True
                    break
            
            # If no new color worked, we could try to change to an existing color to "unlock" future changes?
            # For now, just leave it as is.
            
        return coloring, len(used_colors)

    def local_search(coloring, max_iters=50000):
        """
        Local search to increase the number of distinct colors.
        """
        current_coloring = list(coloring)
        used_colors = set(current_coloring)
        current_score = len(used_colors)
        
        if not is_valid(current_coloring):
            return None, 0

        stuck_counter = 0
        max_stuck = 1000
        
        for it in range(max_iters):
            if time.time() > deadline - 2.0:
                break
                
            if current_score == MAX_COLOR + 1:
                break
                
            # Pick a random edge
            idx = rng.randint(0, NUM_PAIRS - 1)
            
            # Try to add a new color
            unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
            
            if unused:
                new_color = rng.choice(unused)
                if check_edge_change(current_coloring, idx, new_color):
                    current_coloring[idx] = new_color
                    used_colors.add(new_color)
                    current_score += 1
                    stuck_counter = 0
                    continue
            
            # If we didn't add a color, try a random perturbation with existing colors
            # This helps escape local optima
            stuck_counter += 1
            if stuck_counter > max_stuck:
                # Perform a random valid recoloring
                p_idx = rng.randint(0, NUM_PAIRS - 1)
                p_old = current_coloring[p_idx]
                # Pick a random used color
                p_new = rng.choice(list(used_colors))
                if p_new != p_old and check_edge_change(current_coloring, p_idx, p_new):
                    current_coloring[p_idx] = p_new
                    stuck_counter = 0
                    max_stuck = int(max_stuck * 1.2)
                else:
                    # Try a swap
                    q_idx = rng.randint(0, NUM_PAIRS - 1)
                    if q_idx != p_idx:
                        c_p = current_coloring[p_idx]
                        c_q = current_coloring[q_idx]
                        if c_p != c_q:
                            current_coloring[p_idx] = c_q
                            current_coloring[q_idx] = c_p
                            # Check validity for both edges
                            valid_p = check_edge_change(current_coloring, p_idx, c_q)
                            valid_q = check_edge_change(current_coloring, q_idx, c_p)
                            if valid_p and valid_q:
                                stuck_counter = 0
                            else:
                                # Revert
                                current_coloring[p_idx] = c_p
                                current_coloring[q_idx] = c_q
        
        if is_valid(current_coloring):
            return current_coloring, current_score
        return None, 0

    # 1. Greedy Construction
    coloring, score = greedy_construct()
    if coloring is not None and score > best_score:
        best_score = score
        best_coloring = coloring
        
    # 2. Local Search Refinement
    if best_coloring is not None:
        refined, ref_score = local_search(best_coloring)
        if refined is not None and ref_score > best_score:
            best_score = ref_score
            best_coloring = refined

    # 3. Try a few more random restarts if time permits
    restarts = 0
    while time.time() < deadline - 10.0:
        restarts += 1
        if restarts > 20: break
        
        coloring, score = greedy_construct()
        if coloring is not None and score > best_score:
            best_score = score
            best_coloring = coloring
            
        if best_coloring is not None:
            refined, ref_score = local_search(best_coloring, max_iters=10000)
            if refined is not None and ref_score > best_score:
                best_score = ref_score
                best_coloring = refined

    if best_coloring is None:
        return [0] * NUM_PAIRS
        
    return best_coloring
# EVOLVE-BLOCK-END
