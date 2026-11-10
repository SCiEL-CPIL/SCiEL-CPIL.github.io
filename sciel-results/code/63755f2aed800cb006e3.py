import time
import random
import sys
from itertools import combinations

N = 16
NUM_PAIRS = N * (N - 1) // 2
MAX_COLOR = 119

def pair_index(i, j):
    if i > j: i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

# Precompute cycles and edge-cycle relationships
ALL_CYCLES = []
EDGE_CYCLES = [[] for _ in range(NUM_PAIRS)]
OTHER_EDGES = [None] * NUM_PAIRS 

def build_cycles():
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
# OTHER_EDGES[e] is a list of tuples, where each tuple contains the other 3 edge indices
# in a cycle that includes edge e.
for e in range(NUM_PAIRS):
    OTHER_EDGES[e] = []
    for cyc_idx in EDGE_CYCLES[e]:
        cyc = ALL_CYCLES[cyc_idx]
        others = [x for x in cyc if x != e]
        OTHER_EDGES[e].append(tuple(others))

def is_valid(coloring):
    """
    Check if the entire coloring is valid.
    Used for final verification and initial checks.
    """
    for cyc in ALL_CYCLES:
        c0, c1, c2, c3 = coloring[cyc[0]], coloring[cyc[1]], coloring[cyc[2]], coloring[cyc[3]]
        if len({c0, c1, c2, c3}) == 4:
            return False
    return True

def check_edge_change(coloring, edge_idx, new_color):
    """
    Check if changing coloring[edge_idx] to new_color creates a rainbow cycle.
    Returns True if valid, False otherwise.
    Assumes coloring[edge_idx] is currently valid.
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
            
        # If we get here, all 4 colors are distinct -> Rainbow cycle violation
        return False
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    rng = random.Random(42)
    
    best_score = 0
    best_coloring = None

    def greedy_construct():
        """
        Greedily assign colors to edges to maximize distinct colors while maintaining validity.
        """
        coloring = [0] * NUM_PAIRS
        used_colors = {0}
        
        # Shuffle edges to break symmetry and allow different structures
        edges = list(range(NUM_PAIRS))
        rng.shuffle(edges)
        
        for e in edges:
            current_color = coloring[e]
            
            # We want to try to assign a NEW color to this edge
            # Generate a list of candidate new colors
            # To be efficient, we pick random unused colors
            # We keep trying until we find one that works or we give up (keep old color)
            
            # Optimization: If we have used all 120 colors, we can't add more
            if len(used_colors) > MAX_COLOR:
                continue

            # Try a few random unused colors
            # We sample from the range of unused colors
            # Instead of building a list of all unused (expensive if many used),
            # we just pick random integers and check if they are used.
            # However, as used_colors grows, this becomes inefficient.
            # For N=16, max 120 colors.
            
            # Better approach: maintain a list of unused colors? 
            # Or just random sampling.
            
            found = False
            # Try up to 50 random unused colors
            for _ in range(50):
                c = rng.randint(0, MAX_COLOR)
                if c in used_colors:
                    continue
                if check_edge_change(coloring, e, c):
                    coloring[e] = c
                    used_colors.add(c)
                    found = True
                    break
            
            # If no new color worked, keep the current color (which is valid)
            # We don't try to change to an existing color here, as that doesn't increase diversity
            # and might disrupt future potential new color assignments? 
            # Actually, keeping the initial 0 is fine. The local search will handle diversification.
            
        return coloring, len(used_colors)

    def local_search(coloring, max_iters=100000):
        """
        Local search to increase the number of distinct colors.
        """
        current_coloring = list(coloring)
        used_colors = set(current_coloring)
        current_score = len(used_colors)
        
        if not is_valid(current_coloring):
            return None, 0

        stuck_counter = 0
        max_stuck = 100
        
        for it in range(max_iters):
            if time.time() > deadline - 2.0:
                break
                
            if current_score > MAX_COLOR:
                break
                
            # Pick a random edge
            idx = rng.randint(0, NUM_PAIRS - 1)
            
            # Try to add a new color
            # Pick a random unused color
            # We need an efficient way to pick unused colors.
            # Let's just pick random and check.
            success = False
            for _ in range(20): # Try 20 times to find a valid new color
                new_color = rng.randint(0, MAX_COLOR)
                if new_color in used_colors:
                    continue
                if check_edge_change(current_coloring, idx, new_color):
                    current_coloring[idx] = new_color
                    used_colors.add(new_color)
                    current_score += 1
                    stuck_counter = 0
                    success = True
                    break
            
            if success:
                continue
            
            # If we didn't add a color, try a random perturbation with existing colors
            # This helps escape local optima where no single edge can take a new color,
            # but a sequence of changes might allow it.
            stuck_counter += 1
            if stuck_counter > max_stuck:
                # Perform a random valid recoloring
                p_idx = rng.randint(0, NUM_PAIRS - 1)
                p_old = current_coloring[p_idx]
                
                # Pick a random used color
                # To avoid bias, pick from used_colors
                # If used_colors is large, this is O(1) with list conversion or just random pick from set
                # For 120 items, list(used_colors) is fine.
                p_new = rng.choice(list(used_colors))
                
                if p_new != p_old and check_edge_change(current_coloring, p_idx, p_new):
                    current_coloring[p_idx] = p_new
                    stuck_counter = 0
                    # Optionally increase max_stuck to allow more exploration
                    # max_stuck = int(max_stuck * 1.1)
        
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
        refined, ref_score = local_search(best_coloring, max_iters=50000)
        if refined is not None and ref_score > best_score:
            best_score = ref_score
            best_coloring = refined

    # 3. Try a few more random restarts if time permits
    restarts = 0
    while time.time() < deadline - 10.0:
        restarts += 1
        if restarts > 50: break
        
        coloring, score = greedy_construct()
        if coloring is not None and score > best_score:
            best_score = score
            best_coloring = coloring
            
        if best_coloring is not None:
            # Shorter local search for restarts
            refined, ref_score = local_search(best_coloring, max_iters=10000)
            if refined is not None and ref_score > best_score:
                best_score = ref_score
                best_coloring = refined

    if best_coloring is None:
        return [0] * NUM_PAIRS
        
    # Final validation
    if not is_valid(best_coloring):
        # This should not happen, but if it does, return a known valid solution (all zeros)
        return [0] * NUM_PAIRS
        
    return best_coloring
