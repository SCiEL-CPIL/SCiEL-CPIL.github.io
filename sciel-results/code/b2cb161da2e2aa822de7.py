import time
import random
import sys
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 120

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

    # Precompute all 4-cycles and map edge to cycles
    cycles_with_edge = [[] for _ in range(NUM_EDGES)]
    
    # For fast checking, we also store the other 3 edges in each cycle
    # cycle_data[e] = list of (other1, other2, other3)
    cycle_data = [[] for _ in range(NUM_EDGES)]

    for verts in combinations(range(N), 4):
        a, b, c, d = verts
        # 3 cycles in K4
        e1 = [edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(a,d)]
        e2 = [edge_idx(a,b), edge_idx(b,d), edge_idx(c,d), edge_idx(a,c)]
        e3 = [edge_idx(a,c), edge_idx(b,c), edge_idx(b,d), edge_idx(a,d)]

        for cycle_edges in [e1, e2, e3]:
            for e in cycle_edges:
                others = tuple(x for x in cycle_edges if x != e)
                cycle_data[e].append(others)

    def count_distinct(colors):
        return len(set(colors))

    def is_valid(colors):
        for e in range(NUM_EDGES):
            for others in cycle_data[e]:
                if len(set([colors[e]] + [colors[x] for x in others])) == 4:
                    return False
        return True

    def get_valid_new_colors(colors, e):
        """
        Returns a set of colors that can be assigned to edge e 
        without creating a rainbow 4-cycle.
        """
        # Start with all possible colors
        valid = set(range(MAX_COLOR))
        for others in cycle_data[e]:
            c_others = set(colors[x] for x in others)
            if len(c_others) == 3:
                # To avoid rainbow, colors[e] must be in c_others
                valid.intersection_update(c_others)
            elif len(c_others) == 2:
                # If others have 2 colors, any color is safe?
                # If we pick a 3rd color, we have 3 distinct. Safe.
                # If we pick a 4th color, we have 4 distinct. Rainbow.
                # So valid colors are c_others U {new}?
                # Wait. If others are {1, 2}, and we pick 3, set is {1, 2, 3} (size 3). Safe.
                # If we pick 4, set is {1, 2, 4} (size 3). Safe.
                # So NO constraint from cycles with < 3 distinct other colors.
                pass
            # If len(c_others) < 3, no constraint.
        return valid

    # Initial Coloring: Modular
    def modular_coloring(a, b, p):
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                colors[edge_idx(i, j)] = (i * a + j * b) % p
        return colors

    random.seed(42)
    
    # Find a good base
    best_colors = None
    best_count = 0
    for p in [17, 19, 23]:
        for a in range(1, p):
            for b in range(1, p):
                if a == b: continue
                colors = modular_coloring(a, b, p)
                if is_valid(colors):
                    cnt = count_distinct(colors)
                    if cnt > best_count:
                        best_count = cnt
                        best_colors = colors[:]
    
    if best_colors is None:
        best_colors = [0] * NUM_EDGES
        best_count = 1

    used_colors = set(best_colors)
    
    # Phase 1: Greedy introduction of new colors
    # Order edges by constraint tightness (fewer valid new colors first? Or most?)
    # Let's just iterate and try to add.
    
    edges_order = list(range(NUM_EDGES))
    random.shuffle(edges_order)
    
    # We will do multiple passes
    while time.time() < deadline - 10:
        progress = False
        for e in edges_order:
            if time.time() > deadline - 10:
                break
            valid_new = get_valid_new_colors(best_colors, e)
            # Filter to unused colors
            new_candidates = [c for c in valid_new if c not in used_colors]
            if new_candidates:
                # Pick one
                new_c = random.choice(new_candidates)
                best_colors[e] = new_c
                used_colors.add(new_c)
                best_count += 1
                progress = True
        if not progress:
            # If no progress in a full pass, break or try swaps
            break

    # Phase 2: Local Search with Swaps to enable new colors
    # We try to swap colors of two edges to create a situation where 
    # a new color can be introduced.
    
    # To speed up, we only check validity of the specific edges involved in swaps
    # and the cycles they touch.
    
    def try_swap_and_add(colors, e1, e2, new_edge, new_color):
        # This is complex to implement efficiently inline.
        # Instead, we do a simpler local search:
        # 1. Pick e1, e2. Swap colors.
        # 2. Check if valid.
        # 3. If valid, try to add a new color to a random edge.
        # 4. If added, keep. If not, revert or keep swap if it helps future?
        # For simplicity, we only keep the swap if it immediately allows a new color.
        
        pass

    # Simpler Phase 2: Randomized Hill Climbing
    # Objective: Maximize distinct colors.
    # Move: 
    #   A. Change color of e to new unused c (if valid).
    #   B. Swap colors of e1, e2. (Only if it doesn't reduce distinct count? 
    #      Swapping doesn't change distinct count. But it might enable Move A.)
    
    # We will use a "Look-ahead" swap:
    # Try a swap. Then try to add a new color. 
    # If successful, keep both. If not, revert.
    
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        
        current = best_colors[:]
        current_used = set(current)
        current_count = len(current_used)
        
        for step in range(1000):
            if time.time() > deadline - 5:
                break
            
            # Try to add a new color directly
            e = random.randint(0, NUM_EDGES - 1)
            valid_new = get_valid_new_colors(current, e)
            new_cands = [c for c in valid_new if c not in current_used]
            if new_cands:
                nc = random.choice(new_cands)
                current[e] = nc
                current_used.add(nc)
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_colors = current[:]
                continue
            
            # Try a swap to enable a new color
            # Pick e1, e2
            e1 = random.randint(0, NUM_EDGES - 1)
            e2 = random.randint(0, NUM_EDGES - 1)
            if e1 == e2:
                continue
            
            c1 = current[e1]
            c2 = current[e2]
            if c1 == c2:
                continue
            
            # Perform swap
            current[e1] = c2
            current[e2] = c1
            
            # Check if swap is valid
            valid_swap = True
            # Check cycles for e1 and e2
            for others in cycle_data[e1]:
                if len(set([current[e1]] + [current[x] for x in others])) == 4:
                    valid_swap = False
                    break
            if valid_swap:
                for others in cycle_data[e2]:
                    if len(set([current[e2]] + [current[x] for x in others])) == 4:
                        valid_swap = False
                        break
            
            if not valid_swap:
                # Revert
                current[e1] = c1
                current[e2] = c2
                continue
            
            # Swap is valid. Try to add a new color
            added = False
            # Try a few random edges for new color
            for _ in range(5):
                e_add = random.randint(0, NUM_EDGES - 1)
                valid_new_add = get_valid_new_colors(current, e_add)
                new_cands_add = [c for c in valid_new_add if c not in current_used]
                if new_cands_add:
                    nc_add = random.choice(new_cands_add)
                    current[e_add] = nc_add
                    current_used.add(nc_add)
                    current_count += 1
                    added = True
                    break
            
            if added:
                if current_count > best_count:
                    best_count = current_count
                    best_colors = current[:]
            else:
                # Revert swap
                current[e1] = c1
                current[e2] = c2

    # Final Verification
    if not is_valid(best_colors):
        # Fallback
        for i in range(N - 1):
            for j in range(i + 1, N):
                best_colors[edge_idx(i, j)] = 0

    return best_colors
