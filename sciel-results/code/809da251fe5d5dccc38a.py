import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

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
                    # Three 4-cycles on vertices {a,b,c,d}
                    # Cycle 1: a-b-c-d-a
                    cycles.append((EDGE_IDX[a][b], EDGE_IDX[b][c], EDGE_IDX[c][d], EDGE_IDX[d][a]))
                    # Cycle 2: a-b-d-c-a
                    cycles.append((EDGE_IDX[a][b], EDGE_IDX[b][d], EDGE_IDX[d][c], EDGE_IDX[c][a]))
                    # Cycle 3: a-c-b-d-a
                    cycles.append((EDGE_IDX[a][c], EDGE_IDX[c][b], EDGE_IDX[b][d], EDGE_IDX[d][a]))
    return cycles

CYCLES = build_cycles()
NUM_CYCLES = len(CYCLES)

EDGE_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci, cycle in enumerate(CYCLES):
    for e in cycle:
        EDGE_CYCLES[e].append(ci)

def can_take_color(colors, edge, new_color):
    """
    Check if changing 'edge' to 'new_color' keeps the coloring valid.
    A 4-cycle is invalid if it has 4 distinct colors.
    For a cycle (e0, e1, e2, e3) where we change e0 to new_color:
    Let c1, c2, c3 be the colors of e1, e2, e3.
    The set of colors is {new_color, c1, c2, c3}.
    This set has size 4 if and only if:
    1. c1, c2, c3 are all distinct.
    2. new_color is distinct from c1, c2, c3.
    """
    for ci in EDGE_CYCLES[edge]:
        cycle = CYCLES[ci]
        # Determine the other 3 edges
        # cycle is a tuple of 4 edge indices
        # We need to find the index of 'edge' in cycle
        # Since cycle is small, we can just check
        if cycle[0] == edge:
            c1, c2, c3 = colors[cycle[1]], colors[cycle[2]], colors[cycle[3]]
        elif cycle[1] == edge:
            c1, c2, c3 = colors[cycle[0]], colors[cycle[2]], colors[cycle[3]]
        elif cycle[2] == edge:
            c1, c2, c3 = colors[cycle[0]], colors[cycle[1]], colors[cycle[3]]
        else: # cycle[3] == edge
            c1, c2, c3 = colors[cycle[0]], colors[cycle[1]], colors[cycle[2]]
        
        # Check if c1, c2, c3 are distinct
        if c1 != c2 and c1 != c3 and c2 != c3:
            # They are distinct. new_color must match one of them
            if new_color != c1 and new_color != c2 and new_color != c3:
                return False
    return True

def is_valid(colors):
    for cycle in CYCLES:
        c0, c1, c2, c3 = colors[cycle[0]], colors[cycle[1]], colors[cycle[2]], colors[cycle[3]]
        if len({c0, c1, c2, c3}) == 4:
            return False
    return True

def construct():
    deadline = time.time() + 55  # 55s safety margin
    
    # Initialize with all 0s
    colors = [0] * NUM_EDGES
    used_colors = {0}
    best_score = 1
    best_colors = list(colors)
    
    # List of all possible colors
    all_colors = list(range(120))
    
    # Greedy phase + Drift phase
    # We alternate between trying to add new colors and drifting
    
    no_improve = 0
    max_no_improve = 100
    
    while time.time() < deadline:
        # Try to add a new color
        added = False
        edges = list(range(NUM_EDGES))
        random.shuffle(edges)
        
        for e in edges:
            if time.time() >= deadline:
                break
            
            # Try unused colors first
            unused = [c for c in all_colors if c not in used_colors]
            if not unused:
                break # All colors used (unlikely)
            
            random.shuffle(unused)
            
            for nc in unused:
                if can_take_color(colors, e, nc):
                    colors[e] = nc
                    used_colors.add(nc)
                    added = True
                    if len(used_colors) > best_score:
                        best_score = len(used_colors)
                        best_colors = list(colors)
                    break # Try next edge
            
            if added:
                no_improve = 0
                # If we added a color, we might want to continue adding
                # But to ensure we explore, we break out of the edge loop
                # and restart the shuffle to try others
                break
        
        if not added:
            no_improve += 1
            
            # If stuck, perform drift moves
            if no_improve > max_no_improve:
                # Reset no_improve and do a batch of drift moves
                no_improve = 0
                max_no_improve = random.randint(50, 200)
                
                # Drift: recolor edges to existing colors
                for _ in range(50):
                    if time.time() >= deadline:
                        break
                    e = random.randint(0, NUM_EDGES - 1)
                    # Pick a random color from used_colors
                    # Or sometimes pick a new color? No, drift is about rearranging.
                    # But we can also try to swap colors?
                    # Simple recolor is fine.
                    nc = random.choice(list(used_colors))
                    if can_take_color(colors, e, nc):
                        # Check if this move changes the set of used colors
                        # If old color was unique and we change it, we lose a color.
                        # We should avoid losing colors unless necessary.
                        old_c = colors[e]
                        # Count occurrences of old_c
                        # This is O(N). 120 edges. Fast.
                        count_old = 0
                        for i in range(NUM_EDGES):
                            if colors[i] == old_c:
                                count_old += 1
                                if count_old > 1:
                                    break
                        
                        if count_old == 1 and nc != old_c:
                            # This would remove old_c from used_colors
                            # We generally don't want to reduce the score
                            # But it might help escape a local optimum?
                            # For now, let's avoid reducing the score.
                            continue
                        
                        colors[e] = nc
                        # If nc was not used, it is now used (but we picked from used, so it was)
                        # If old_c is now gone, used_colors changes
                        # We don't strictly need to update used_colors here if we only pick from used
                        # But if we removed the last instance of old_c, we should remove it from used_colors
                        # However, since we checked count_old > 1 or nc == old_c,
                        # if count_old == 1 and nc != old_c, we skipped.
                        # So used_colors set remains valid (subset of actual colors).
                        # Actually, if we skipped, colors[e] didn't change.
                        # So used_colors is still correct.
            
            # Also, try to "kick" by trying a new color on a random edge
            # even if greedy failed, maybe a different order works?
            # The greedy shuffle handles this.
            
    # Final verification
    if not is_valid(best_colors):
        # Fallback: all zeros
        return [0] * NUM_EDGES
        
    return best_colors
