import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def build_four_cycles():
    cycles = []
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                for d in range(c+1, N):
                    e1 = [edge_index(a,b), edge_index(b,c), edge_index(c,d), edge_index(d,a)]
                    cycles.append(e1)
                    e2 = [edge_index(a,b), edge_index(b,d), edge_index(d,c), edge_index(c,a)]
                    cycles.append(e2)
                    e3 = [edge_index(a,c), edge_index(c,b), edge_index(b,d), edge_index(d,a)]
                    cycles.append(e3)
    return cycles

CYCLES = build_four_cycles()
CYCLES_BY_EDGE = [[] for _ in range(NUM_EDGES)]
for cycle in CYCLES:
    for e in cycle:
        CYCLES_BY_EDGE[e].append(cycle)

# Precompute the "other 3 edges" for each edge in each cycle it belongs to
# CYCLES_BY_EDGE[e] is a list of cycles.
# Let's transform this to a list of tuples (other1, other2, other3) for each e.
EDGE_OTHERS = [[] for _ in range(NUM_EDGES)]
for e in range(NUM_EDGES):
    for cycle in CYCLES_BY_EDGE[e]:
        others = [x for x in cycle if x != e]
        EDGE_OTHERS[e].append(tuple(others))

def is_valid_change(colors, edge_idx, new_color):
    if colors[edge_idx] == new_color:
        return True
    
    for others in EDGE_OTHERS[edge_idx]:
        c1 = colors[others[0]]
        c2 = colors[others[1]]
        c3 = colors[others[2]]
        if len({new_color, c1, c2, c3}) == 4:
            return False
    return True

def construct():
    deadline = time.time() + 55
    rng = random.Random(42)
    
    colors = [0] * NUM_EDGES
    color_counts = [0] * (MAX_COLOR + 1)
    color_counts[0] = NUM_EDGES
    
    unused_colors = set(range(1, MAX_COLOR + 1))
    
    best_count = 1
    best_colors = colors[:]
    
    while time.time() < deadline:
        # 80% chance to try to add a new color
        if unused_colors and rng.random() < 0.8:
            # Pick a random unused color
            new_color = rng.choice(list(unused_colors))
            # Pick a random edge
            e = rng.randint(0, NUM_EDGES - 1)
            
            if is_valid_change(colors, e, new_color):
                old_color = colors[e]
                colors[e] = new_color
                color_counts[old_color] -= 1
                color_counts[new_color] += 1
                
                if color_counts[old_color] == 0:
                    unused_colors.add(old_color)
                unused_colors.remove(new_color)
                
                cur_count = MAX_COLOR + 1 - len(unused_colors)
                if cur_count > best_count:
                    best_count = cur_count
                    best_colors = colors[:]
        
        else:
            # Shuffle: Recolor an edge to an existing color
            e = rng.randint(0, NUM_EDGES - 1)
            old_color = colors[e]
            
            # Pick a random used color
            # We can pick a random index and check if count > 0
            # Or maintain a list of used colors.
            # Let's just pick a random color and check.
            new_color = rng.randint(0, MAX_COLOR)
            
            if new_color == old_color:
                continue
                
            if color_counts[new_color] == 0:
                # This is an expansion attempt, but we are in shuffle mode.
                # We can still accept it if valid, as it's a valid move.
                # But we treat it as shuffle? No, it changes the count.
                # Let's allow it.
                pass
                
            if is_valid_change(colors, e, new_color):
                colors[e] = new_color
                color_counts[old_color] -= 1
                color_counts[new_color] += 1
                
                if color_counts[old_color] == 0:
                    unused_colors.add(old_color)
                if color_counts[new_color] == 0: # Should not happen if we checked count>0, but if it was 0 before, now it is 1.
                    # If it was 0, it was in unused_colors. Now it is used.
                    unused_colors.remove(new_color)
                # If it was >0, it was not in unused_colors. Still not.
                
                # Update best?
                # The count of used colors might have decreased (if old_color count became 0 and new_color count was >0)
                # Or increased (if new_color count was 0 and old_color count became >0? No, old_color count decreases).
                # Actually, if we recolor to an existing color, the number of used colors can only decrease or stay same.
                # Unless we recolor to a color that was unused (count 0).
                # In that case, it's an expansion.
                
                cur_count = MAX_COLOR + 1 - len(unused_colors)
                if cur_count > best_count:
                    best_count = cur_count
                    best_colors = colors[:]

    return best_colors
