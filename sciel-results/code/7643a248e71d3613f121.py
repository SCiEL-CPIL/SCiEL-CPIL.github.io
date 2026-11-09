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

def generate_dendrogram_coloring():
    """
    Generates a valid coloring using a random binary hierarchical clustering (dendrogram).
    Each internal node gets a unique color.
    Edge color is the color of the LCA of its two endpoints.
    This guarantees that any 4-cycle has at most 3 distinct colors because the 4 vertices
    form a subtree, and the "highest" LCA in the cycle structure limits the diversity.
    Specifically, for any 4 vertices, the set of LCAs of the pairs forming a cycle will
    be limited. In a binary tree, the number of distinct internal nodes visited by a 
    4-cycle is at most 3.
    """
    vertices = list(range(N))
    random.shuffle(vertices)
    
    # We will build a tree of sets.
    # Each node is a set of vertices.
    # We merge pairs of sets until one remains.
    
    current_sets = [[v] for v in vertices]
    colors = [0] * NUM_EDGES
    color_id = 0
    
    # We need to assign colors to the merges.
    # There will be N-1 merges.
    
    while len(current_sets) > 1:
        # Pick two sets to merge
        i = random.randint(0, len(current_sets) - 1)
        j = random.randint(0, len(current_sets) - 1)
        if i == j:
            continue
        
        set_i = current_sets[i]
        set_j = current_sets[j]
        
        # All edges between set_i and set_j get color_id
        for u in set_i:
            for v in set_j:
                if u < v:
                    e_idx = EDGE_IDX[u][v]
                    colors[e_idx] = color_id
                else:
                    e_idx = EDGE_IDX[v][u]
                    colors[e_idx] = color_id
        
        new_set = set_i + set_j
        # Remove i and j, add new_set
        # To keep indices stable, just append and mark old ones as None or swap
        # Simpler: just build a list of active sets
        current_sets.pop(i)
        current_sets.pop(j)
        current_sets.append(new_set)
        
        color_id += 1
        
    # color_id will be N-1 = 15
    # Colors used are 0..14
    return colors

def local_search_sa(colors, deadline):
    """
    Simulated Annealing to maximize distinct colors.
    Starts from a valid dendrogram coloring.
    """
    colors = list(colors)
    used_colors = set(colors)
    best_score = len(used_colors)
    best_colors = list(colors)
    
    # Temperature schedule
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    # Precompute available colors
    all_colors = set(range(120))
    
    while time.time() < deadline and T > min_T:
        # Pick a random edge
        e = random.randint(0, NUM_EDGES - 1)
        current_color = colors[e]
        
        # Decide target color
        # 10% chance to pick a random unused color (to increase diversity)
        # 90% chance to pick a random used color (to shuffle)
        if random.random() < 0.1:
            unused = list(all_colors - used_colors)
            if not unused:
                new_color = current_color
            else:
                new_color = random.choice(unused)
        else:
            new_color = random.choice(list(used_colors))
            
        if new_color == current_color:
            continue
            
        # Check validity
        if can_take_color(colors, e, new_color):
            # Calculate delta in score
            old_score = len(used_colors)
            
            # Apply move
            colors[e] = new_color
            
            # Update used_colors
            # If current_color was only used here, remove it
            # If new_color was not used here, add it
            # This is expensive to check exactly every time.
            # Approximation: just recalculate set? 120 elements is fast.
            new_used = set(colors)
            new_score = len(new_used)
            
            delta = new_score - old_score
            
            if delta > 0:
                # Accept
                used_colors = new_used
                if new_score > best_score:
                    best_score = new_score
                    best_colors = list(colors)
            else:
                # Accept with probability
                if random.random() < np.exp(-delta / T):
                    used_colors = new_used
                else:
                    # Revert
                    colors[e] = current_color
        
        T *= cooling_rate
        
    return best_colors, best_score

def construct():
    deadline = time.time() + 55  # 55s safety
    
    best_colors = [0] * NUM_EDGES
    best_score = 0
    
    # Try multiple dendrogram seeds
    for restart in range(100):
        if time.time() >= deadline:
            break
            
        # Generate a fresh dendrogram coloring
        base_colors = generate_dendrogram_coloring()
        
        # Verify it's valid (it should be)
        if not is_valid(base_colors):
            continue
            
        # Run SA for a short duration
        # If we have lots of time left, run longer
        # Else, run short
        time_left = deadline - time.time()
        if time_left < 5:
            break
            
        # Allocate 0.5s to 2s per restart depending on how many we can fit
        # We want to try many structures
        sa_time = min(2.0, time_left * 0.1) 
        if restart > 80:
             sa_time = time_left # Use remaining time for last few
        
        ls_colors, score = local_search_sa(base_colors, time.time() + sa_time)
        
        if score > best_score:
            best_score = score
            best_colors = ls_colors
            
        # If we reach a high score, we can be more conservative with time or stop
        if best_score >= 25:
            break

    # Final verification
    if not is_valid(best_colors):
        return [0] * NUM_EDGES
        
    return best_colors

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
