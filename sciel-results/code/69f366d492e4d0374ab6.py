import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def construct():
    # Precompute edge endpoints
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
            
    # Map edge index to (u, v)
    edge_to_uv = {idx: edges[idx] for idx in range(NUM_EDGES)}
    
    # Precompute constraints for each edge
    # constraints[e_idx] is a list of tuples (e1, e2, e3) such that (e_idx, e1, e2, e3) is a 4-cycle
    constraints = [[] for _ in range(NUM_EDGES)]
    for e_idx in range(NUM_EDGES):
        u, v = edges[e_idx]
        others = [k for k in range(N) if k != u and k != v]
        for x, y in itertools.combinations(others, 2):
            # Cycle 1: u-v-x-y-u
            e1 = edge_idx(v, x)
            e2 = edge_idx(x, y)
            e3 = edge_idx(y, u)
            constraints[e_idx].append((e1, e2, e3))
            # Cycle 2: u-v-y-x-u
            e1 = edge_idx(v, y)
            e2 = edge_idx(y, x)
            e3 = edge_idx(x, u)
            constraints[e_idx].append((e1, e2, e3))

    def is_valid_color(e_idx, c_new, colors):
        """Check if assigning c_new to e_idx is valid given current colors."""
        for e1, e2, e3 in constraints[e_idx]:
            c1 = colors[e1]
            c2 = colors[e2]
            c3 = colors[e3]
            # Invalid if c_new, c1, c2, c3 are all distinct
            if c_new != c1 and c_new != c2 and c_new != c3:
                if c1 != c2 and c1 != c3 and c2 != c3:
                    return False
        return True

    best_colors = [0] * NUM_EDGES
    best_distinct = 0
    
    # Run multiple restarts
    for restart in range(30):
        # Initialize randomly
        colors = [0] * NUM_EDGES
        # Greedy init: assign random valid colors
        for e in range(NUM_EDGES):
            # Since all start at 0, 0 is always valid.
            # We can just pick a random valid color to start diversity
            # But 0 is safe. Let's pick random 0..119 and check.
            # Actually, if we start with all 0s, it's valid.
            # Let's just use 0 for simplicity and let local search do the work.
            # Or better, random valid.
            c = random.randint(0, 119)
            if is_valid_color(e, c, colors):
                colors[e] = c
            else:
                colors[e] = 0 # Fallback to 0, which is always valid if neighbors are 0?
                # If neighbors are not 0, 0 might be invalid.
                # To be safe, just start with 0.
        
        # Recalculate freq and distinct
        freq = [0] * 120
        for c in colors:
            freq[c] += 1
        current_distinct = sum(1 for f in freq if f > 0)
        
        # Local Search
        for step in range(2000):
            improved = False
            edge_order = list(range(NUM_EDGES))
            random.shuffle(edge_order)
            
            for e_idx in edge_order:
                old_color = colors[e_idx]
                best_new_color = -1
                best_delta = 0
                
                # We can optimize: if we already have a delta > 0, we can stop looking for better?
                # No, we want the BEST delta.
                # But if we find delta = 2 (impossible, max 1? No, if freq[old]==1 and freq[new]==0, delta=1.
                # If freq[old]>1 and freq[new]==0, delta=1.
                # If freq[old]==1 and freq[new]>0, delta=0.
                # If freq[old]>1 and freq[new]>0, delta=0.
                # So max delta is 1.
                # If we find a move with delta=1, we can accept it immediately?
                # Yes, because we can't get better than 1.
                
                for c_new in range(120):
                    if c_new == old_color:
                        continue
                    if not is_valid_color(e_idx, c_new, colors):
                        continue
                    
                    # Calculate delta distinct
                    delta = 0
                    if freq[old_color] == 1:
                        delta -= 1
                    if freq[c_new] == 0:
                        delta += 1
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_new_color = c_new
                        if best_delta == 1:
                            break # Can't get better than 1
                    elif delta == best_delta and best_new_color != -1:
                        if random.random() < 0.5:
                            best_new_color = c_new

                if best_new_color != -1:
                    freq[old_color] -= 1
                    freq[best_new_color] += 1
                    colors[e_idx] = best_new_color
                    current_distinct += best_delta
                    improved = True
            
            if not improved:
                break
        
        if current_distinct > best_distinct:
            best_distinct = current_distinct
            best_colors = colors[:]
            
    return best_colors
