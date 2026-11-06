import itertools
import random
import time

def construct():
    start_time = time.time()
    time_limit = 118.0  # 120s limit, keep 2s margin
    
    # We will try n=34 first
    n = 34
    edges = list(itertools.combinations(range(n), 3))
    num_edges = len(edges)
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    
    # Precompute 4-sets
    sets_4 = list(itertools.combinations(range(n), 4))
    num_sets_4 = len(sets_4)
    # For each 4-set, store the indices of the 4 edges
    edges_in_set_4 = []
    for s in sets_4:
        # 4 vertices a,b,c,d -> 4 triples: (a,b,c), (a,b,d), (a,c,d), (b,c,d)
        # Ensure lex order of triples matches edge_to_idx
        tris = [
            (s[0], s[1], s[2]),
            (s[0], s[1], s[3]),
            (s[0], s[2], s[3]),
            (s[1], s[2], s[3])
        ]
        idxs = [edge_to_idx[t] for t in tris]
        edges_in_set_4.append(idxs)
        
    # Precompute 5-sets
    sets_5 = list(itertools.combinations(range(n), 5))
    num_sets_5 = len(sets_5)
    # For each 5-set, store the indices of the 10 edges
    edges_in_set_5 = []
    for s in sets_5:
        tris = list(itertools.combinations(s, 3))
        idxs = [edge_to_idx[t] for t in tris]
        edges_in_set_5.append(idxs)
        
    # For each edge, store which 4-sets and 5-sets it belongs to
    # edge_to_sets_4: list of lists
    edge_to_sets_4 = [[] for _ in range(num_edges)]
    for i, idxs in enumerate(edges_in_set_4):
        for idx in idxs:
            edge_to_sets_4[idx].append(i)
            
    edge_to_sets_5 = [[] for _ in range(num_edges)]
    for i, idxs in enumerate(edges_in_set_5):
        for idx in idxs:
            edge_to_sets_5[idx].append(i)

    best_colors = None
    best_cost = float('inf')
    
    # Multiple restarts
    max_restarts = 100
    for r in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        colors = [random.randint(0, 1) for _ in range(num_edges)]
        
        # Initialize counts
        red_count_4 = [0] * num_sets_4
        blue_count_5 = [0] * num_sets_5
        
        # Count red edges in each 4-set
        for i, idxs in enumerate(edges_in_set_4):
            count = 0
            for idx in idxs:
                if colors[idx] == 0: # Red
                    count += 1
            red_count_4[i] = count
            
        # Count blue edges in each 5-set
        for i, idxs in enumerate(edges_in_set_5):
            count = 0
            for idx in idxs:
                if colors[idx] == 1: # Blue
                    count += 1
            blue_count_5[i] = count
            
        # Calculate initial cost
        cost = sum(1 for c in red_count_4 if c == 4) + sum(1 for c in blue_count_5 if c == 10)
        
        if cost == 0:
            best_colors = colors
            break
            
        # Local Search
        steps = 2000
        for step in range(steps):
            if time.time() - start_time > time_limit:
                break
                
            if cost == 0:
                best_colors = colors
                break
                
            # Pick random edge
            e_idx = random.randint(0, num_edges - 1)
            curr_color = colors[e_idx]
            
            # Calculate delta cost
            delta = 0
            
            if curr_color == 0: # Red, flip to Blue
                # Affects 4-sets: red_count decreases by 1
                for s_idx in edge_to_sets_4[e_idx]:
                    if red_count_4[s_idx] == 4:
                        delta -= 1 # Breaks a red K4
                    elif red_count_4[s_idx] == 3:
                        delta += 1 # Creates a red K4? No, 3->2. Wait.
                        # If red_count is 3, and we flip one Red to Blue, it becomes 2. No new conflict.
                        # If red_count is 4, it becomes 3. Conflict removed.
                        # So delta is -1 if count==4, else 0.
                        pass # Correction:
                        # If count==4, delta -= 1
                        # If count==3, delta += 0 (3->2, no conflict)
                        # If count<3, delta += 0
                        # Wait, what if we flip Blue to Red?
                        # If count==3, 3->4, delta += 1
                        # If count==4, 4->5 (impossible, max 4), so 4->3? No, 4->4? No.
                        # Max red edges in 4-set is 4.
                        
                # Affects 5-sets: blue_count increases by 1
                for s_idx in edge_to_sets_5[e_idx]:
                    if blue_count_5[s_idx] == 10:
                        # Impossible if current edge is Red (0). Blue count doesn't include this edge.
                        # So blue_count_5[s_idx] is count of OTHER blue edges.
                        # Max is 9.
                        pass
                    if blue_count_5[s_idx] == 9:
                        # 9 -> 10. Creates blue K5.
                        delta += 1
                    # else 0
                    
            else: # Blue, flip to Red
                # Affects 4-sets: red_count increases by 1
                for s_idx in edge_to_sets_4[e_idx]:
                    if red_count_4[s_idx] == 3:
                        # 3 -> 4. Creates red K4.
                        delta += 1
                    # if red_count_4[s_idx] == 4, impossible (edge is Blue)
                    
                # Affects 5-sets: blue_count decreases by 1
                for s_idx in edge_to_sets_5[e_idx]:
                    if blue_count_5[s_idx] == 10:
                        # 10 -> 9. Breaks blue K5.
                        delta -= 1
                    # else 0

            # Apply flip if delta < 0
            if delta < 0:
                colors[e_idx] = 1 - curr_color
                cost += delta
                
                # Update counts
                if curr_color == 0: # Was Red, now Blue
                    for s_idx in edge_to_sets_4[e_idx]:
                        red_count_4[s_idx] -= 1
                    for s_idx in edge_to_sets_5[e_idx]:
                        blue_count_5[s_idx] += 1
                else: # Was Blue, now Red
                    for s_idx in edge_to_sets_4[e_idx]:
                        red_count_4[s_idx] += 1
                    for s_idx in edge_to_sets_5[e_idx]:
                        blue_count_5[s_idx] -= 1
            elif delta == 0 and random.random() < 0.1:
                # Random walk to escape local minima
                colors[e_idx] = 1 - curr_color
                # cost unchanged
                
                if curr_color == 0:
                    for s_idx in edge_to_sets_4[e_idx]:
                        red_count_4[s_idx] -= 1
                    for s_idx in edge_to_sets_5[e_idx]:
                        blue_count_5[s_idx] += 1
                else:
                    for s_idx in edge_to_sets_4[e_idx]:
                        red_count_4[s_idx] += 1
                    for s_idx in edge_to_sets_5[e_idx]:
                        blue_count_5[s_idx] -= 1

        if cost == 0:
            best_colors = colors
            break

    if best_colors:
        return {'n': n, 'colors': best_colors}
    else:
        # Fallback to n=9
        n = 9
        return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n), 3)]}
