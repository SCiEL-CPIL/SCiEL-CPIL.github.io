import time
import random
import sys

_START_TIME = time.time()
DEADLINE = 110.0

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

def construct():
    n = 20
    # Initialize with complete graph K_n
    # adj[i] is a bitmask of neighbors of i
    init_adj = [(1 << n) - 1 for _ in range(n)]
    for i in range(n):
        init_adj[i] &= ~(1 << i)
    
    best_edges = 0
    best_graph = []
    
    attempts = 0
    while _time_left() > 5.0:
        random.seed(2000 + attempts)
        
        # Deep copy adj
        curr_adj = init_adj[:]
        
        # Helper to get edge count
        def get_edge_count(adj_list):
            count = 0
            for i in range(n):
                count += bin(adj_list[i]).count('1')
            return count // 2
        
        curr_edge_count = get_edge_count(curr_adj)
        
        # Deletion loop
        # We need to find a C4 and remove an edge
        # A C4 exists if there are u, w not connected, with >= 2 common neighbors
        while True:
            found_c4 = False
            # To speed up, we can randomize the order of checking pairs (u, w)
            pairs = [(u, w) for u in range(n) for w in range(u + 1, n)]
            random.shuffle(pairs)
            
            for u, w in pairs:
                # Check if u and w are not connected
                if curr_adj[u] & (1 << w):
                    continue
                
                # Find common neighbors
                common = curr_adj[u] & curr_adj[w]
                # Count bits
                if bin(common).count('1') >= 2:
                    # Found a C4. Pick two common neighbors v, x
                    # Get list of common neighbors
                    common_list = []
                    c = common
                    while c:
                        lsb = c & (-c)
                        idx = (lsb.bit_length() - 1)
                        common_list.append(idx)
                        c ^= lsb
                    
                    # Pick two distinct v, x from common_list
                    v, x = random.sample(common_list, 2)
                    
                    # The cycle is u-v-w-x-u
                    # Edges: (u,v), (v,w), (w,x), (x,u)
                    # We need to remove one of these edges.
                    # Heuristic: Remove the edge that is in the most C4s.
                    # Approximation: For each edge (a,b), count how many C4s it is part of.
                    
                    candidates = [(u, v), (v, w), (w, x), (x, u)]
                    scores = []
                    for a, b in candidates:
                        # Count C4s containing edge (a,b)
                        # We need c in N(b) \ {a} and d in N(a) \ {b} such that c~d
                        count = 0
                        nb_b = curr_adj[b] & ~(1 << a)
                        nb_a = curr_adj[a] & ~(1 << b)
                        
                        # Iterate c in nb_b
                        c_mask = nb_b
                        while c_mask:
                            lsb_c = c_mask & (-c_mask)
                            c = lsb_c.bit_length() - 1
                            c_mask ^= lsb_c
                            
                            # d must be in nb_a and d ~ c
                            potential_d = nb_a & curr_adj[c]
                            count += bin(potential_d).count('1')
                        
                        scores.append(count)
                    
                    # Pick the edge with max score (break ties randomly)
                    max_score = max(scores)
                    best_indices = [i for i, s in enumerate(scores) if s == max_score]
                    chosen_idx = random.choice(best_indices)
                    a, b = candidates[chosen_idx]
                    
                    # Remove edge (a, b)
                    curr_adj[a] &= ~(1 << b)
                    curr_adj[b] &= ~(1 << a)
                    curr_edge_count -= 1
                    found_c4 = True
                    break # Restart pair search as adj changed
            
            if not found_c4:
                break
        
        # End of deletion loop for this attempt
        
        if curr_edge_count > best_edges:
            best_edges = curr_edge_count
            # Reconstruct edge list
            best_graph = []
            for i in range(n):
                mask = curr_adj[i]
                while mask:
                    lsb = mask & (-mask)
                    j = lsb.bit_length() - 1
                    if i < j:
                        best_graph.append((i, j))
                    mask ^= lsb
        
        attempts += 1
        
    # Return result
    result = [[u, v, 0] for u, v in best_graph]
    return result
