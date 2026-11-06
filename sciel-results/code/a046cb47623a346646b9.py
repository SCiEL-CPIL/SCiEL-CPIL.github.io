import random
import time

def construct():
    d = 7
    n = 2 ** d
    total_edges = n * d // 2
    
    # Generate all edges
    edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
    
    # Precompute neighbors for fast lookup? 
    # Or just use bit operations.
    # We need to check if an edge (u, v) can be added to a set E.
    # Condition: No w in N(u) such that (u,w) in E and (w, z) in E for some z in N(v) such that (v,z) in E.
    # Actually, simpler:
    # For edge (u,v), check all w in N(u) \ {v}.
    # If (u,w) is in E, then check all z in N(v) \ {u}.
    # If (v,z) is in E and (w,z) is in E, then C4 exists.
    
    # To make this fast, we maintain a set of edges `chosen_set`.
    # Lookup in set is O(1).
    
    def can_add(u, v, chosen_set):
        # Check for C4 involving (u,v)
        # Neighbors of u: u ^ (1<<i)
        # Neighbors of v: v ^ (1<<i)
        for i in range(d):
            w = u ^ (1 << i)
            if w == v:
                continue
            if (min(u, w), max(u, w)) in chosen_set:
                # Edge (u,w) exists
                for j in range(d):
                    z = v ^ (1 << j)
                    if z == u:
                        continue
                    if (min(v, z), max(v, z)) in chosen_set:
                        # Edge (v,z) exists
                        # Check if (w,z) exists
                        if (min(w, z), max(w, z)) in chosen_set:
                            return False
        return True

    def greedy_add(chosen_set, edge_list, random_order=False):
        if random_order:
            random.shuffle(edge_list)
        for e in edge_list:
            u, v = e
            if can_add(u, v, chosen_set):
                chosen_set.add((u, v))
        return chosen_set

    best_size = 0
    best_set = set()
    
    # Time limit
    start_time = time.time()
    max_time = 110.0 # Leave 10s buffer
    
    # Multiple restarts with local search
    while time.time() - start_time < max_time:
        # 1. Random Greedy Initialization
        current_set = set()
        # Use a random permutation of edges
        random.shuffle(edges)
        greedy_add(current_set, edges)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
        
        # 2. Local Search / Kicks
        # Try to add more edges
        # We can iterate over all edges to see if any can be added
        # But if greedy was complete, no single edge can be added.
        # So we need kicks.
        
        kick_size = 5
        for _ in range(100): # Max kicks per restart
            if time.time() - start_time > max_time:
                break
            
            # Remove kick_size random edges
            to_remove = random.sample(list(current_set), min(kick_size, len(current_set)))
            for e in to_remove:
                current_set.remove(e)
            
            # Try to add back edges, prioritizing those that were removed or random
            # We can just run a greedy pass again on a subset or all edges
            # To be efficient, we can try adding the removed ones first, then others
            # But a full greedy pass on all edges is O(E * 49) which is ~448 * 49 ~ 22k ops. Very fast.
            # So we can just do a full greedy pass after each kick.
            
            # However, greedy depends on order.
            # Let's shuffle edges again and greedy add.
            random.shuffle(edges)
            greedy_add(current_set, edges)
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
                
            # If we reached the upper bound, stop
            if best_size >= 308:
                break

    # Convert best_set to list of lists
    return [list(e) for e in best_set]
