import time
import random
from itertools import combinations

def _is_2_colorable(n, edges):
    """
    Checks if the 5-uniform hypergraph is 2-colorable.
    Returns True if 2-colorable, False otherwise.
    Uses backtracking with unit propagation.
    """
    if not edges:
        return True
    
    colors = [-1] * n
    
    def propagate():
        changed = True
        while changed:
            changed = False
            for edge in edges:
                c0 = 0
                c1 = 0
                unassigned = []
                for v in edge:
                    if colors[v] == 0:
                        c0 += 1
                    elif colors[v] == 1:
                        c1 += 1
                    else:
                        unassigned.append(v)
                
                if c0 == 5 or c1 == 5:
                    return False
                
                if c0 == 4 and len(unassigned) == 1:
                    if colors[unassigned[0]] == -1:
                        colors[unassigned[0]] = 1
                        changed = True
                elif c1 == 4 and len(unassigned) == 1:
                    if colors[unassigned[0]] == -1:
                        colors[unassigned[0]] = 0
                        changed = True
        return True

    def backtrack(idx):
        if not propagate():
            return False
        
        next_v = -1
        for v in range(n):
            if colors[v] == -1:
                next_v = v
                break
        
        if next_v == -1:
            return True
        
        colors[next_v] = 0
        if backtrack(idx + 1):
            return True
        
        colors[next_v] = 1
        if backtrack(idx + 1):
            return True
        
        colors[next_v] = -1
        return False

    return backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Seed solution: Complete 5-uniform hypergraph on 9 vertices
    seed_n = 9
    seed_edges = list(combinations(range(seed_n), 5))
    
    best_n = seed_n
    best_edges = seed_edges
    best_len = len(best_edges)
    
    def minimize(n, edges, time_limit):
        current_edges = list(edges)
        random.shuffle(current_edges)
        
        i = 0
        while i < len(current_edges):
            if time.time() > time_limit:
                break
            
            e = current_edges[i]
            # Try removing e
            test_edges = [x for x in current_edges if x != e]
            
            if not _is_2_colorable(n, test_edges):
                # Still non-2-colorable, so e is not essential
                current_edges = test_edges
                # Do not increment i
            else:
                # e is essential
                i += 1
                
        return current_edges

    # 1. Minimize the seed
    if not _is_2_colorable(best_n, best_edges):
        # It is non-2-colorable
        t_limit = min(start_time + 20.0, deadline)
        minimized = minimize(best_n, best_edges, t_limit)
        if len(minimized) < best_len:
            best_edges = minimized
            best_len = len(best_edges)

    # 2. Random search for smaller instances
    # Focus on n=9, 10, 11
    for n in range(9, 12):
        if time.time() > deadline:
            break
            
        all_edges = list(combinations(range(n), 5))
        
        # Try several random configurations
        for restart in range(10):
            if time.time() > deadline:
                break
                
            # Pick a random subset
            # Start with size ~60-80
            k = min(len(all_edges), 70)
            random_subset = random.sample(all_edges, k)
            
            # Check if non-2-colorable
            if not _is_2_colorable(n, random_subset):
                # It is non-2-colorable. Minimize it.
                t_limit = min(time.time() + 5.0, deadline)
                minimized = minimize(n, random_subset, t_limit)
                
                if len(minimized) < best_len:
                    best_edges = minimized
                    best_n = n
                    best_len = len(minimized)
                    
                    # If we found a very small one, we might stop early or continue
                    # to see if we can go even smaller.
                    # But 51 is the target. If we hit 51, we are good.
                    # Let's continue searching for a bit.
            else:
                # 2-colorable, try denser in next restart?
                # For now, just continue.
                pass

    return {'n': best_n, 'edges': [list(e) for e in best_edges]}
