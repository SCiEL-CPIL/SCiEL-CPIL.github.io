import random
import time

def construct():
    d = 7
    n_vertices = 1 << d
    edges = []
    
    # Generate all edges
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v))
                
    # Helper to check if adding edge (u, v) creates a C4
    def can_add(u, v, selected):
        # Find the bit index that differs
        diff = u ^ v
        i = diff.bit_length() - 1
        
        for j in range(d):
            if j == i:
                continue
            mask = 1 << j
            w1 = u ^ mask
            w2 = v ^ mask
            
            e1 = (min(u, w1), max(u, w1))
            e2 = (min(v, w2), max(v, w2))
            e3 = (min(w1, w2), max(w1, w2))
            
            if e1 in selected and e2 in selected and e3 in selected:
                return False
        return True

    best_set = set()
    start_time = time.time()
    # Set deadline to 100s to ensure we finish well within the 120s limit
    deadline = start_time + 100.0
    
    # Phase 1: Randomized Greedy
    while time.time() < deadline:
        random.shuffle(edges)
        current_set = set()
        for u, v in edges:
            if can_add(u, v, current_set):
                current_set.add((u, v))
        
        if len(current_set) > len(best_set):
            best_set = current_set
    
    # Phase 2: Local search - try to improve by removing one edge and adding another
    for _ in range(20):
        if time.time() > deadline:
            break
        improved = False
        for e_remove in list(best_set):
            if time.time() > deadline:
                break
            temp_set = set(best_set)
            temp_set.discard(e_remove)
            for e_add in edges:
                if e_add in temp_set:
                    continue
                if can_add(e_add[0], e_add[1], temp_set):
                    new_set = set(temp_set)
                    new_set.add(e_add)
                    if len(new_set) > len(best_set):
                        best_set = new_set
                        improved = True
                        break
            if improved:
                break
    
    # Phase 3: More greedy iterations
    while time.time() < deadline:
        random.shuffle(edges)
        current_set = set()
        for u, v in edges:
            if can_add(u, v, current_set):
                current_set.add((u, v))
        
        if len(current_set) > len(best_set):
            best_set = current_set
    
    # Convert to list of lists
    return [[u, v] for u, v in best_set]
