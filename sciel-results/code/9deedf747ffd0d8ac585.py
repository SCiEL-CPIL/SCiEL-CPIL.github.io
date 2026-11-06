import time
import random

def construct():
    n = 40
    m = 401
    start_time = time.time()
    deadline = start_time + 115 # 5s margin
    
    # Initialize random graph
    edges = set()
    while len(edges) < m:
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u != v:
            edges.add((min(u,v), max(u,v)))
            
    # Convert to masks
    masks = [0] * n
    for u, v in edges:
        masks[u] |= 1 << v
        masks[v] |= 1 << u
        
    def get_max_book():
        max_b = 0
        for u, v in edges:
            common = masks[u] & masks[v]
            b = common.bit_count()
            if b > max_b:
                max_b = b
        return max_b
        
    current_score = get_max_book()
    best_score = current_score
    best_edges = set(edges)
    
    # Precompute non-edges? Or just pick random and check.
    # n=40, total pairs = 780.
    # non_edges = 780 - 401 = 379.
    
    no_improve_count = 0
    while time.time() < deadline:
        # Pick random edge to remove
        u, v = random.choice(list(edges))
        # Pick random non-edge to add
        # To avoid checking many times, we can pick random pair and check if in edges
        # Or maintain list of non-edges. Updating list is O(1) if we swap with last.
        # Let's just pick random pair. Probability of hit is 401/780 ~ 50%.
        # So ~2 tries on average.
        
        while True:
             x = random.randint(0, n-1)
             y = random.randint(0, n-1)
             if x == y: continue
             x, y = min(x,y), max(x,y)
             if (x,y) not in edges:
                 break
                 
        # Perform swap
        edges.remove((u,v))
        edges.add((x,y))
        
        # Update masks
        masks[u] &= ~(1 << v)
        masks[v] &= ~(1 << u)
        masks[x] |= 1 << y
        masks[y] |= 1 << x
        
        new_score = get_max_book()
        
        if new_score <= current_score:
            if new_score < best_score:
                best_score = new_score
                best_edges = set(edges)
                no_improve_count = 0
            else:
                # Accept if equal to move around
                if random.random() < 0.5: 
                    current_score = new_score
                else:
                    # Revert
                    edges.remove((x,y))
                    edges.add((u,v))
                    masks[x] &= ~(1 << y)
                    masks[y] &= ~(1 << x)
                    masks[u] |= 1 << v
                    masks[v] |= 1 << u
        else:
            # Revert
            edges.remove((x,y))
            edges.add((u,v))
            masks[x] &= ~(1 << y)
            masks[y] &= ~(1 << x)
            masks[u] |= 1 << v
            masks[v] |= 1 << u
            no_improve_count += 1
            
            # If stuck, restart with new random graph?
            # Or just continue.
            if no_improve_count > 10000:
                 # Re-initialize
                 edges = set()
                 while len(edges) < m:
                     u = random.randint(0, n-1)
                     v = random.randint(0, n-1)
                     if u != v:
                         edges.add((min(u,v), max(u,v)))
                 masks = [0] * n
                 for u, v in edges:
                     masks[u] |= 1 << v
                     masks[v] |= 1 << u
                 current_score = get_max_book()
                 no_improve_count = 0

    # Convert best_edges to list of lists
    return [[u, v] for u, v in best_edges]
