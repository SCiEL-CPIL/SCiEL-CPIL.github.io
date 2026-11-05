import random
import time
import math

def construct():
    n = 16
    target_edges = 65
    
    # Initialize with K_{8,8} + 1 edge
    # Vertices 0-7 in part A, 8-15 in part B
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append((u, v))
    edges.append((0, 1))
    
    # Convert to set for fast lookup
    edge_set = set(edges)
    
    def get_score(edge_set):
        # Calculate degrees
        deg = [0] * n
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
            
        # Find triangles and max degree sum
        max_sum = 0
        # For each pair (u, v) that is an edge, check common neighbors
        # To avoid duplicates, only consider u < v
        # And for triangle (u, v, w), we can just check if u<v and v<w? No, just check all triangles.
        # A triangle is a set {u, v, w}.
        # We can iterate over all triples? 16^3 = 4096. That's fine.
        # Or iterate over edges.
        
        # Let's iterate over edges to be faster if sparse, but n is small.
        # Iterate u, v, w with u < v < w
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) not in edge_set and (v, u) not in edge_set:
                    continue
                for w in range(v + 1, n):
                    if (v, w) not in edge_set and (w, v) not in edge_set:
                        continue
                    if (u, w) not in edge_set and (w, u) not in edge_set:
                        continue
                    # Triangle (u, v, w) found
                    s = deg[u] + deg[v] + deg[w]
                    if s > max_sum:
                        max_sum = s
        return max_sum

    current_edges = set(edge_set)
    current_score = get_score(current_edges)
    best_edges = set(current_edges)
    best_score = current_score
    
    # Simulated Annealing
    start_time = time.time()
    timeout = 110.0 # Safety margin
    T = 1.0
    cooling = 0.9999
    
    while time.time() - start_time < timeout:
        # Generate a move: swap an edge
        # Pick a random edge to remove
        if not current_edges:
            break
        remove_edge = random.choice(list(current_edges))
        
        # Pick a random non-edge to add
        # Generate random pairs until we find a non-edge
        # n=16, total pairs = 120. Edges = 65. Non-edges = 55.
        # Expected 120/55 ~ 2.2 tries
        add_edge = None
        for _ in range(10): # Limit retries
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if (u, v) not in current_edges:
                add_edge = (u, v)
                break
        
        if add_edge is None:
            continue
            
        if remove_edge == add_edge:
            continue

        # Apply move
        new_edges = set(current_edges)
        new_edges.discard(remove_edge)
        new_edges.add(add_edge)
        
        # Check if we accidentally removed the only instance? 
        # Since it's a set, discard is safe.
        # But we must ensure we didn't remove an edge that wasn't there? 
        # remove_edge was from current_edges, so it's there.
        # add_edge was not in current_edges, so it's new.
        
        new_score = get_score(new_edges)
        
        # Acceptance criteria
        if new_score < current_score:
            current_edges = new_edges
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_edges = new_edges
        else:
            # Simulated annealing step
            delta = new_score - current_score
            if delta > 0:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_edges = new_edges
                    current_score = new_score
            else:
                # Should not happen if logic is correct, but safe
                current_edges = new_edges
                current_score = new_score
        
        T *= cooling
    
    # Convert best_edges set to list of lists for return
    # Ensure consistent ordering
    best_list = sorted([list(e) for e in best_edges])
    
    return {'n': n, 'edges': best_list}
