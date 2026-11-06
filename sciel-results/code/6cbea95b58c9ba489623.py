import time
import sys

def construct():
    start_time = time.time()
    deadline = 60.0
    
    N = 256
    # Build graph
    # adj[i] is a set of neighbors of i (0-indexed, so i corresponds to i+1)
    adj = [set() for _ in range(N)]
    
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            if (a * b) % (a + b) == 0:
                adj[i].add(j)
                adj[j].add(i)
    
    # Maximum Independent Set using Branch and Bound
    # We use a simple recursive search with bitmasks for speed
    # N=256 is too large for a single 256-bit int in some contexts? 
    # Python handles big ints natively.
    
    best_size = 0
    best_set = set()
    
    # Order nodes by degree (descending) to help pruning?
    # Or just use 0..N-1.
    
    def mis_search(vertices, current_size, current_set):
        nonlocal best_size, best_set
        nonlocal deadline
        if time.time() > deadline:
            return
        
        if not vertices:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            return
        
        # Upper bound: current_size + len(vertices)
        if current_size + len(vertices) <= best_size:
            return
            
        # Pick a vertex with highest degree in the induced subgraph?
        # For simplicity, pick the first one, or one with max degree in 'vertices'
        # Let's pick v with max degree in the subgraph induced by 'vertices'
        
        # To speed up, we can use a simple heuristic: pick v in vertices
        # such that v has max degree in the subgraph.
        # Or just pick the first one.
        
        # Let's try picking the vertex with the highest degree in the remaining graph.
        # This might be expensive to compute every time.
        # Given N=256, maybe just picking the first is fine if pruning is good.
        # But let's try to be smart.
        
        # Actually, a standard MIS algorithm picks a vertex v, then branches on:
        # 1. Include v: remove v and its neighbors from vertices.
        # 2. Exclude v: remove v from vertices.
        
        # Let's use a bitmask representation for vertices.
        # But Python sets are fine for N=256.
        
        # Heuristic: Pick v in vertices with max degree in the subgraph induced by vertices.
        v = None
        max_deg = -1
        for u in vertices:
            deg = len(adj[u].intersection(vertices))
            if deg > max_deg:
                max_deg = deg
                v = u
                
        # Branch 1: Include v
        # New vertices = vertices \ ({v} U N(v))
        new_vertices_include = vertices - {v} - adj[v]
        current_set.add(v)
        mis_search(new_vertices_include, current_size + 1, current_set)
        current_set.remove(v)
        
        # Branch 2: Exclude v
        new_vertices_exclude = vertices - {v}
        mis_search(new_vertices_exclude, current_size, current_set)

    initial_vertices = set(range(N))
    mis_search(initial_vertices, 0, set())
    
    # Convert back to 1-indexed values
    result = [v + 1 for v in best_set]
    
    # If no result found (should not happen), return seed
    if not result:
        return list(range(1, 257, 2))
        
    return result

if __name__ == "__main__":
    # Test
    res = construct()
    print(len(res))
