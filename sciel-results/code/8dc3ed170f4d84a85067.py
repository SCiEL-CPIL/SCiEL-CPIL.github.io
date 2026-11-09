import random
import time

def construct():
    """
    Constructs a C4-free edge set of Q7 maximizing the number of edges.
    Uses randomized local search (iterated greedy).
    """
    d = 7
    n_vertices = 2 ** d
    n_edges = d * (2 ** (d - 1))
    
    # Precompute neighbors for each vertex
    # neighbors[v] is a list of (neighbor, axis)
    neighbors = [[] for _ in range(n_vertices)]
    for v in range(n_vertices):
        for i in range(d):
            u = v ^ (1 << i)
            if u > v:
                neighbors[v].append((u, i))
                neighbors[u].append((v, i))
            # Actually, for the check, we need to know for a given edge (u,v) with axis i,
            # which other edges are in the faces.
            # Face for edge (u,v) axis i and axis j:
            # Vertices: u, v, u^j, v^j
            # Edges: (u,v), (u, u^j), (v, v^j), (u^j, v^j)
            
    # We will represent the edge set as a set of tuples (min(u,v), max(u,v))
    # And also a boolean array for fast lookup? 
    # Edge index: 
    # We can map (u,v) to an index.
    # Since it's a hypercube, we can just use a set of tuples. 
    # Lookup is O(1).
    
    def is_valid_add(edge_set, u, v, axis):
        """
        Check if adding edge (u,v) with axis 'axis' creates a C4.
        It creates a C4 if there exists an axis j != axis such that:
        (u, u^j) is in edge_set
        (v, v^j) is in edge_set
        (u^j, v^j) is in edge_set
        """
        for j in range(d):
            if j == axis:
                continue
            u_j = u ^ (1 << j)
            v_j = v ^ (1 << j)
            
            # Check the other 3 edges of the face
            e1 = (min(u, u_j), max(u, u_j))
            e2 = (min(v, v_j), max(v, v_j))
            e3 = (min(u_j, v_j), max(u_j, v_j))
            
            if e1 in edge_set and e2 in edge_set and e3 in edge_set:
                return False
        return True

    def get_axis(u, v):
        diff = u ^ v
        # diff is a power of 2
        return diff.bit_length() - 1

    # Generate all edges with their axes
    all_edges = []
    for u in range(n_vertices):
        for v in range(u + 1, n_vertices):
            if (u ^ v).bit_count() == 1:
                axis = get_axis(u, v)
                all_edges.append(((u, v), axis))
                
    # Shuffle edges for randomization
    random.shuffle(all_edges)

    best_edges = set()
    best_size = 0
    
    start_time = time.time()
    time_limit = 110 # 10s margin
    
    # Initial greedy pass
    for edge, axis in all_edges:
        if is_valid_add(best_edges, edge[0], edge[1], axis):
            best_edges.add(edge)
            
    current_size = len(best_edges)
    if current_size > best_size:
        best_size = current_size
        
    # Local Search
    # We maintain a list of missing edges to try adding
    # And a set of current edges
    
    # To speed up "try to add", we can just iterate through a shuffled list of missing edges
    # But maintaining the list of missing edges is expensive.
    # Instead, we can just pick a random edge from all_edges and check if it's in the set.
    # If not, try to add.
    
    # A better strategy: 
    # 1. Try to add random edges.
    # 2. If stuck for a while, remove some edges.
    
    stuck_counter = 0
    max_stuck = 1000
    
    while time.time() - start_time < time_limit:
        # Try to add a random edge
        # Pick a random edge from all_edges
        edge, axis = random.choice(all_edges)
        
        if edge in best_edges:
            # Already present, do nothing or try another
            continue
            
        if is_valid_add(best_edges, edge[0], edge[1], axis):
            best_edges.add(edge)
            stuck_counter = 0
            if len(best_edges) > best_size:
                best_size = len(best_edges)
        else:
            stuck_counter += 1
            
        if stuck_counter > max_stuck:
            # We are stuck. Remove some random edges to allow new configurations.
            # Remove 3-5 random edges
            num_remove = random.randint(3, 6)
            edges_list = list(best_edges)
            random.shuffle(edges_list)
            for e in edges_list[:num_remove]:
                best_edges.remove(e)
            stuck_counter = 0
            
    return [list(e) for e in best_edges]
