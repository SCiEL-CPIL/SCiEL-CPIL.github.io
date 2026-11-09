import time
import random
import itertools

N = 16
ALL_EDGES = [(i, j) for i in range(N) for j in range(i+1, N)]
EDGE_SET = set(ALL_EDGES)

# Precompute all 8-vertex subsets
SUBSETS_8 = list(itertools.combinations(range(N), 8))

def find_q3(edge_set):
    """
    Finds a Q3 subgraph in the graph defined by edge_set.
    Returns a list of 12 edges forming the Q3, or None.
    """
    # Shuffle subsets for randomness
    random.shuffle(SUBSETS_8)
    
    for verts in SUBSETS_8:
        # Build induced subgraph info
        # Map global vertex to local index 0..7
        # We just need to check if the induced subgraph contains Q3
        
        # Count edges in induced subgraph
        edge_count = 0
        # We can just check degrees and bipartiteness
        
        # 1. Check if induced subgraph is bipartite
        # Build adjacency list for these 8 vertices
        adj = {v: [] for v in verts}
        for u in verts:
            for v in verts:
                if u < v and (u, v) in edge_set:
                    adj[u].append(v)
                    adj[v].append(u)
                    edge_count += 1
        
        if edge_count < 12:
            continue
            
        # Check bipartiteness
        color = {}
        is_bip = True
        for start in verts:
            if start in color:
                continue
            color[start] = 0
            queue = [start]
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if v not in color:
                        color[v] = color[u] ^ 1
                        queue.append(v)
                    elif color[v] == color[u]:
                        is_bip = False
                        break
            if not is_bip:
                break
        
        if not is_bip:
            continue
            
        # Get parts
        part_0 = [v for v in verts if color[v] == 0]
        part_1 = [v for v in verts if color[v] == 1]
        
        if len(part_0) < 4 or len(part_1) < 4:
            continue
            
        # Try all subsets of size 4
        # Ensure part_0 is the smaller one? No, Q3 is 4-4.
        # We can swap if needed, but let's just iterate.
        
        # Optimization: Check degrees in the induced bipartite graph
        # For a candidate Q3, every vertex must have degree 3 in the Q3.
        # So in the induced graph, the degree must be >= 3.
        
        # Pre-filter vertices in part_0 and part_1
        # deg(v) in induced graph
        
        # Iterate subsets
        # To speed up, we can iterate over combinations of part_0 of size 4
        for a_sub in itertools.combinations(part_0, 4):
            # Check if all a in a_sub have degree >= 3 in the full induced graph?
            # No, degree in the subgraph induced by a_sub and b_sub.
            # This depends on b_sub.
            
            # Just iterate b_sub
            for b_sub in itertools.combinations(part_1, 4):
                # Check if edges between a_sub and b_sub contain Q3
                # We need to check if there is a bijection between a_sub and Q3_Part_A
                # and b_sub and Q3_Part_B such that all edges match.
                
                # Simple check:
                # For each a in a_sub, count neighbors in b_sub.
                # If any count < 3, skip.
                valid = True
                for a in a_sub:
                    deg = 0
                    for b in b_sub:
                        if (a, b) in edge_set or (b, a) in edge_set:
                            deg += 1
                    if deg < 3:
                        valid = False
                        break
                if not valid:
                    continue
                
                for b in b_sub:
                    deg = 0
                    for a in a_sub:
                        if (a, b) in edge_set or (b, a) in edge_set:
                            deg += 1
                    if deg < 3:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Now check specific Q3 structure
                # We can just check if the bipartite graph on a_sub, b_sub is isomorphic to a supergraph of Q3.
                # Since degrees are >=3, and max is 4, the missing edges form a matching (if deg=3) or empty (if deg=4).
                # Q3 is K4,4 minus a perfect matching.
                # So we need to check if the missing edges form a perfect matching.
                
                # Build missing edges set
                missing = set()
                for a in a_sub:
                    for b in b_sub:
                        if not ((a, b) in edge_set or (b, a) in edge_set):
                            missing.add((a, b))
                
                # Check if missing is a perfect matching
                if len(missing) == 4:
                    # Check if it's a matching
                    a_used = set()
                    b_used = set()
                    is_matching = True
                    for (a, b) in missing:
                        if a in a_used or b in b_used:
                            is_matching = False
                            break
                        a_used.add(a)
                        b_used.add(b)
                    if is_matching and len(a_used) == 4:
                        # Found Q3!
                        # Return the edges of this Q3
                        q3_edges = []
                        for a in a_sub:
                            for b in b_sub:
                                if (a, b) in edge_set or (b, a) in edge_set:
                                    q3_edges.append((min(a,b), max(a,b)))
                        return q3_edges
                elif len(missing) == 0:
                    # K4,4 contains Q3? Yes.
                    # We can just pick any Q3 subgraph from K4,4.
                    # For simplicity, let's just return the edges of K4,4? No, must be 12 edges.
                    # We can construct a Q3 from K4,4 by removing a perfect matching.
                    # But we need to return the actual edges present in G.
                    # So we just need to find 12 edges that form Q3.
                    # Since it's K4,4, any Q3 subgraph works.
                    # Let's just pick a standard Q3 embedding.
                    # a_sub[0] connected to b_sub[1], b_sub[2], b_sub[3]
                    # etc.
                    # This is getting complicated.
                    # Just return the first 12 edges? No.
                    # Let's just say we found it and return a dummy list?
                    # No, we need the actual edges to remove one.
                    
                    # Let's just pick a random perfect matching from the edges and remove it from the list of 16 edges.
                    # The 16 edges are all (a,b) for a in a_sub, b in b_sub.
                    all_edges_ab = [(a, b) for a in a_sub for b in b_sub]
                    # Shuffle and pick 12?
                    # Actually, any 12 edges forming Q3.
                    # Let's just return all 16? No, function returns Q3 edges.
                    # We can just return a valid Q3 set.
                    # Let's hardcode a Q3 on 0..7 and map it.
                    # This is overkill.
                    # Just return None and continue? No, we found a K4,4 which contains Q3.
                    # We should return a valid set of 12 edges.
                    
                    # Let's just pick 3 neighbors for each a.
                    # This is a valid Q3.
                    q3_edges = []
                    for i, a in enumerate(a_sub):
                        # Connect to all b except b_sub[i]
                        for j, b in enumerate(b_sub):
                            if i != j:
                                q3_edges.append((min(a,b), max(a,b)))
                    return q3_edges

    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Start with complete graph
    current_edges = set(ALL_EDGES)
    
    # Greedy removal
    while time.time() < deadline:
        q3_edges = find_q3(current_edges)
        if q3_edges is None:
            break
        
        # Remove one edge from q3_edges
        # Strategy: Remove the edge that appears in the most other Q3s?
        # We don't know that.
        # Just remove a random one.
        e_to_remove = random.choice(q3_edges)
        current_edges.remove(e_to_remove)
        
        # If we are getting close to deadline, force return
        if time.time() > deadline - 10.0:
            break
            
    # Convert to list of lists
    result = [[u, v] for u, v in current_edges]
    return result
