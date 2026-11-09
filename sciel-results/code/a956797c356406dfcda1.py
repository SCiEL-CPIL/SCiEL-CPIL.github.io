import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 10s margin
    
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    # Adjacency matrix as list of sets for efficiency
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i].add(j)
                adj[j].add(i)
                
    best_n = n
    best_adj = [s.copy() for s in adj]
    
    # Try to extend to 42 vertices
    target_n = 42
    
    # Precompute 4-cliques and 4-independent sets for constraint checking
    # This might be expensive for 37 vertices, but let's see.
    # Actually, checking constraints on the fly is better.
    
    def is_valid_extension(adj, n, new_v_adj):
        # Check if adding vertex n with adjacency new_v_adj (set of indices < n) creates K5 or independent 5
        # K5 check: For any 4-clique {a,b,c,d} in old graph, if new_v_adj contains all of them, invalid.
        # Independent 5 check: For any 4-independent set {a,b,c,d} in old graph, if new_v_adj contains none of them, invalid.
        
        # We need to iterate over all 4-subsets of old vertices.
        # C(37, 4) = 66045. This is small enough.
        
        old_n = n
        # Get all 4-subsets
        # Optimization: Only check subsets that are close to being cliques/independent sets
        
        # For K5:
        for i in range(old_n):
            if i not in new_v_adj: continue
            for j in range(i+1, old_n):
                if j not in new_v_adj: continue
                if j not in adj[i]: continue
                for k in range(j+1, old_n):
                    if k not in new_v_adj: continue
                    if k not in adj[i] or k not in adj[j]: continue
                    for l in range(k+1, old_n):
                        if l not in new_v_adj: continue
                        if l not in adj[i] or l not in adj[j] or l not in adj[k]: continue
                        return False # Found K5
        
        # For Independent 5:
        for i in range(old_n):
            if i in new_v_adj: continue
            for j in range(i+1, old_n):
                if j in new_v_adj: continue
                if j in adj[i]: continue # Must be non-edge
                for k in range(j+1, old_n):
                    if k in new_v_adj: continue
                    if k in adj[i] or k in adj[j]: continue
                    for l in range(k+1, old_n):
                        if l in new_v_adj: continue
                        if l in adj[i] or l in adj[j] or l in adj[k]: continue
                        return False # Found Independent 5
                        
        return True

    # Greedy / Random search for extension
    # We will try to add vertices one by one.
    # For each new vertex, we try random subsets of old vertices as its neighborhood.
    
    current_adj = [s.copy() for s in best_adj]
    current_n = best_n
    
    for new_v in range(current_n, target_n):
        if time.time() > deadline:
            break
            
        found = False
        # Try multiple random configurations for the new vertex
        for attempt in range(500):
            if time.time() > deadline:
                break
            # Random subset of [0, current_n)
            # Bias towards roughly half connections
            new_v_adj = set()
            for i in range(current_n):
                if random.random() < 0.5:
                    new_v_adj.add(i)
                    
            if is_valid_extension(current_adj, current_n, new_v_adj):
                # Add the vertex
                current_adj.append(new_v_adj)
                for i in new_v_adj:
                    current_adj[i].add(new_v)
                current_n += 1
                found = True
                break
        
        if not found:
            # Try a few more random restarts or different biases?
            # Or just break
            break
            
        best_n = current_n
        best_adj = [s.copy() for s in current_adj]

    # Convert to edge list
    edges = []
    for i in range(best_n):
        for j in best_adj[i]:
            if i < j:
                edges.append((i, j))
                
    return best_n, edges
