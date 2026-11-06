import time
import random
import sys

def construct():
    """
    Construct a graph with no K5 and no independent 5-set.
    Aims to find a graph with n > 37 vertices.
    """
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin
    
    # 1. Initialize with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    # Adjacency matrix as list of sets for fast lookup
    adj = [set() for _ in range(q)]
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i].add(j)
                adj[j].add(i)
    
    n = q
    best_n = n
    best_adj = [set(s) for s in adj]

    # Helper to check if a set of vertices is a clique or independent set
    def is_clique(vertices, adj):
        for v in vertices:
            for u in vertices:
                if v != u and u not in adj[v]:
                    return False
        return True

    def is_independent(vertices, adj):
        for v in vertices:
            for u in vertices:
                if v != u and u in adj[v]:
                    return False
        return True

    # Precompute 4-cliques and 4-independent sets for the initial graph
    # This is expensive for large n, but n=37 is small.
    # Number of 4-subsets is C(37,4) ~ 66k. Very fast.
    
    def get_4_cliques(adj, n):
        cliques = []
        vertices = list(range(n))
        for i in range(n):
            for j in range(i+1, n):
                if j in adj[i]:
                    for k in range(j+1, n):
                        if k in adj[i] and k in adj[j]:
                            for l in range(k+1, n):
                                if l in adj[i] and l in adj[j] and l in adj[k]:
                                    cliques.append((i, j, k, l))
        return cliques

    def get_4_independent(adj, n):
        indeps = []
        vertices = list(range(n))
        for i in range(n):
            for j in range(i+1, n):
                if j not in adj[i]:
                    for k in range(j+1, n):
                        if k not in adj[i] and k not in adj[j]:
                            for l in range(k+1, n):
                                if l not in adj[i] and l not in adj[j] and l not in adj[k]:
                                    indeps.append((i, j, k, l))
        return indeps

    # We will try to add vertices one by one.
    # For each new vertex, we need to find a binary vector x of length n
    # such that:
    # 1. For every 4-clique C in current graph, not all x[i]=1 for i in C.
    # 2. For every 4-independent set I in current graph, not all x[i]=0 for i in I.
    
    # We use a simple randomized local search / backtracking for the new vertex.
    
    def try_add_vertex(adj, n):
        """
        Tries to add a new vertex to the graph defined by adj (size n).
        Returns True and updates adj if successful, False otherwise.
        """
        # Get constraints
        cliques = get_4_cliques(adj, n)
        indeps = get_4_independent(adj, n)
        
        # If there are no constraints, any x works. 
        # But there will always be some.
        
        # We want to find x in {0,1}^n.
        # Let's use a greedy approach with randomization and backtracking.
        # Or simply random restarts.
        
        # Heuristic: 
        # Start with x all 0s. 
        # Violations: 
        # - For each clique C, if all x[i]==1, violation. (Avoided by starting with 0s)
        # - For each indep I, if all x[i]==0, violation. (Many violations initially)
        
        # So we need to set some x[i] to 1 to break independent sets.
        # But setting x[i] to 1 might create clique violations.
        
        # Let's use a simple iterative improvement.
        # 1. Randomly initialize x.
        # 2. Calculate violations.
        # 3. Flip bit that reduces violations the most.
        
        max_attempts = 5000
        for attempt in range(max_attempts):
            # Random initialization
            x = [random.randint(0, 1) for _ in range(n)]
            
            # Check validity
            valid = True
            for c in cliques:
                if all(x[i] == 1 for i in c):
                    valid = False
                    break
            if valid:
                for i_set in indeps:
                    if all(x[i] == 0 for i in i_set):
                        valid = False
                        break
            
            if valid:
                # Success! Add the vertex
                for i in range(n):
                    if x[i] == 1:
                        adj[i].add(n)
                        adj.append(set([i])) # placeholder, will fix below
                # Actually, let's just return the vector
                return x
            
            # If not valid, try local search? 
            # For simplicity and speed, we just do random restarts.
            # If n is small, random restarts work well.
            # If n grows, it might fail.
            
            # To improve, we can bias the initialization.
            # The graph is roughly half edges. So random is okay.
            
            # Optimization: If we fail many times, we might need a smarter search.
            # But for n=37..42, the constraint graph is not too dense.
            
        return None

    # Main loop to add vertices
    while time.time() < deadline:
        # Re-evaluate constraints for current n
        # We need to find x for the new vertex n
        x = try_add_vertex(adj, n)
        if x is not None:
            # Add vertex n
            new_adj = [set(s) for s in adj]
            new_adj.append(set())
            for i in range(n):
                if x[i] == 1:
                    new_adj[i].add(n)
                    new_adj[n].add(i)
            adj = new_adj
            n += 1
            best_n = n
            best_adj = [set(s) for s in adj]
            
            # If we reach 42, we can stop or try for more (but 42 is known max)
            if n >= 42:
                break
        else:
            # Failed to add a vertex. 
            # We could try to modify the existing graph to make it possible?
            # That's much harder. 
            # For now, if we can't add, we stop.
            # But maybe we got stuck at 38?
            # We can try randomizing the *order* of bits or using a different heuristic.
            # Let's try a few more times with different random seeds or slight perturbations.
            # Actually, if try_add_vertex failed, it means no valid x was found in 5000 tries.
            # It's possible one exists.
            # Let's increase attempts for the first few steps?
            # Or break if we fail.
            
            # To be safe, if we fail, we return the best found so far.
            # But we might have found 38, 39, etc.
            # Let's try to "shake" the search.
            # We can't easily change the existing graph without breaking previous constraints.
            # So we just break.
            
            # However, if n is still 37, we haven't improved.
            # We should probably return the seed if we can't beat it.
            
            # Let's try one more thing: if we fail, we might have bad luck.
            # We can retry the same n with more attempts if time permits.
            # But try_add_vertex already did 5000.
            # Let's just break.
            break

    # Construct edge list
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
                
    return n, edges
