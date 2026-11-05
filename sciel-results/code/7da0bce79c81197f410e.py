import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 285  # Safety margin before 330s kill

    def time_left():
        return deadline - time.time()

    # 1. Generate Paley 37 as seed
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    
    # We will work with 42 vertices initially (Exoo bound)
    n_target = 42
    # adj[i] is a set of neighbors of vertex i
    adj = [set() for _ in range(n_target)]
    
    # Fill first 37x37 with Paley 37
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i].add(j)
                adj[j].add(i)
                
    # Initialize remaining vertices (37-41) with random edges
    random.seed(42)
    for i in range(q, n_target):
        for j in range(i + 1, n_target):
            if random.random() < 0.5:
                adj[i].add(j)
                adj[j].add(i)
            # Edges to existing vertices
            for k in range(i):
                if random.random() < 0.5:
                    adj[i].add(k)
                    adj[k].add(i)

    # Helper to check for K5 and independent 5 in a subset of vertices
    # Returns True if a K5 or Ind5 is found in the induced subgraph on 'vertices'
    def has_bad_subgraph(vertices):
        """
        Checks if the induced subgraph on 'vertices' contains a K5 or an independent 5.
        'vertices' is a list of vertex indices.
        """
        n_v = len(vertices)
        if n_v < 5:
            return False
        
        # Precompute adjacency within the subset for speed? 
        # Or just use global adj sets.
        
        # Check for K5
        # Iterate through all 5-combinations? C(42,5) is ~850k. 
        # This is called frequently in SA, so we need it to be fast or incremental.
        # For now, let's write a recursive clique finder.
        
        def find_clique(nodes, k):
            if k == 1:
                return True
            if k == 2:
                for i in range(len(nodes)):
                    for j in range(i+1, len(nodes)):
                        if nodes[j] in adj[nodes[i]]:
                            return True
                return False
            if k == 3:
                for i in range(len(nodes)):
                    for j in range(i+1, len(nodes)):
                        if nodes[j] in adj[nodes[i]]:
                            common = [x for x in nodes[j+1:] if x in adj[nodes[i]] and x in adj[nodes[j]]]
                            if common:
                                return True
                return False
            if k == 4:
                for i in range(len(nodes)):
                    for j in range(i+1, len(nodes)):
                        if nodes[j] in adj[nodes[i]]:
                            common = [x for x in nodes[j+1:] if x in adj[nodes[i]] and x in adj[nodes[j]]]
                            if find_clique(common, 3):
                                return True
                return False
            if k == 5:
                for i in range(len(nodes)):
                    for j in range(i+1, len(nodes)):
                        if nodes[j] in adj[nodes[i]]:
                            common = [x for x in nodes[j+1:] if x in adj[nodes[i]] and x in adj[nodes[j]]]
                            if find_clique(common, 4):
                                return True
                return False
            return False
        
        if find_clique(vertices, 5):
            return True
            
        # Check for Independent 5 (Clique 5 in complement)
        def find_independent(nodes, k):
            if k == 1:
                return True
            if k == 2:
                for i in range(len(nodes)):
                    for j in range(i+1, len(nodes)):
                        if nodes[j] not in adj[nodes[i]]:
                            return True
                return False
            if k == 3:
                for i in range(len(nodes)):
                    for j in range(i+1, len(nodes)):
                        if nodes[j] not in adj[nodes[i]]:
                            common = [x for x in nodes[j+1:] if x not in adj[nodes[i]] and x not in adj[nodes[j]]]
                            if common:
                                return True
                return False
            if k == 4:
                for i in range(len(nodes)):
                    for j in range(i+1, len(nodes)):
                        if nodes[j] not in adj[nodes[i]]:
                            common = [x for x in nodes[j+1:] if x not in adj[nodes[i]] and x not in adj[nodes[j]]]
                            if find_independent(common, 3):
                                return True
                return False
            if k == 5:
                for i in range(len(nodes)):
                    for j in range(i+1, len(nodes)):
                        if nodes[j] not in adj[nodes[i]]:
                            common = [x for x in nodes[j+1:] if x not in adj[nodes[i]] and x not in adj[nodes[j]]]
                            if find_independent(common, 4):
                                return True
                return False
            return False
            
        if find_independent(vertices, 5):
            return True
            
        return False

    # Initial cost check: Is the initial random 42-vertex graph valid?
    # If yes, return immediately.
    if not has_bad_subgraph(list(range(n_target))):
        edges = [(i, j) for i in range(n_target) for j in range(i + 1, n_target) if j in adj[i]]
        return n_target, edges

    # If not valid, we need to repair it or start SA.
    # However, checking has_bad_subgraph is expensive.
    # Let's use a Simulated Annealing approach with a simpler cost function:
    # Cost = Number of K5s + Number of Ind5s.
    # But counting all K5s is hard.
    
    # Alternative: Use the "Add Vertex" strategy from Inspiration 2.
    # Start with Paley 37 (valid).
    # Try to add vertices one by one until we can't anymore or time runs out.
    
    # Reset to Paley 37
    n = q
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i].add(j)
                adj[j].add(i)
                
    best_n = n
    best_adj = [set(s) for s in adj]
    
    # Function to check if a new vertex with neighbor set 'new_neigh' can be added
    def can_add_vertex(current_n, new_neigh):
        # 1. new_neigh must not contain a K4
        if len(new_neigh) < 4:
            clique_ok = True
        else:
            # Check for K4 in new_neigh
            # Reusing find_clique logic but on the set
            nodes = list(new_neigh)
            clique_ok = not find_clique(nodes, 4)
            if not clique_ok:
                return False
                
        # 2. Non-neighbors must not contain an Independent 4
        non_neigh = set(range(current_n)) - new_neigh
        if len(non_neigh) < 4:
            indep_ok = True
        else:
            nodes = list(non_neigh)
            indep_ok = not find_independent(nodes, 4)
            if not indep_ok:
                return False
                
        return True

    # Greedy/Randomized Vertex Addition
    for target_n in range(n + 1, 46): # Upper bound is 46, so max 45
        if time_left() < 5:
            break
            
        found = False
        attempts = 0
        max_attempts = 50000 # Limit per vertex to avoid getting stuck
        
        while attempts < max_attempts and not found:
            if time_left() < 5:
                break
                
            # Random subset of existing vertices
            # Bias towards 50% connection, but maybe tune based on density
            # Paley 37 has degree ~18. So 50% is reasonable.
            new_neigh = set(v for v in range(target_n) if random.random() < 0.5)
            
            if can_add_vertex(target_n, new_neigh):
                # Add the vertex
                adj.append(set(new_neigh))
                for v in new_neigh:
                    adj[v].add(target_n)
                
                best_n = target_n + 1
                best_adj = [set(s) for s in adj]
                found = True
                break
            
            attempts += 1
            
        if not found:
            # If we can't add a vertex easily, maybe try SA on the current graph to improve structure?
            # Or just stop?
            # Let's try a few random restarts of the addition process for this specific n.
            # Actually, if we fail to add vertex n, we might still be able to add n+1 if we rearranged edges.
            # But rearranging edges is complex.
            # Let's try to run a short SA on the current best graph to see if we can make it "more expandable".
            # Or just break and return best.
            
            # Let's try one more thing: Local Search to fix the current graph if it's invalid?
            # No, we are building incrementally, so it's always valid.
            
            # If we fail to add vertex k, we can try to add vertex k again with different random seeds?
            # Or just move on? No, we need consecutive vertices.
            # Let's break the loop if we can't extend.
            break

    # Return the best valid graph found
    if best_n > q:
        edges = [(i, j) for i in range(best_n) for j in range(i + 1, best_n) if j in best_adj[i]]
        return best_n, edges
    else:
        # Fallback to Paley 37
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if j in best_adj[i]]
        return q, edges

