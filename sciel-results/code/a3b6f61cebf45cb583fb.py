import random
import time
from itertools import combinations, permutations

def construct():
    n = 15
    # Precompute all simple paths of length 3 (4 vertices s, u, v, t)
    # A path is defined by the sequence of vertices.
    # Edges are (s,u), (u,v), (v,t).
    # Constraint: deg(s) != deg(t) in the current graph.
    
    # Generate all permutations of 4 distinct vertices
    # There are 15 * 14 * 13 * 12 = 32760 such paths.
    # We store (s, t) for each path.
    # Note: The path s-u-v-t is the same as t-v-u-s in terms of endpoints.
    # But the set of edges required is {(s,u), (u,v), (v,t)}.
    # We need to check if these 3 edges exist in the graph.
    
    # To make checking fast, we can just iterate through all potential paths
    # and check if the edges exist.
    
    # Better: Precompute list of (s, u, v, t) tuples.
    paths = []
    vertices = list(range(n))
    for s in vertices:
        for u in vertices:
            if u == s: continue
            for v in vertices:
                if v == s or v == u: continue
                for t in vertices:
                    if t == s or t == u or t == v: continue
                    # Path s-u-v-t
                    # Endpoints s, t
                    paths.append((s, u, v, t))
                    
    # There are 32760 paths.
    # To check validity of a graph (adjacency matrix or set of edges):
    # 1. Compute degrees.
    # 2. For each path (s, u, v, t) in paths:
    #    If edges (s,u), (u,v), (v,t) exist:
    #       If deg[s] == deg[t]: return False
    # 3. Return True
    
    def is_valid(edges_set, deg):
        # edges_set is a set of frozensets or tuples (u,v) with u<v
        # deg is a list of 15 ints
        for s, u, v, t in paths:
            # Check if edges exist
            # Helper to check edge existence
            if u < s:
                e1 = (u, s)
            else:
                e1 = (s, u)
            
            if v < u:
                e2 = (v, u)
            else:
                e2 = (u, v)
                
            if t < v:
                e3 = (v, t)
            else:
                e3 = (v, t)
                
            # Wait, t and v order.
            if t < v:
                e3 = (t, v)
            else:
                e3 = (v, t)
                
            if e1 in edges_set and e2 in edges_set and e3 in edges_set:
                if deg[s] == deg[t]:
                    return False
        return True

    def get_degree(edges_set):
        deg = [0] * n
        for u, v in edges_set:
            deg[u] += 1
            deg[v] += 1
        return deg

    def normalize_edge(u, v):
        if u > v:
            return (v, u)
        return (u, v)

    # Local Search
    best_edges = set()
    best_count = 0
    
    start_time = time.time()
    deadline = 85.0 # Safety margin
    
    # Initialize with K_{7,8}
    init_edges = set()
    for u in range(7):
        for v in range(7, 15):
            init_edges.add(normalize_edge(u, v))
            
    # Check init
    deg_init = get_degree(init_edges)
    if is_valid(init_edges, deg_init):
        best_edges = init_edges
        best_count = len(init_edges)
        
    # Local Search Loop
    while time.time() < deadline:
        # Random Restart Strategy
        # If we are stuck or just for diversity, sometimes restart
        # But let's try to improve current best first
        
        # 1. Try to add edges
        # Generate random non-edges
        all_edges = set()
        for i in range(n):
            for j in range(i+1, n):
                all_edges.add((i, j))
                
        current_edges = set(best_edges)
        current_deg = get_degree(current_edges)
        
        # Try adding random edges
        non_edges = list(all_edges - current_edges)
        random.shuffle(non_edges)
        
        added_any = False
        for e in non_edges:
            if time.time() > deadline: break
            u, v = e
            new_edges = current_edges | {e}
            # Update degrees
            new_deg = current_deg[:]
            new_deg[u] += 1
            new_deg[v] += 1
            
            if is_valid(new_edges, new_deg):
                current_edges = new_edges
                current_deg = new_deg
                added_any = True
                # If we added an edge, we might be able to add more, 
                # but we should re-shuffle or continue?
                # Let's restart the shuffle loop to try other edges
                # Actually, just continue checking other non-edges
                # But degrees changed, so previous checks are invalid?
                # No, we only care if the NEW state is valid.
                # We are greedily adding.
                pass
        
        if len(current_edges) > best_count:
            best_edges = current_edges
            best_count = len(current_edges)
            
        # 2. Try swaps (remove one, add one)
        # This helps escape local minima where no single edge can be added
        if len(current_edges) > 0:
            current_edges_copy = set(current_edges)
            current_deg_copy = current_deg[:]
            
            # Try removing a random edge
            edge_list = list(current_edges_copy)
            random.shuffle(edge_list)
            
            improved = False
            for e_rem in edge_list:
                if time.time() > deadline: break
                
                u_rem, v_rem = e_rem
                # Remove e_rem
                temp_edges = current_edges_copy - {e_rem}
                temp_deg = current_deg_copy[:]
                temp_deg[u_rem] -= 1
                temp_deg[v_rem] -= 1
                
                # Try adding a random non-edge (from original non-edges or new ones)
                # The set of non-edges changed (e_rem is now a non-edge)
                # We can try adding e_rem back (no change) or something else
                # Let's try adding a random edge from all_edges not in temp_edges
                # To be efficient, pick a few candidates
                
                candidates = list(all_edges - temp_edges)
                # Exclude e_rem? No, adding it back is a no-op.
                # We want to add a DIFFERENT edge.
                # Filter out e_rem
                candidates = [c for c in candidates if c != e_rem]
                
                if not candidates: continue
                
                random.shuffle(candidates)
                
                # Try a few candidates
                for e_add in candidates[:10]: # Limit tries per removal
                    if time.time() > deadline: break
                    u_add, v_add = e_add
                    new_edges = temp_edges | {e_add}
                    new_deg = temp_deg[:]
                    new_deg[u_add] += 1
                    new_deg[v_add] += 1
                    
                    if is_valid(new_edges, new_deg):
                        # Valid swap found
                        current_edges = new_edges
                        current_deg = new_deg
                        improved = True
                        break # Break candidate loop, try next removal?
                        # Or break out of removal loop?
                        # If we improved, we should restart the search from this new state
                        break
                
                if improved:
                    if len(current_edges) > best_count:
                        best_edges = current_edges
                        best_count = len(current_edges)
                    break # Break removal loop, restart outer loop with new state
        
        # If no improvement in this iteration, maybe restart from scratch?
        # Or just continue.
        
        # Random Restart: With small probability, reset to random graph
        if random.random() < 0.05:
             current_edges = set()
             # Add random edges
             for e in all_edges:
                 if random.random() < 0.5:
                     current_edges.add(e)
             current_deg = get_degree(current_edges)
             if not is_valid(current_edges, current_deg):
                 # Try to fix by removing edges
                 while not is_valid(current_edges, current_deg) and current_edges:
                     e_rem = random.choice(list(current_edges))
                     current_edges.remove(e_rem)
                     u, v = e_rem
                     current_deg[u] -= 1
                     current_deg[v] -= 1
             if len(current_edges) > best_count:
                 best_edges = current_edges
                 best_count = len(current_edges)
                 # Continue optimizing this new start

    # Convert best_edges to list of lists
    # Ensure types are correct (int)
    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])
        
    return result

# Run construct to test? No, just define it.
# The evaluator calls construct().

if __name__ == "__main__":
    # Simple test
    pass
