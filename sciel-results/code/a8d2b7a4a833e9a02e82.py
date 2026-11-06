import random
import time
import sys
sys.setrecursionlimit(10000)

def has_c5(n, adj):
    """
    Check if graph has a C5. Returns (True, cycle) or (False, []).
    adj: list of sets of neighbors.
    """
    for u in range(n):
        N_u = adj[u]
        # Get list of neighbors
        neigh = list(N_u)
        for i in range(len(neigh)):
            v = neigh[i]
            for j in range(i+1, len(neigh)):
                w = neigh[j]
                # We need a path v-a-b-w in G-u
                # a in N(v) \ {u, w}
                # b in N(w) \ {u, v}
                # edge (a,b)
                # a != b
                cand_a = [x for x in adj[v] if x != u and x != w]
                cand_b = [y for y in adj[w] if y != u and y != v]
                
                # Optimization: if len(cand_a) * len(cand_b) is large, use set intersection?
                # For each a in cand_a, check if any neighbor of a is in cand_b
                # and that neighbor is not v or w (already handled in cand_b def? no, cand_b excludes v and w)
                # We need edge (a,b).
                for a in cand_a:
                    # Neighbors of a
                    Na = adj[a]
                    # Check intersection of Na and cand_b
                    # Also a != b is required.
                    # If a is in Na? No, simple graph.
                    for b in cand_b:
                        if b in Na:
                            # Check distinctness
                            # u, v, w are distinct (v,w in N_u, v!=w)
                            # a in N(v), a!=v. a!=u, a!=w.
                            # b in N(w), b!=w. b!=u, b!=v.
                            # a != b checked.
                            # So u, v, a, b, w are distinct.
                            return True, [u, v, a, b, w]
    return False, []

def find_independent_set_9(n, adj):
    """
    Find an independent set of size 9. Returns the set or None.
    Uses branch and bound.
    """
    def backtrack(remaining, current):
        if len(current) == 9:
            return current[:]
        if not remaining:
            return None
        # Upper bound: current + remaining < 9
        if len(current) + len(remaining) < 9:
            return None
        
        # Pick a vertex
        v = remaining[0]
        rest = remaining[1:]
        
        # Option 1: Include v
        # New remaining: rest \ N[v]
        new_remaining = [u for u in rest if u not in adj[v] and u != v]
        res = backtrack(new_remaining, current + [v])
        if res:
            return res
        
        # Option 2: Exclude v
        res = backtrack(rest, current)
        if res:
            return res
            
        return None

    return backtrack(list(range(n)), [])

def construct():
    start_time = time.time()
    timeout = 115  # Leave 5s margin
    
    # Try to construct graphs for n from 35 down to 29
    for n in range(35, 28, -1):
        # Multiple attempts for each n
        for attempt in range(100):
            if time.time() - start_time > timeout:
                break
                
            # Initialize graph
            # Density heuristic: need high density to keep alpha low.
            # For n=35, p=0.5 is a good start.
            p = 0.5
            adj = [set() for _ in range(n)]
            edges = []
            
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        adj[i].add(j)
                        adj[j].add(i)
                        edges.append([i, j])
            
            # Repair loop
            valid = False
            for step in range(1000):
                if time.time() - start_time > timeout:
                    break
                
                # Check C5
                c5_found, cycle = has_c5(n, adj)
                if c5_found:
                    # Remove a random edge from the cycle
                    # Cycle is [u, v, a, b, w]
                    # Edges: (u,v), (v,a), (a,b), (b,w), (w,u)
                    # Pick a random edge to remove
                    e_idx = random.randint(0, 4)
                    u1 = cycle[e_idx]
                    u2 = cycle[(e_idx+1)%5]
                    
                    # Remove edge
                    adj[u1].remove(u2)
                    adj[u2].remove(u1)
                    # Remove from edges list
                    # Find and remove
                    for k in range(len(edges)):
                        if (edges[k][0] == u1 and edges[k][1] == u2) or (edges[k][0] == u2 and edges[k][1] == u1):
                            edges.pop(k)
                            break
                    continue
                
                # Check Independent Set
                ind_set = find_independent_set_9(n, adj)
                if ind_set is None:
                    valid = True
                    break
                else:
                    # Add an edge to break the independent set
                    # Pick two random vertices from the independent set
                    i1 = random.randint(0, len(ind_set)-1)
                    i2 = random.randint(0, len(ind_set)-1)
                    if i1 == i2:
                        continue
                    v1 = ind_set[i1]
                    v2 = ind_set[i2]
                    
                    # Add edge if not present
                    if v2 not in adj[v1]:
                        adj[v1].add(v2)
                        adj[v2].add(v1)
                        edges.append([v1, v2])
                    else:
                        # Should not happen if ind_set is independent
                        break
            
            if valid:
                return {'n': n, 'edges': edges}
    
    # Fallback to seed
    return {'n': 28, 'edges': [[4 * b + i, 4 * b + j] for b in range(7) for i in range(4) for j in range(i + 1, 4)]}
