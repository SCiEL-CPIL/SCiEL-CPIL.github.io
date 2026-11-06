import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Fallback: Paley graph on 17 vertices
    # This is a known valid graph for R(4,6) > 17
    squares = {x * x % 17 for x in range(1, 17)}
    best_n = 17
    best_edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    
    # Try to find larger graphs, starting from the known record n=35 downwards
    # We prioritize n=35 as it is the published record.
    
    for n in range(35, 16, -1):
        if time.time() > deadline:
            break
            
        # Try multiple random restarts for this n
        found = False
        attempts = 0
        max_attempts = 20 if n == 35 else 50
        
        while time.time() < deadline - 10.0 and not found and attempts < max_attempts:
            attempts += 1
            
            # Random initialization
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            # Local Search / Repair
            success = local_search(n, adj, deadline)
            if success:
                # Extract edges
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
                
    return {'n': best_n, 'edges': best_edges}

def local_search(n, adj, deadline):
    """
    Attempts to repair the graph to be K4-free and I6-free.
    Returns True if a valid graph is found.
    """
    mask = (1 << n) - 1
    
    def has_k4():
        """
        Returns a K4 (u, v, w, x) if found, else None.
        """
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    # Common neighbors of u and v
                    common = adj[u] & adj[v]
                    # Remove u and v from common (they are not neighbors of themselves in adj mask? 
                    # adj[u] does not contain u. So common does not contain u or v.
                    # But common contains w, x such that w~u, w~v, x~u, x~v.
                    # We need w~x.
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        # Check if w is connected to any other vertex in common
                        # If adj[w] & common != 0, then there is x in common such that w~x.
                        if adj[w] & common:
                            return (u, v, w) # Just returning 3 vertices is enough to identify the violation
        return None

    def has_i6():
        """
        Checks for an independent set of size 6.
        Equivalent to finding a K6 in the complement graph.
        Returns a set of 6 vertices if found, else None.
        """
        # Build complement adjacency
        comp_adj = [mask ^ adj[i] ^ (1 << i) for i in range(n)]
        
        # Use Bron-Kerbosch with pivoting to find a clique of size 6
        # We can prune if we just need to find *any* clique of size 6.
        
        def bron_kerbosch(R, P, X, depth):
            if len(R) >= 6:
                return R[:]
            if not P:
                return None
            
            # Pivot selection: choose v from P union X with max degree in P
            # For simplicity and speed in this specific case, we can just pick the first one or a heuristic one.
            # A good pivot reduces recursion.
            v = -1
            max_deg = -1
            union_px = P | X
            c = union_px
            while c:
                lsb = c & -c
                u = lsb.bit_length() - 1
                c ^= lsb
                deg = (comp_adj[u] & P).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v = u
            
            if v != -1:
                # Recurse on P \ N(v)
                # P_new = P & ~comp_adj[v]
                P_new = P & ~comp_adj[v]
                c = P_new
                while c:
                    lsb = c & -c
                    u = lsb.bit_length() - 1
                    c ^= lsb
                    R.append(u)
                    result = bron_kerbosch(R, P & comp_adj[u], X & comp_adj[u], depth + 1)
                    if result:
                        return result
                    R.pop()
            else:
                c = P
                while c:
                    lsb = c & -c
                    u = lsb.bit_length() - 1
                    c ^= lsb
                    R.append(u)
                    result = bron_kerbosch(R, P & comp_adj[u], X & comp_adj[u], depth + 1)
                    if result:
                        return result
                    R.pop()
            return None

        R = []
        P = mask
        X = 0
        result = bron_kerbosch(R, P, X, 0)
        return result

    steps = 0
    max_steps = 5000
    
    while steps < max_steps:
        if time.time() > deadline - 2.0:
            return False
        
        steps += 1
        
        # Check K4
        k4_violation = has_k4()
        if k4_violation:
            u, v, w = k4_violation
            # The K4 is u, v, w, x where x is in common neighbors.
            # We need to break the K4. Flipping any edge in the K4 works.
            # Edges: (u,v), (u,w), (v,w), (u,x), (v,x), (w,x)
            # We know (u,v), (u,w), (v,w) are edges.
            # Let's pick a random edge from the triangle (u,v,w) to flip.
            # This might not break the K4 if the 4th vertex x is still connected, but it reduces density.
            # Better: Find the 4th vertex x.
            common = adj[u] & adj[v]
            # Find x in common such that adj[w] & (1 << x)
            c = common
            x = -1
            while c:
                lsb = c & -c
                potential_x = lsb.bit_length() - 1
                c ^= lsb
                if adj[w] & (1 << potential_x):
                    x = potential_x
                    break
            
            if x != -1:
                # Flip a random edge in the K4 {u, v, w, x}
                edges_in_k4 = [(u, v), (u, w), (u, x), (v, w), (v, x), (w, x)]
                a, b = random.choice(edges_in_k4)
                # Flip
                adj[a] ^= (1 << b)
                adj[b] ^= (1 << a)
            else:
                # Should not happen if has_k4 returned valid
                continue
                
        else:
            # Check I6
            i6_violation = has_i6()
            if i6_violation:
                # i6_violation is a list of 6 vertices
                # We need to break this independent set.
                # Add a random edge between two vertices in the set.
                u, v = random.sample(i6_violation, 2)
                # If the edge already exists, it's not an independent set. 
                # But has_i6 ensures they are non-adjacent in adj, so edge does not exist.
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            else:
                # No K4 and no I6 found.
                # This is a valid graph!
                return True
                
        # If stuck (no violations found but maybe we missed something? No, checks are exact)
        # But if we are just cycling, we might need a kick.
        # For now, the repair step always changes the graph if a violation is found.
        # If no violation is found, we return True.
        
    return False

# Ensure random is imported at top level for local_search if it was defined there? 
# No, random is imported at the top of the file.
