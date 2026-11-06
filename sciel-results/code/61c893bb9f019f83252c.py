import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    def try_construct(n, target_deg, time_limit):
        """
        Attempts to construct a C4-free graph on n vertices with min degree >= target_deg.
        Returns (success, edges)
        """
        if target_deg <= 0:
            return True, []
        
        max_attempts = 200
        attempt = 0
        while attempt < max_attempts and time.time() < time_limit:
            attempt += 1
            
            adj = [0] * n
            degrees = [0] * n
            
            # List of all potential edges
            potential_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
            random.shuffle(potential_edges)
            
            # Greedy pass 1: Add edges if they don't create C4
            for u, v in potential_edges:
                if time.time() > time_limit:
                    return False, None
                
                # Check if adding (u, v) creates a C4
                # C4 exists if there is a in N(u) such that N(a) intersects N(v)
                # We iterate over neighbors of u
                creates_c4 = False
                nu = adj[u]
                # Iterate through bits in nu
                temp_nu = nu
                while temp_nu:
                    a = (temp_nu & -temp_nu).bit_length() - 1
                    temp_nu &= temp_nu - 1
                    # If a is connected to any neighbor of v, we have a path u-a-b-v
                    # which forms a cycle u-a-b-v-u if u-v is added?
                    # Wait. The cycle is u-a-b-v-u.
                    # Edges: (u,a), (a,b), (b,v), (v,u).
                    # We are adding (v,u).
                    # So we need (u,a), (a,b), (b,v) to exist.
                    # (u,a) exists because a in N(u).
                    # (b,v) exists because b in N(v).
                    # (a,b) exists if a in N(b) or b in N(a).
                    # So if N(a) & N(v) is not empty, there is a b in N(v) such that b in N(a).
                    # This creates the C4.
                    if adj[a] & adj[v]:
                        creates_c4 = True
                        break
                
                if not creates_c4:
                    # Add edge
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    degrees[u] += 1
                    degrees[v] += 1
            
            min_deg = min(degrees) if n > 0 else 0
            if min_deg >= target_deg:
                # Success
                edges = []
                for u in range(n):
                    for v in range(u+1, n):
                        if (adj[u] >> v) & 1:
                            edges.append([u, v])
                return True, edges
                
            # If failed, we could try a second pass focusing on low degree vertices
            # But simple restarts with different shuffles are often effective for small n
            # Let's do a few local improvement steps
            for _ in range(5):
                if time.time() > time_limit:
                    return False, None
                
                # Find vertices with degree < target_deg
                low_deg_vertices = [i for i in range(n) if degrees[i] < target_deg]
                if not low_deg_vertices:
                    break
                    
                # Try to add edges involving low degree vertices
                # Create a list of candidate edges
                candidates = []
                for u in low_deg_vertices:
                    for v in range(n):
                        if v != u and not (adj[u] >> v) & 1:
                            candidates.append((u, v))
                
                if not candidates:
                    break
                    
                random.shuffle(candidates)
                
                added_any = False
                for u, v in candidates:
                    if time.time() > time_limit:
                        return False, None
                    if degrees[u] >= target_deg and degrees[v] >= target_deg:
                        continue
                    
                    creates_c4 = False
                    nu = adj[u]
                    temp_nu = nu
                    while temp_nu:
                        a = (temp_nu & -temp_nu).bit_length() - 1
                        temp_nu &= temp_nu - 1
                        if adj[a] & adj[v]:
                            creates_c4 = True
                            break
                    
                    if not creates_c4:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        degrees[u] += 1
                        degrees[v] += 1
                        added_any = True
                
                if not added_any:
                    break
            
            min_deg = min(degrees) if n > 0 else 0
            if min_deg >= target_deg:
                edges = []
                for u in range(n):
                    for v in range(u+1, n):
                        if (adj[u] >> v) & 1:
                            edges.append([u, v])
                return True, edges

        return False, None

    # Try n=47 first
    for n in [47, 46, 45, 44]:
        if time.time() > deadline:
            break
            
        target_deg = n - 39
        if target_deg <= 0:
            return {'n': n, 'edges': []}
            
        success, edges = try_construct(n, target_deg, deadline)
        if success:
            return {'n': n, 'edges': edges}
            
    # Fallback: return a valid small graph
    # n=39, target_deg=0. Empty graph works.
    return {'n': 39, 'edges': []}
