import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Try n from 47 down to 39
    for n in range(47, 38, -1):
        target_deg = n - 39
        # If target_deg <= 0, the empty graph works, but we want max n.
        # For n=39, target_deg=0.
        
        # We want to find a C4-free graph with min degree >= target_deg.
        # Let's try to construct one.
        
        max_restarts = 100
        found = None
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize random graph
            # Aim for a degree slightly higher than target to allow removal
            init_deg = target_deg + 2 if target_deg + 2 < n else target_deg
            # Random graph
            adj = [0] * n
            # Add random edges
            # To ensure connectivity or just random
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.3: # Density 0.3
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Repair phase: Remove edges that create C4
            # A C4 exists if there are u, v with |N(u) & N(v)| >= 2.
            # We iterate until no C4.
            changed = True
            while changed:
                changed = False
                # Find a C4
                # To speed up, we can pick a random pair or scan
                # Scanning all pairs is O(n^2)
                for u in range(n):
                    for v in range(u+1, n):
                        common = adj[u] & adj[v]
                        if common & (common - 1): # has at least 2 bits
                            # Found C4 involving u, v and two common neighbors
                            # Pick a common neighbor w
                            # Remove edge (u, w) or (v, w) or (u, v)?
                            # Removing (u, v) might break connectivity but doesn't necessarily break C4 if there are other paths?
                            # Actually, if N(u) & N(v) has >= 2, say w1, w2.
                            # The C4 is u-w1-v-w2-u.
                            # Removing any edge in this cycle breaks this specific C4.
                            # Removing (u, w1) is a good choice.
                            w1 = common & -common
                            w1_idx = w1.bit_length() - 1
                            # Remove edge (u, w1_idx)
                            adj[u] &= ~(1 << w1_idx)
                            adj[w1_idx] &= ~(1 << u)
                            changed = True
                            break
                    if changed:
                        break
                
                # If we removed an edge, check time
                if time.time() > deadline:
                    return found if found else {'n': 39, 'edges': []}
            
            # Check min degree
            min_deg = min(bin(adj[i]).count('1') for i in range(n))
            if min_deg >= target_deg:
                # Success!
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                found = {'n': n, 'edges': edges}
                # We found a valid graph for this n.
                # Since we iterate n downwards, this is the max n for this restart.
                # We can return immediately if n is high?
                # The task says "Maximize n".
                # So if we find n=47, we return.
                # If we find n=46, we might want to keep searching for 47?
                # But we are iterating n from 47 down.
                # So if we enter the loop with n=47 and find a solution, we return it.
                # If we fail to find a solution for n=47 within restarts/time, we move to n=46.
                return found
        
        # If no restart succeeded for this n, continue to next n
        
    # Fallback: n=39, empty graph
    return {'n': 39, 'edges': []}
