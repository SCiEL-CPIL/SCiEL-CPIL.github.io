import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 150s kill
    
    # Try n from 47 down to 40 (47 is the theoretical max per task)
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        min_deg_needed = n - 39
        if min_deg_needed < 0:
            min_deg_needed = 0
        
        # Number of attempts scales with remaining time
        remaining = deadline - time.time()
        max_attempts = max(10, int(remaining * 2))  # ~2 attempts per second remaining
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Use set-based adjacency for O(1) C4 checks
            adj = [set() for _ in range(n)]
            edges = []
            
            # Helper: check if adding (u,v) creates a C4
            def can_add(u, v):
                if u in adj[v]:
                    return False
                # C4 exists if there's w in N(u) and x in N(v) with w-x in E
                # Equivalently: exists w in N(u) such that N(w) intersects N(v)
                nu = adj[u]
                nv = adj[v]
                # For efficiency, iterate the smaller set
                if len(nu) > len(nv):
                    nu, nv = nv, nu
                for w in nu:
                    if adj[w] & nv:
                        return False
                return True
            
            # Generate all pairs and order them with a bias
            # Strategy: prioritize pairs involving low-degree vertices to raise min degree
            # Use a weighted random ordering
            
            pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
            
            # Create a priority-based shuffle: pairs with lower current degree sum get higher priority
            # We'll do this by assigning random priorities but weighted by degree deficiency
            # Since we start with all degrees 0, we just shuffle initially
            random.shuffle(pairs)
            
            # Process edges in rounds, re-prioritizing based on current degrees
            processed = set()
            current_pairs = pairs[:]
            
            while current_pairs:
                if time.time() > deadline:
                    break
                
                # Sort by degree sum (lower degree vertices first) to boost min degree
                # But full sort is O(m log m); do it every few iterations
                if attempt % 3 == 0 or len(current_pairs) < 200:
                    current_pairs.sort(key=lambda p: len(adj[p[0]]) + len(adj[p[1]]))
                    random.shuffle(current_pairs)  # Add randomness to break ties
                
                # Try to add the first K edges that can be added
                k = min(50, len(current_pairs))
                still_pending = []
                for idx in range(k):
                    u, v = current_pairs[idx]
                    if (u, v) in processed:
                        continue
                    if can_add(u, v):
                        adj[u].add(v)
                        adj[v].add(u)
                        edges.append([u, v])
                        processed.add((u, v))
                    else:
                        still_pending.append((u, v))
                
                # Remaining pairs
                for idx in range(k, len(current_pairs)):
                    still_pending.append(current_pairs[idx])
                
                current_pairs = still_pending
                if not current_pairs:
                    break
            
            # Check min degree
            min_deg = min(len(adj[i]) for i in range(n))
            
            if min_deg >= min_deg_needed:
                return {'n': n, 'edges': edges}
        
        # If we've spent too much time on this n, move to smaller n
        if time.time() > deadline - 5:
            break
    
    # Fallback: Return a valid small graph
    return {'n': 39, 'edges': []}
