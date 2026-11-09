import random
import time

def construct():
    start_time = time.time()
    time_limit = 110.0  # seconds
    
    best_result = {'n': 39, 'edges': []}
    
    # Helper to calculate popcount
    def popcount(x):
        return x.bit_count()
        
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        min_deg_required = n - 39
        if min_deg_required < 1:
            min_deg_required = 1
            
        # Try multiple random restarts for this n
        for attempt in range(100):
            if time.time() - start_time > time_limit:
                break
                
            adj = [0] * n
            
            def can_add(u, v):
                if u == v:
                    return False
                if (adj[u] >> v) & 1:
                    return False
                common = adj[u] & adj[v]
                return common.bit_count() <= 1

            def find_c4():
                for u in range(n):
                    au = adj[u]
                    for v in range(u + 1, n):
                        if (au >> v) & 1:
                            common = au & adj[v]
                            if common.bit_count() >= 2:
                                # Extract two bits
                                w = (common & -common).bit_length() - 1
                                common ^= (1 << w)
                                x = (common & -common).bit_length() - 1
                                return (u, v, w, x)
                return None

            # Initialize
            # Start with a random graph with density approx min_deg_required / n
            # But ensure we have enough edges to potentially reach min degree
            target_edges = int(n * min_deg_required / 2) + 10
            edges_added = 0
            # Shuffle vertex pairs for better randomization
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(pairs)
            
            for u, v in pairs:
                if time.time() - start_time > time_limit:
                    break
                if edges_added >= target_edges:
                    break
                if can_add(u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges_added += 1
            
            # Remove C4s
            for _ in range(2000):
                if time.time() - start_time > time_limit:
                    break
                c4 = find_c4()
                if not c4:
                    break
                u, v, w, x = c4
                edges_in_c4 = [(u, w), (w, v), (v, x), (x, u)]
                e_u, e_v = random.choice(edges_in_c4)
                adj[e_u] &= ~(1 << e_v)
                adj[e_v] &= ~(1 << e_u)
                
            # Check min degree
            min_deg = min(adj[i].bit_count() for i in range(n))
            if min_deg >= min_deg_required:
                # Valid
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if (adj[u] >> v) & 1:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
                
            # If not valid, we could try local search, but for simplicity, restart
            # A simple local search: try to add edges to low-degree vertices
            # If we have time, do a few steps
            if time.time() - start_time < time_limit - 10:
                for _ in range(100):
                    # Find vertex with min degree
                    min_idx = min(range(n), key=lambda i: adj[i].bit_count())
                    if adj[min_idx].bit_count() >= min_deg_required:
                        break
                    # Try to add an edge from min_idx to a random v
                    added = False
                    for _ in range(50):
                        v = random.randint(0, n - 1)
                        if can_add(min_idx, v):
                            adj[min_idx] |= (1 << v)
                            adj[v] |= (1 << min_idx)
                            added = True
                            break
                    if not added:
                        break
                # Re-check min degree after local search
                min_deg = min(adj[i].bit_count() for i in range(n))
                if min_deg >= min_deg_required:
                    # Ensure no C4s (local search might have introduced them? No, can_add prevents it)
                    # But can_add only checks immediate C4 formation.
                    # If we add (u,v) with <=1 common neighbor, no new C4 is formed.
                    # So graph remains C4-free.
                    edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if (adj[u] >> v) & 1:
                                edges.append([u, v])
                    return {'n': n, 'edges': edges}

    return best_result
