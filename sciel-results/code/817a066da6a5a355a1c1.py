import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # We want to maximize n. Try from 47 down to 40.
    for n in range(47, 39, -1):
        target_deg = n - 39

        # Try multiple random seeds
        max_attempts = 2000
        attempts = 0

        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            if time.time() > deadline:
                break

            # Initialize random graph
            # p = (target_deg + 1.5) / n  # Slightly denser to ensure connectivity/degree potential
            p = (target_deg + 1.5) / n
            if p > 0.5: p = 0.5
            
            adj = [0] * n

            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < p:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)

            # Repair phase
            success = True
            max_repair_steps = 1000
            steps = 0

            while steps < max_repair_steps:
                steps += 1
                if time.time() > deadline:
                    success = False
                    break

                c4_found = False
                for u in range(n):
                    if not success: break
                    for v in range(u + 1, n):
                        if not success: break
                        common = adj[u] & adj[v]
                        # Check for at least 2 common neighbors
                        if common and (common & (common - 1)):
                            # Extract two neighbors
                            x = (common & -common).bit_length() - 1
                            common ^= (1 << x)
                            y = (common & -common).bit_length() - 1

                            # C4 edges: (u,x), (x,v), (v,y), (y,u)
                            candidates = [
                                (u, x),
                                (x, v),
                                (v, y),
                                (y, u)
                            ]
                            
                            # Randomly select a candidate to remove
                            # Filter by degree constraint
                            valid_candidates = []
                            for a, b in candidates:
                                deg_a = adj[a].bit_count()
                                deg_b = adj[b].bit_count()
                                if deg_a - 1 >= target_deg and deg_b - 1 >= target_deg:
                                    valid_candidates.append((a, b))
                            
                            if valid_candidates:
                                a, b = random.choice(valid_candidates)
                                adj[a] &= ~(1 << b)
                                adj[b] &= ~(1 << a)
                                c4_found = True
                            else:
                                success = False
                            break
                    if c4_found or not success:
                        break

                if not success:
                    break
                
                if not c4_found:
                    # No C4s, check min degree
                    min_deg = min(adj[i].bit_count() for i in range(n))
                    if min_deg >= target_deg:
                        edges = []
                        for u in range(n):
                            for v in range(u + 1, n):
                                if adj[u] & (1 << v):
                                    edges.append([u, v])
                        return {'n': n, 'edges': edges}
                    else:
                        success = False
                        break

        if time.time() > deadline:
            break

    return {'n': 39, 'edges': []}
