import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Helper to check validity
    def is_valid(n, colors):
        adj = [ [set() for _ in range(n)] for _ in range(5) ]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                adj[c][i].add(j)
                adj[c][j].add(i)
                idx += 1
        for c in range(5):
            for u in range(n):
                for v in range(u+1, n):
                    if len(adj[c][u].intersection(adj[c][v])) >= 2:
                        return False
        return True

    # Local search function
    def local_search(n, time_limit):
        num_edges = n * (n - 1) // 2
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        
        # Precompute edge indices for faster access?
        # edge_idx[i][j]
        edge_idx = [[-1]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                edge_idx[i][j] = idx
                edge_idx[j][i] = idx
                idx += 1

        # Maintain adjacency sets for each color
        adj = [ [set() for _ in range(n)] for _ in range(5) ]
        for i in range(n):
            for j in range(i+1, n):
                c = colors[edge_idx[i][j]]
                adj[c][i].add(j)
                adj[c][j].add(i)

        def count_c4s():
            count = 0
            for c in range(5):
                for u in range(n):
                    for v in range(u+1, n):
                        if len(adj[c][u].intersection(adj[c][v])) >= 2:
                            count += 1
            return count

        current_violations = count_c4s()
        
        while time.time() < time_limit:
            if current_violations == 0:
                return colors
            
            # Find a violation
            viol_found = False
            for c in range(5):
                for u in range(n):
                    for v in range(u+1, n):
                        inter = adj[c][u].intersection(adj[c][v])
                        if len(inter) >= 2:
                            # Recolor edge (u,v)
                            e_idx = edge_idx[u][v]
                            old_c = colors[e_idx]
                            
                            # Try to change to a color that reduces violations
                            best_new_c = -1
                            best_count = current_violations + 1
                            
                            for new_c in range(5):
                                if new_c == old_c:
                                    continue
                                # Simulate change
                                adj[old_c][u].remove(v)
                                adj[old_c][v].remove(u)
                                adj[new_c][u].add(v)
                                adj[new_c][v].add(u)
                                colors[e_idx] = new_c
                                
                                # Recount violations (expensive, but n is small)
                                # Optimization: Only check neighborhoods of u and v
                                # But for simplicity, full count if n < 30
                                new_count = count_c4s()
                                
                                if new_count < best_count:
                                    best_count = new_count
                                    best_new_c = new_c
                                    
                                # Revert
                                adj[new_c][u].remove(v)
                                adj[new_c][v].remove(u)
                                adj[old_c][u].add(v)
                                adj[old_c][v].add(u)
                                colors[e_idx] = old_c
                            
                            if best_new_c != -1:
                                # Apply best change
                                e_idx = edge_idx[u][v]
                                old_c = colors[e_idx]
                                adj[old_c][u].remove(v)
                                adj[old_c][v].remove(u)
                                adj[best_new_c][u].add(v)
                                adj[best_new_c][v].add(u)
                                colors[e_idx] = best_new_c
                                current_violations = best_count
                                viol_found = True
                                break
                    if viol_found:
                        break
                if viol_found:
                    break
            
            if not viol_found:
                break
        
        return colors if is_valid(n, colors) else None

    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try to find better n
    for n in range(26, 11, -1):
        if time.time() > deadline:
            break
        # Allocate remaining time for this n
        remaining = deadline - time.time()
        if remaining < 5:
            break
            
        colors = local_search(n, time.time() + remaining - 2)
        if colors is not None:
            best_n = n
            best_colors = colors
            break # Found valid for this n, try higher? No, we iterate down.
            # Actually, we want max n. So if 26 works, we stop.
            # If 26 fails, try 25.
            
    return {'n': best_n, 'colors': best_colors}
