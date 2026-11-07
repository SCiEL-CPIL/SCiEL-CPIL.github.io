import time
import random

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 85
    
    def get_degrees(adj):
        return [sum(row) for row in adj]
    
    def is_valid(adj, deg):
        # Check if any path s-u-v-t has deg[s] == deg[t]
        # Optimize: Only check pairs with same degree
        for s in range(n):
            for t in range(s+1, n):
                if deg[s] != deg[t]:
                    continue
                # Check if there is a path of length 3 between s and t
                # s - u - v - t
                # u must be neighbor of s
                # v must be neighbor of t
                # u-v must be edge
                # s, t, u, v distinct
                found_path = False
                for u in range(n):
                    if u == s or u == t:
                        continue
                    if not adj[s][u]:
                        continue
                    for v in range(n):
                        if v in (s, t, u):
                            continue
                        if adj[u][v] and adj[v][t]:
                            found_path = True
                            break
                    if found_path:
                        break
                if found_path:
                    return False
        return True

    def count_edges(adj):
        return sum(sum(row) for row in adj) // 2

    best_adj = None
    best_count = 0

    # Start with complete graph
    adj = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            adj[i][j] = 1
            adj[j][i] = 1
    
    deg = get_degrees(adj)
    
    # Iteratively remove edges to fix violations
    # We prioritize removing edges that are part of many violations
    # Or just random removal
    
    for trial in range(100):
        if time.time() > deadline - 5:
            break
            
        # Randomize start if needed, or just perturb
        current_adj = [row[:] for row in adj]
        current_deg = get_degrees(current_adj)
        
        # Try to fix by removing edges
        # Find all violating pairs
        # For simplicity, just check validity and remove random edge if invalid
        # This is slow. Better: find a specific violation and remove an edge on the path
        
        # Let's do a simple hill climbing:
        # 1. If invalid, find a violating path s-u-v-t.
        # 2. Remove one of the edges (s,u), (u,v), (v,t).
        # 3. Repeat.
        
        valid = is_valid(current_adj, current_deg)
        steps = 0
        while not valid and steps < 1000:
            if time.time() > deadline - 5:
                break
            # Find a violation
            violation_found = False
            for s in range(n):
                for t in range(s+1, n):
                    if current_deg[s] != current_deg[t]:
                        continue
                    for u in range(n):
                        if u in (s, t) or not current_adj[s][u]:
                            continue
                        for v in range(n):
                            if v in (s, t, u) or not (current_adj[u][v] and current_adj[v][t]):
                                continue
                            # Found path s-u-v-t
                            # Remove a random edge from this path
                            edges_to_remove = [(s, u), (u, v), (v, t)]
                            e_i, e_j = random.choice(edges_to_remove)
                            if current_adj[e_i][e_j]:
                                current_adj[e_i][e_j] = 0
                                current_adj[e_j][e_i] = 0
                                current_deg[e_i] -= 1
                                current_deg[e_j] -= 1
                                violation_found = True
                                break
                        if violation_found:
                            break
                    if violation_found:
                        break
                if violation_found:
                    break
            
            if not violation_found:
                break
                
            valid = is_valid(current_adj, current_deg)
            steps += 1
        
        if valid:
            cnt = count_edges(current_adj)
            if cnt > best_count:
                best_count = cnt
                best_adj = [row[:] for row in current_adj]
        
        # Local search: try to add edges back
        if valid and time.time() < deadline - 5:
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                # Try adding each missing edge
                for i in range(n):
                    for j in range(i+1, n):
                        if not current_adj[i][j]:
                            # Try add
                            current_adj[i][j] = 1
                            current_adj[j][i] = 1
                            current_deg[i] += 1
                            current_deg[j] += 1
                            if is_valid(current_adj, current_deg):
                                improved = True
                            else:
                                current_adj[i][j] = 0
                                current_adj[j][i] = 0
                                current_deg[i] -= 1
                                current_deg[j] -= 1
                if improved:
                    cnt = count_edges(current_adj)
                    if cnt > best_count:
                        best_count = cnt
                        best_adj = [row[:] for row in current_adj]

    # Build result
    result = []
    if best_adj is not None:
        for i in range(n):
            for j in range(i+1, n):
                if best_adj[i][j]:
                    result.append([int(i), int(j)])
    else:
        # Fallback: Empty graph
        pass

    return result
