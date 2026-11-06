import time

def construct():
    start_time = time.time()
    deadline = 100.0
    
    # 1. Construct Polarity Graph for q=5
    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for y in range(q):
        points.append((0, 1, y))
    points.append((0, 0, 1))
    
    n_pol = len(points)
    adj = [[False]*n_pol for _ in range(n_pol)]
    for i in range(n_pol):
        for j in range(i+1, n_pol):
            u = points[i]
            v = points[j]
            val = (u[0]*v[1] + u[1]*v[0] + u[2]*v[2]) % q
            if val == 0:
                adj[i][j] = True
                adj[j][i] = True
                
    # 2. Add edges to reduce independence number
    # Check all non-edges
    for i in range(n_pol):
        for j in range(i+1, n_pol):
            if not adj[i][j]:
                # Count common neighbors
                common = 0
                for k in range(n_pol):
                    if adj[i][k] and adj[j][k]:
                        common += 1
                        if common > 1:
                            break
                if common <= 1:
                    adj[i][j] = True
                    adj[j][i] = True
                    
    # 3. Determine max k triangles to add
    # We need alpha_new + k < 11
    # And 31 + 3k <= 43 => 3k <= 12 => k <= 4
    
    def has_independent_set_size(M, adj, n):
        if M <= 0: return True
        if n < M: return False
        
        degrees = [sum(adj[i][j] for j in range(n) if j != i) for i in range(n)]
        order = sorted(range(n), key=lambda x: degrees[x])
        
        # Simple backtracking
        def backtrack(start_idx, current_set, remaining_needed):
            if remaining_needed == 0:
                return True
            if len(current_set) + (n - start_idx) < remaining_needed:
                return False
            
            for i in range(start_idx, n):
                is_independent = True
                for v in current_set:
                    if adj[v][i]:
                        is_independent = False
                        break
                if not is_independent:
                    continue
                
                if backtrack(i+1, current_set + [i], remaining_needed - 1):
                    return True
            return False
        
        return backtrack(0, [], M)

    best_k = 0
    for k in range(4, -1, -1):
        target_alpha = 11 - k
        if time.time() > deadline:
            break
        exists = has_independent_set_size(target_alpha, adj, n_pol)
        if not exists:
            best_k = k
            break
            
    n_total = n_pol + 3 * best_k
    edges = []
    
    for i in range(n_pol):
        for j in range(i+1, n_pol):
            if adj[i][j]:
                edges.append([i, j])
                
    for t in range(best_k):
        a = n_pol + 3*t
        b = n_pol + 3*t + 1
        c = n_pol + 3*t + 2
        edges.append([a, b])
        edges.append([a, c])
        edges.append([b, c])
        
    return {'n': n_total, 'edges': edges}
