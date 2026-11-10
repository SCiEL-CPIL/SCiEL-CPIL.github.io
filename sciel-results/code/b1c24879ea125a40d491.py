    def solve_csp(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Convert to sets for faster membership tests
        adj_sets = [set(neigh) for neigh in adj]
        
        # Initial domains
        domains = [set(lst) for lst in lists]
        
        # AC-3 arc consistency
        def ac3():
            queue = []
            for u in range(n):
                for v in adj[u]:
                    queue.append((u, v))
            
            while queue:
                i, j = queue.pop()
                removed = False
                for ci in list(domains[i]):
                    if not any(ci != cj for cj in domains[j]):
                        domains[i].remove(ci)
                        removed = True
                        if not domains[i]:
                            return False
                if removed:
                    for k in adj_sets[i]:
                        if k != j:
                            queue.append((k, i))
            return True
        
        if not ac3():
            return None
        
        # Backtracking search with forward checking and trail
        def get_order():
            order = sorted(range(n), key=lambda v: (len(domains[v]), -len(adj[v])))
            return order
        
        order = get_order()
        assigned = [-1] * n
        trail = []
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == len(order):
                return True
            
            var = order[idx]
            
            for c in sorted(domains[var]):
                valid = True
                for u in adj[var]:
                    if assigned[u] == c:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                assigned[var] = c
                ok = True
                for u in adj[var]:
                    if assigned[u] == -1:
                        if c in domains[u]:
                            domains[u].remove(c)
                            trail.append((u, c))
                            if not domains[u]:
                                ok = False
                                break
                
                if ok:
                    if backtrack(idx + 1):
                        return True
                
                while trail:
                    u, c_removed = trail.pop()
                    domains[u].add(c_removed)
                
                assigned[var] = -1
            
            return False

        if backtrack(0):
            colour = [0] * n
            for v in range(n):
                colour[v] = assigned[v]
            return colour
        
        return None
