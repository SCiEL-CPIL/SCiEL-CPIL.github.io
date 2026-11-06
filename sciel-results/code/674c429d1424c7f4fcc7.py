import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit

    def make_paley(q):
        if q % 4 != 1:
            return None
        squares = {x * x % q for x in range(1, q)}
        edges = []
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in squares:
                    edges.append([i, j])
        return {'n': q, 'edges': edges}

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def find_triangles(n, adj):
        triangles = []
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    common = adj[i] & adj[j]
                    for k in common:
                        if k > j:
                            triangles.append((i, j, k))
        return triangles

    def find_indep_sets_5(n, adj):
        indep_sets = []
        
        def backtrack(start, current):
            if len(current) == 5:
                indep_sets.append(tuple(sorted(current)))
                return
            for v in range(start, n):
                if all(v not in adj[u] for u in current):
                    current.append(v)
                    backtrack(v + 1, current)
                    current.pop()
                    if len(indep_sets) > 2000:
                        return

        backtrack(0, [])
        return indep_sets

    def is_valid(g):
        n = g['n']
        edges = g['edges']
        adj = build_adj(n, edges)
        
        # Efficient K4 check
        for i in range(n):
            for j in range(i+1, n):
                if j in adj[i]:
                    for k in range(j+1, n):
                        if k in adj[i] and k in adj[j]:
                            common = adj[i] & adj[j] & adj[k]
                            if common:
                                return False
                                
        # Check if there's an independent set of size 6
        def has_indep_6(start, current):
            if len(current) == 6:
                return True
            if start + (6 - len(current)) > n:
                return False
            
            for v in range(start, n):
                if all(v not in adj[u] for u in current):
                    if has_indep_6(v + 1, current + [v]):
                        return True
            return False
        
        if has_indep_6(0, []):
            return False
            
        return True

    def walk_sat(n, clauses, max_iters=2000):
        x = [random.choice([True, False]) for _ in range(n)]
        
        def is_sat(clause):
            for var, sign in clause:
                val = x[var]
                if sign == 1:
                    if val: return True
                else:
                    if not val: return True
            return False

        for _ in range(max_iters):
            unsat = [c for c in clauses if not is_sat(c)]
            if not unsat:
                return x
            
            clause = random.choice(unsat)
            if random.random() < 0.5:
                var, _ = random.choice(clause)
            else:
                var = random.randint(0, n - 1)
            
            x[var] = not x[var]
            
        return None

    def try_extend(n, adj, edges, max_restarts=5):
        """Try to add one vertex. Returns new (n, adj, edges) or None."""
        triangles = find_triangles(n, adj)
        indep_sets_5 = find_indep_sets_5(n, adj)
        
        clauses = []
        for tri in triangles:
            u, v, w = tri
            clauses.append([(u, -1), (v, -1), (w, -1)])
        for ind in indep_sets_5:
            u, v, w, x, y = ind
            clauses.append([(u, 1), (v, 1), (w, 1), (x, 1), (y, 1)])
        
        if not clauses:
            assignment = [False] * n
        else:
            assignment = None
            for _ in range(max_restarts):
                assignment = walk_sat(n, clauses, max_iters=3000)
                if assignment is not None:
                    break
            
            if assignment is None and not indep_sets_5:
                # Greedy fallback
                x = [True] * n
                changed = True
                while changed:
                    changed = False
                    for tri in triangles:
                        u, v, w = tri
                        if x[u] and x[v] and x[w]:
                            idx = random.choice([u, v, w])
                            x[idx] = False
                            changed = True
                assignment = x
        
        if assignment is None:
            return None
        
        # Verify
        valid = True
        for c in clauses:
            sat = False
            for var, sign in c:
                val = assignment[var]
                if sign == 1:
                    if val: sat = True; break
                else:
                    if not val: sat = True; break
            if not sat:
                valid = False
                break
        
        if not valid:
            return None
        
        new_n = n + 1
        new_edges = edges[:]
        for i in range(n):
            if assignment[i]:
                new_edges.append([i, new_n - 1])
        
        new_adj = build_adj(new_n, new_edges)
        return (new_n, new_adj, new_edges)

    # Try multiple starting points
    best_g = None
    best_n = 0
    
    for q in [29, 17, 13, 5]:
        if time.time() > deadline - 20:
            break
        
        G = make_paley(q)
        if G is None:
            continue
        
        if not is_valid(G):
            continue
        
        n = G['n']
        edges = G['edges']
        adj = build_adj(n, edges)
        
        if n > best_n:
            best_g = {'n': n, 'edges': edges}
            best_n = n
        
        # Extension loop
        extended = True
        while extended and time.time() < deadline - 5:
            result = try_extend(n, adj, edges, max_restarts=3)
            if result is None:
                extended = False
            else:
                n, adj, edges = result
                if n > best_n:
                    best_g = {'n': n, 'edges': edges}
                    best_n = n
        
        if best_n >= 35:
            break  # Can't do much better than 35
    
    if best_g is None:
        return {'n': 1, 'edges': []}
    
    # Final verification
    if not is_valid(best_g):
        # Fallback to a known valid small graph
        return {'n': 1, 'edges': []}
    
    return best_g

# EVOLVE-BLOCK-END
