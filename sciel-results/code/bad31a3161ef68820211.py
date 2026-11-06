import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin for 120s limit

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

    def build_adj_bit(n, edges):
        adj_bits = [0] * n
        for u, v in edges:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
        return adj_bits

    def is_valid(n, edges):
        """Check if graph is K4-free and has no independent set of size 6."""
        if n < 6:
            return True
        adj = build_adj(n, edges)
        adj_bits = build_adj_bit(n, edges)
        
        # K4 check
        for i in range(n):
            for j in range(i + 1, n):
                if j in adj[i]:
                    for k in range(j + 1, n):
                        if k in adj[i] and k in adj[j]:
                            common = adj[i] & adj[j] & adj[k]
                            if common:
                                return False
        
        # Independent set of size 6 check
        # Backtracking with pruning
        def has_indep_6():
            # Sort vertices by degree (ascending) to prune faster?
            # Or just use bitmask
            def backtrack(start, current_mask, size):
                if size == 6:
                    return True
                remaining = n - start
                if remaining < 6 - size:
                    return False
                for v in range(start, n):
                    if adj_bits[v] & current_mask == 0:
                        if backtrack(v + 1, current_mask | (1 << v), size + 1):
                            return True
                return False
            return backtrack(0, 0, 0)
            
        if has_indep_6():
            return False
            
        return True

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

    def find_indep_sets_5(n, adj_bits, max_sets=5000):
        indep_sets = []
        
        def backtrack(start, current_mask, current_list, size):
            if len(indep_sets) >= max_sets:
                return
            if size == 5:
                indep_sets.append(tuple(current_list))
                return
            for v in range(start, n):
                if len(indep_sets) >= max_sets:
                    return
                if adj_bits[v] & current_mask == 0:
                    current_list.append(v)
                    backtrack(v + 1, current_mask | (1 << v), current_list, size + 1)
                    current_list.pop()
        
        backtrack(0, 0, [], 0)
        return indep_sets

    def extend_graph(n, adj, adj_bits, edges, time_limit):
        """
        Try to add one vertex to the graph.
        Returns (new_n, new_adj, new_adj_bits, new_edges) or None.
        """
        start = time.time()
        triangles = find_triangles(n, adj)
        indep_sets_5 = find_indep_sets_5(n, adj_bits, max_sets=5000)
        
        # Build clauses for the new vertex connection variables x[0..n-1]
        # x[i] = 1 means edge (new_vertex, i) exists.
        # Clause types:
        # 1. Triangle (u,v,w): NOT (x[u] AND x[v] AND x[w]) => ~x[u] OR ~x[v] OR ~x[w]
        # 2. Indep5 (u,v,w,x,y): x[u] OR x[v] OR x[w] OR x[x] OR x[y]
        
        clauses = []
        # Map variable to list of clause indices it appears in, for fast delta calculation
        var_to_clauses = [[] for _ in range(n)]
        
        for u, v, w in triangles:
            # Clause: ~x[u] OR ~x[v] OR ~x[w]
            # Represented as list of (var, sign)
            c = [(u, -1), (v, -1), (w, -1)]
            idx = len(clauses)
            clauses.append(c)
            var_to_clauses[u].append(idx)
            var_to_clauses[v].append(idx)
            var_to_clauses[w].append(idx)
            
        for ind in indep_sets_5:
            u, v, w, x, y = ind
            # Clause: x[u] OR x[v] OR x[w] OR x[x] OR x[y]
            c = [(u, 1), (v, 1), (w, 1), (x, 1), (y, 1)]
            idx = len(clauses)
            clauses.append(c)
            var_to_clauses[u].append(idx)
            var_to_clauses[v].append(idx)
            var_to_clauses[w].append(idx)
            var_to_clauses[x].append(idx)
            var_to_clauses[y].append(idx)
            
        if not clauses:
            # No constraints, just pick a valid assignment (e.g., all False)
            assignment = [False] * n
            # But we must ensure no K4 or Indep6.
            # If no triangles and no indep5, any assignment works?
            # Actually, if no triangles, no K4 can be formed with the new vertex.
            # If no indep5, no Indep6 can be formed.
            # So any assignment is valid locally.
            # We should pick one that maximizes connectivity or something?
            # Let's pick random.
            assignment = [random.random() < 0.5 for _ in range(n)]
        else:
            # Local Search / Simulated Annealing on the assignment
            # Objective: Minimize number of violated clauses.
            
            # Initialize assignment
            x = [random.random() < 0.5 for _ in range(n)]
            
            def count_violations(x):
                count = 0
                for c in clauses:
                    sat = False
                    for var, sign in c:
                        val = x[var]
                        if sign == 1:
                            if val:
                                sat = True
                                break
                        else:
                            if not val:
                                sat = True
                                break
                    if not sat:
                        count += 1
                return count
            
            current_violations = count_violations(x)
            best_x = x[:]
            best_violations = current_violations
            
            T = 1.0
            cooling = 0.999
            max_iters = 200000
            
            for i in range(max_iters):
                if time.time() - start > time_limit:
                    break
                if current_violations == 0:
                    break
                
                # Pick random variable
                var = random.randint(0, n - 1)
                
                # Calculate delta violations
                # Find all clauses containing 'var'
                old_violations = current_violations
                x[var] = not x[var]
                new_violations = count_violations(x)
                
                delta = new_violations - old_violations
                
                if delta < 0:
                    # Accept
                    current_violations = new_violations
                    if current_violations < best_violations:
                        best_violations = current_violations
                        best_x = x[:]
                else:
                    # Accept with probability exp(-delta / T)
                    if random.random() < pow(2.71828, -delta / T):
                        current_violations = new_violations
                    else:
                        # Reject: revert
                        x[var] = not x[var]
                
                T *= cooling
                if T < 0.01:
                    T = 0.01
            
            if best_violations == 0:
                assignment = best_x
            else:
                # Failed to find a perfect assignment
                # Check if we can repair it?
                # Or just return None
                # Maybe try a few more random restarts if time permits?
                # For now, return None to be safe.
                # Or return the best effort? No, must be valid.
                return None
        
        # Verify the assignment against the clauses one last time
        valid = True
        for c in clauses:
            sat = False
            for var, sign in c:
                val = assignment[var]
                if sign == 1:
                    if val:
                        sat = True
                        break
                else:
                    if not val:
                        sat = True
                        break
            if not sat:
                valid = False
                break
        
        if not valid:
            return None
        
        # Construct new graph
        new_n = n + 1
        new_edges = edges[:]
        new_node = n
        for i in range(n):
            if assignment[i]:
                new_edges.append([i, new_node])
        
        new_adj = build_adj(new_n, new_edges)
        new_adj_bits = build_adj_bit(new_n, new_edges)
        
        return (new_n, new_adj, new_adj_bits, new_edges)

    # Main Search Logic
    best_g = None
    best_n = 0
    
    # Start with Paley(29)
    G = make_paley(29)
    if G is None:
        return {'n': 1, 'edges': []}
    
    n = G['n']
    edges = G['edges']
    adj = build_adj(n, edges)
    adj_bits = build_adj_bit(n, edges)
    
    if is_valid(n, edges):
        best_g = {'n': n, 'edges': edges[:]}
        best_n = n
    
    # Try to extend up to 36
    target_n = 36
    current_n = n
    current_adj = adj
    current_adj_bits = adj_bits
    current_edges = edges
    
    while current_n < target_n:
        remaining_time = deadline - time.time()
        if remaining_time < 10:
            break
            
        # Allocate time for this extension step
        # Later steps are harder, so maybe give more time?
        # Or just use a fixed chunk.
        step_time = min(15, remaining_time / (target_n - current_n))
        
        result = extend_graph(current_n, current_adj, current_adj_bits, current_edges, step_time)
        
        if result is None:
            break
        
        current_n, current_adj, current_adj_bits, current_edges = result
        
        # Verify validity of the new graph
        if is_valid(current_n, current_edges):
            if current_n > best_n:
                best_g = {'n': current_n, 'edges': current_edges[:]}
                best_n = current_n
        else:
            # Should not happen if extend_graph is correct, but check
            break

    # If we didn't reach 36, try a different strategy?
    # Maybe start from Paley(17) and extend?
    # Or try a random restart of the extension process?
    
    if best_n < 36 and time.time() < deadline - 20:
        # Try to restart from a different seed or perturb the current best
        # For now, just return the best found
        
        # One more attempt: Direct search for 36?
        # Too slow.
        pass

    if best_g is None:
        return {'n': 1, 'edges': []}
    
    return best_g
