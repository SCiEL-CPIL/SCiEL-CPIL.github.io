import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    def check_time():
        return time.time() < deadline
    
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    # Build initial Paley graph
    n = q
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i].add(j)
                adj[j].add(i)
    
    def get_k4s_and_ind4s(adj, n):
        """Find all K4s and independent 4-sets"""
        k4s = []
        ind4s = []
        for combo in combinations(range(n), 4):
            is_clique = True
            is_indep = True
            for u, v in combinations(combo, 2):
                if v not in adj[u]:
                    is_clique = False
                else:
                    is_indep = False
                if not is_clique and not is_indep:
                    break
            if is_clique:
                k4s.append(combo)
            if is_indep:
                ind4s.append(combo)
        return k4s, ind4s
    
    # Precompute K4s and independent 4-sets
    k4s, ind4s = get_k4s_and_ind4s(adj, n)
    
    def try_add_vertex(adj, n, k4s, ind4s, max_attempts=5000):
        """Try to add a vertex with constraint-based search"""
        if not check_time():
            return None
        
        # For each existing vertex, determine constraints
        # A new vertex v with neighbor set S is valid iff:
        # 1. No K4 in S (i.e., S contains no K4)
        # 2. No independent 4-set in complement of S
        
        # Strategy: Start with a random assignment, then fix conflicts via backtracking
        best_neighbors = None
        best_violations = float('inf')
        
        for attempt in range(max_attempts):
            if not check_time():
                break
            
            # Random initial assignment
            neighbors = set()
            for i in range(n):
                if random.random() < 0.5:
                    neighbors.add(i)
            
            # Count violations
            def count_violations(neighbors):
                violations = 0
                # Check K4s in neighbors
                for k4 in k4s:
                    if all(v in neighbors for v in k4):
                        violations += 1
                # Check independent 4-sets in non-neighbors
                non_neighbors = set(range(n)) - neighbors
                for ind4 in ind4s:
                    if all(v in non_neighbors for v in ind4):
                        violations += 1
                return violations
            
            violations = count_violations(neighbors)
            if violations < best_violations:
                best_violations = violations
                best_neighbors = set(neighbors)
            
            if violations == 0:
                return neighbors
            
            # Local search: try flipping one vertex at a time
            improved = True
            while improved and check_time():
                improved = False
                for i in range(n):
                    if i in neighbors:
                        neighbors.remove(i)
                    else:
                        neighbors.add(i)
                    new_violations = count_violations(neighbors)
                    if new_violations < violations:
                        violations = new_violations
                        improved = True
                        if violations == 0:
                            return neighbors
                    else:
                        # Revert
                        if i in neighbors:
                            neighbors.remove(i)
                        else:
                            neighbors.add(i)
        
        return None
    
    # Try to add vertices iteratively
    added_count = 0
    while check_time() and added_count < 10:  # Cap at 10 additions to be safe
        result = try_add_vertex(adj, n, k4s, ind4s)
        if result is None:
            break
        
        # Add the new vertex
        new_vertex = n
        adj.append(set(result))
        for i in result:
            adj[i].add(new_vertex)
        n += 1
        added_count += 1
        
        # Recompute K4s and independent 4-sets
        k4s, ind4s = get_k4s_and_ind4s(adj, n)
        
        # Try a local search pass to see if we can add more
        # by adjusting edges (this is optional and time-dependent)
        if check_time():
            # Try flipping edges to make it easier to add more vertices
            for _ in range(50):
                if not check_time():
                    break
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                # Toggle edge (u, v)
                if v in adj[u]:
                    adj[u].remove(v)
                    adj[v].remove(u)
                else:
                    adj[u].add(v)
                    adj[v].add(u)
                
                # Check if graph is still (5,5)-free
                # Quick check: look for K5 or independent 5
                has_k5 = False
                has_ind5 = False
                for combo in combinations(range(n), 5):
                    is_clique = True
                    is_indep = True
                    for a, b in combinations(combo, 2):
                        if b not in adj[a]:
                            is_clique = False
                        else:
                            is_indep = False
                        if not is_clique and not is_indep:
                            break
                    if is_clique:
                        has_k5 = True
                        break
                    if is_indep:
                        has_ind5 = True
                        break
                    if not check_time():
                        break
                    if has_k5 or has_ind5:
                        break
                
                if has_k5 or has_ind5:
                    # Revert
                    if v in adj[u]:
                        adj[u].remove(v)
                        adj[v].remove(u)
                    else:
                        adj[u].add(v)
                        adj[v].add(u)
        
        # Recompute after any modifications
        k4s, ind4s = get_k4s_and_ind4s(adj, n)
    
    # Final validation: ensure the graph is (5,5)-free
    def is_valid(adj, n):
        for combo in combinations(range(n), 5):
            is_clique = True
            is_indep = True
            for u, v in combinations(combo, 2):
                if v not in adj[u]:
                    is_clique = False
                else:
                    is_indep = False
                if not is_clique and not is_indep:
                    break
            if is_clique or is_indep:
                return False
        return True
    
    # If the graph is invalid, fall back to the original Paley graph
    if not is_valid(adj, n):
        n = q
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % q in residues:
                    adj[i].add(j)
                    adj[j].add(i)
    
    # Return edges
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
    
    return n, edges
