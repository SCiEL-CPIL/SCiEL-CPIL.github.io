"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 15
    # All possible edges in a simple undirected graph on n vertices
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(all_edges)
    
    # Pre-compute edge indices for faster lookup
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    
    def get_degrees_and_adj(edge_set):
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        return deg, adj

    def count_violations(edge_set):
        """Returns the number of violating paths."""
        if not edge_set:
            return 0
        deg, adj = get_degrees_and_adj(edge_set)
        violations = 0
        for u in range(n):
            # Neighbors of u
            for v in adj[u]:
                # S = {deg(s) for s in adj[u] if s != v}
                # T = {deg(t) for t in adj[v] if t != u}
                # Violation if S and T intersect.
                # Count specific pairs for accurate penalty.
                
                deg_s_counts = {}
                for s in adj[u]:
                    if s != v:
                        d = deg[s]
                        deg_s_counts[d] = deg_s_counts.get(d, 0) + 1
                
                if not deg_s_counts:
                    continue
                
                for t in adj[v]:
                    if t != u:
                        dt = deg[t]
                        if dt in deg_s_counts:
                            violations += deg_s_counts[dt]
        return violations

    def is_valid(edge_set):
        return count_violations(edge_set) == 0

    # Simulated Annealing Parameters
    T_start = 10.0
    T_end = 0.01
    cooling_rate = 0.995
    max_iterations = 200000  # Hard cap on steps, time check will break earlier
    
    best_edges = set()
    best_score = -1
    
    # Initial state: Random sparse graph to start valid
    # Or start with a star which is always valid
    current_edges = set([(0, i) for i in range(1, n)])
    current_violations = count_violations(current_edges)
    current_score = len(current_edges) - 100 * current_violations
    
    if current_score > best_score:
        best_score = current_score
        best_edges = set(current_edges)
        
    T = T_start
    step = 0
    
    while T > T_end and time.time() < deadline:
        # Generate a neighbor
        r = random.random()
        if r < 0.3:
            # Add a random edge
            if len(current_edges) < m:
                # Find an edge not in current_edges
                # Optimization: Maintain a list of missing edges? Or just pick random and check
                # For n=15, m=105, picking random is fine
                for _ in range(10): # Try up to 10 times to find a missing edge
                    e = random.choice(all_edges)
                    if e not in current_edges:
                        new_edges = current_edges | {e}
                        break
                else:
                    continue
            else:
                continue
        elif r < 0.6:
            # Remove a random edge
            if current_edges:
                e = random.choice(list(current_edges))
                new_edges = current_edges - {e}
            else:
                continue
        else:
            # Swap: Remove one, Add one
            if current_edges:
                e_remove = random.choice(list(current_edges))
                # Find an edge to add
                added = False
                for _ in range(10):
                    e_add = random.choice(all_edges)
                    if e_add not in current_edges:
                        new_edges = (current_edges - {e_remove}) | {e_add}
                        added = True
                        break
                if not added:
                    continue
            else:
                continue

        # Evaluate new state
        new_violations = count_violations(new_edges)
        new_score = len(new_edges) - 100 * new_violations
        
        delta_score = new_score - current_score
        
        # Acceptance probability
        if delta_score > 0 or random.random() < pow(2.718281828, delta_score / T):
            current_edges = new_edges
            current_violations = new_violations
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_edges = set(current_edges)
                
        T *= cooling_rate
        step += 1
        
        # Periodic restarts if stuck? 
        # If we haven't improved in a long time, maybe restart from best?
        # For simplicity, rely on SA's ability to escape.
        
    # Final Polish: Greedy addition from the best found state
    if time.time() < deadline - 5:
        current_edges = set(best_edges)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for e in all_edges:
                if e in current_edges:
                    continue
                new_edges = current_edges | {e}
                if is_valid(new_edges):
                    current_edges = new_edges
                    improved = True
        if len(current_edges) > len(best_edges):
            best_edges = current_edges

    # Convert to list of lists
    if not best_edges:
        # Fallback
        best_edges = set([(0, i) for i in range(1, n)])
        
    result = [list(e) for e in best_edges]
    # Ensure types are int
    for e in result:
        e[0] = int(e[0])
        e[1] = int(e[1])
    return result

# EVOLVE-BLOCK-END
