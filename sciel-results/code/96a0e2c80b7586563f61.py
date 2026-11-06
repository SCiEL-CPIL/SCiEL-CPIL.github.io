"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def _has_k5_or_indep5(n, adj):
    """Check if graph has K5 or independent set of size 5. adj is list of sets."""
    # Check for K5: any 5-clique
    for combo in itertools.combinations(range(n), 5):
        is_clique = True
        for i in range(5):
            for j in range(i+1, 5):
                if combo[j] not in adj[combo[i]]:
                    is_clique = False
                    break
            if not is_clique:
                break
        if is_clique:
            return True
    
    # Check for independent set of size 5
    for combo in itertools.combinations(range(n), 5):
        is_indep = True
        for i in range(5):
            for j in range(i+1, 5):
                if combo[j] in adj[combo[i]]:
                    is_indep = False
                    break
            if not is_indep:
                break
        if is_indep:
            return True
    
    return False

def _try_add_vertex(n, adj, new_conn_pattern):
    """Try to add a vertex with given connections. Return (success, new_n, new_adj)."""
    new_n = n + 1
    new_adj = [set(s) for s in adj]
    new_adj.append(set())
    
    for i in range(n):
        if new_conn_pattern[i]:
            new_adj[i].add(new_n - 1)
            new_adj[new_n - 1].add(i)
    
    if _has_k5_or_indep5(new_n, new_adj):
        return False, n, adj
    return True, new_n, new_adj

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # Start with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    adj = [set() for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i].add(j)
                adj[j].add(i)
                edges.append((i, j))
    
    best_n = n
    best_adj = [set(s) for s in adj]
    
    # Try to add vertices one at a time
    # For each new vertex, try all possible connection patterns to existing vertices
    # We use a greedy approach: try patterns in order of increasing "density"
    
    current_adj = [set(s) for s in adj]
    current_n = n
    
    max_attempts = 200
    attempts = 0
    
    while current_n < 46 and attempts < max_attempts:
        if time.time() > deadline:
            break
        
        # Try all 2^current_n possible connection patterns is too expensive
        # Instead, try smart patterns:
        # 1. Connect to no one (isolated vertex)
        # 2. Connect to exactly k vertices for small k
        # 3. Random patterns
        
        found = False
        
        # First try isolated vertex
        if current_n < 46:
            success, new_n, new_adj = _try_add_vertex(current_n, current_adj, [False] * current_n)
            if success:
                current_adj = new_adj
                current_n = new_n
                found = True
                if current_n > best_n:
                    best_n = current_n
                    best_adj = [set(s) for s in current_adj]
                attempts += 1
                continue
        
        # Try connecting to exactly k vertices
        for k in range(0, min(current_n, 8) + 1):
            if time.time() > deadline:
                break
            
            # Try random subsets of size k
            for _ in range(100):
                if time.time() > deadline:
                    break
                
                # Randomly choose k vertices to connect to
                if k == 0:
                    pattern = [False] * current_n
                else:
                    chosen = set()
                    while len(chosen) < k:
                        chosen.add(__import__('random').randint(0, current_n - 1))
                    pattern = [v in chosen for v in range(current_n)]
                
                success, new_n, new_adj = _try_add_vertex(current_n, current_adj, pattern)
                if success:
                    current_adj = new_adj
                    current_n = new_n
                    found = True
                    if current_n > best_n:
                        best_n = current_n
                        best_adj = [set(s) for s in current_adj]
                    attempts += 1
                    break
            
            if found:
                break
        
        if not found:
            # Try random patterns
            for _ in range(500):
                if time.time() > deadline:
                    break
                
                import random
                pattern = [random.random() < 0.5 for _ in range(current_n)]
                
                success, new_n, new_adj = _try_add_vertex(current_n, current_adj, pattern)
                if success:
                    current_adj = new_adj
                    current_n = new_n
                    found = True
                    if current_n > best_n:
                        best_n = current_n
                        best_adj = [set(s) for s in current_adj]
                    attempts += 1
                    break
        
        if not found:
            break
    
    # Reconstruct edges from best_adj
    best_edges = []
    for i in range(best_n):
        for j in best_adj[i]:
            if i < j:
                best_edges.append((i, j))
    
    return best_n, best_edges
# EVOLVE-BLOCK-END
