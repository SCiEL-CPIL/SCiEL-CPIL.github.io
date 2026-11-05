"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import itertools
import time
from collections import defaultdict

def _has_k_clique(adj, n, k):
    """Check if graph has a clique of size k using backtracking."""
    # Simple backtracking for small k
    def backtrack(start, current_clique):
        if len(current_clique) == k:
            return True
        # Pruning: not enough vertices left
        if start + (k - len(current_clique)) > n:
            return False
        for v in range(start, n):
            if v in current_clique:
                continue
            # Check if v connects to all in current_clique
            if all(adj[v][u] for u in current_clique):
                if backtrack(v + 1, current_clique | {v}):
                    return True
        return False
    
    return backtrack(0, frozenset())

def _has_k_independent(adj, n, k):
    """Check if graph has an independent set of size k (equivalent to k-clique in complement)."""
    # Build complement adjacency
    comp_adj = [[not adj[i][j] and i != j for j in range(n)] for i in range(n)]
    return _has_k_clique(comp_adj, n, k)

def _is_valid(adj, n, clique_size=5, indep_size=5):
    """Check if graph has no clique of size clique_size and no independent set of size indep_size."""
    if _has_k_clique(adj, n, clique_size):
        return False
    if _has_k_independent(adj, n, indep_size):
        return False
    return True

def _count_cliques(adj, n, k):
    """Count number of k-cliques."""
    count = 0
    def backtrack(start, current):
        nonlocal count
        if len(current) == k:
            count += 1
            return
        if start + (k - len(current)) > n:
            return
        for v in range(start, n):
            if all(adj[v][u] for u in current):
                backtrack(v + 1, current + [v])
    backtrack(0, [])
    return count

def _count_independent(adj, n, k):
    """Count number of independent sets of size k."""
    comp_adj = [[not adj[i][j] and i != j for j in range(n)] for i in range(n)]
    return _count_cliques(comp_adj, n, k)

def _random_perturbation(adj, n, num_perturbations=50):
    """Apply random edge flips and keep if still valid."""
    best_adj = [row[:] for row in adj]
    best_score = 0
    
    for _ in range(num_perturbations):
        new_adj = [row[:] for row in best_adj]
        # Flip a few random edges
        for _ in range(3):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j:
                i, j = min(i, j), max(i, j)
                new_adj[i][j] = not new_adj[i][j]
                new_adj[j][i] = not new_adj[j][i]
        
        if _is_valid(new_adj, n):
            # Count violations (should be 0 for valid, but let's check margin)
            # Actually, we just need valid. Let's try to add vertices later.
            best_adj = [row[:] for row in new_adj]
            best_score += 1
            break  # Keep first valid perturbation to avoid wasting time
    
    return best_adj

def _try_add_vertex(adj, n, new_vertex_id):
    """Try to add a new vertex to the graph while maintaining (5,5)-free property."""
    # The new vertex must not form a K5 with any 4-clique in the existing graph
    # and must not form an independent 5-set with any independent 4-set
    
    # First, find all 4-cliques and independent 4-sets
    # This is expensive, so let's use a heuristic approach
    
    # For each possible connection pattern, check if it's valid
    # A vertex v is "safe" to add with a given neighborhood if:
    # 1. No 4-clique in the graph is fully connected to v (would form K5)
    # 2. No independent 4-set in the graph is fully non-connected to v (would form indep 5-set)
    
    # Let's try a random subset of connections
    for attempt in range(100):
        # Randomly decide connections
        connections = [random.random() < 0.5 for _ in range(n)]
        
        # Check condition 1: no 4-clique all connected to v
        # Enumerate all 4-cliques
        cliques_4 = []
        def find_cliques_4(start, current):
            if len(current) == 4:
                cliques_4.append(current[:])
                return
            if start + (4 - len(current)) > n:
                return
            for v in range(start, n):
                if all(adj[v][u] for u in current):
                    current.append(v)
                    find_cliques_4(v + 1, current)
                    current.pop()
        
        find_cliques_4(0, [])
        
        valid = True
        for clique in cliques_4:
            if all(connections[v] for v in clique):
                valid = False
                break
        
        if not valid:
            continue
        
        # Check condition 2: no independent 4-set all non-connected to v
        indep_4 = []
        def find_indep_4(start, current):
            if len(current) == 4:
                indep_4.append(current[:])
                return
            if start + (4 - len(current)) > n:
                return
            for v in range(start, n):
                if all(not adj[v][u] for u in current):
                    current.append(v)
                    find_indep_4(v + 1, current)
                    current.pop()
        
        find_indep_4(0, [])
        
        for indep_set in indep_4:
            if all(not connections[v] for v in indep_set):
                valid = False
                break
        
        if valid:
            # Extend adjacency matrix
            new_adj = [row + [connections[i] for i in range(n)] for row in adj]
            # Wait, need to be careful with indexing
            new_adj = [row[:] for row in adj]
            for i in range(n):
                new_adj[i].append(connections[i])
            new_adj.append(connections[:])
            # Make symmetric
            for i in range(n):
                for j in range(n):
                    new_adj[i][j] = adj[i][j]
            for i in range(n):
                new_adj[n][i] = connections[i]
                new_adj[i][n] = connections[i]
            new_adj[n][n] = False
            
            return new_adj
    
    return None

def construct():
    # Start with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    
    # Build adjacency matrix
    adj = [[False] * q for _ in range(q)]
    for i, j in edges:
        adj[i][j] = True
        adj[j][i] = True
    
    n = q
    start_time = time.time()
    
    # Try to add vertices one by one
    for _ in range(10):  # Try to add up to 10 more vertices
        if time.time() - start_time > 250:
            break
        
        new_adj = _try_add_vertex(adj, n, n)
        if new_adj is not None:
            adj = new_adj
            n += 1
    
    # Build edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                edges.append((i, j))
    
    return n, edges

# EVOLVE-BLOCK-END
