"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def _has_k_clique_or_independent(adj, n, k):
    """Check if graph has a clique or independent set of size k.
    adj: list of sets of neighbors
    Returns (True, witness) or (False, None)
    """
    # Check for clique of size k
    for i in range(n):
        # Try to build clique starting from i
        stack = [(i, [i], set(adj[i]))]
        while stack:
            last, clique, candidates = stack.pop()
            if len(clique) == k:
                return True, clique
            if not candidates:
                continue
            for v in list(candidates):
                new_clique = clique + [v]
                new_cands = candidates & adj[v]
                if len(new_cands) >= k - len(new_clique):
                    stack.append((v, new_clique, new_cands))
    
    # Check for independent set of size k (clique in complement)
    for i in range(n):
        non_neighbors = set(range(n)) - adj[i] - {i}
        stack = [(i, [i], non_neighbors)]
        while stack:
            last, indep, candidates = stack.pop()
            if len(indep) == k:
                return True, indep
            if not candidates:
                continue
            for v in list(candidates):
                new_indep = indep + [v]
                new_cands = candidates & (set(range(n)) - adj[v] - {v})
                if len(new_cands) >= k - len(new_indep):
                    stack.append((v, new_indep, new_cands))
    
    return False, None

def _is_valid(adj, n, k=5):
    """Check if graph is k-free (no clique or independent set of size k)."""
    has_violation, _ = _has_k_clique_or_independent(adj, n, k)
    return not has_violation

def _paley_graph(q):
    """Construct Paley graph on q vertices."""
    residues = {(x * x) % q for x in range(1, q)}
    adj = [set() for _ in range(q)]
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i].add(j)
                adj[j].add(i)
                edges.append((i, j))
    return q, adj, edges

def _local_search(adj, n, k=5, max_iters=5000, deadline=None):
    """Local search to fix violations by flipping edges.
    Returns (success, adj, n)
    """
    if deadline is not None:
        if time.time() > deadline:
            return False, adj, n
    
    for it in range(max_iters):
        if deadline is not None:
            if time.time() > deadline:
                return False, adj, n
        
        # Check if valid
        if _is_valid(adj, n, k):
            return True, adj, n
        
        # Find a violation and try to fix it by flipping an edge
        # Strategy: find a clique or independent set of size k, 
        # then flip an edge within it to break it
        has_violation, witness = _has_k_clique_or_independent(adj, n, k)
        if not has_violation:
            return True, adj, n
        
        # witness is either a clique or independent set of size k
        # Try flipping edges to break this violation
        # If it's a clique, we need to remove an edge
        # If it's an independent set, we need to add an edge
        
        # Determine if it's a clique or independent set
        is_clique = all(w in adj[v] for v in witness[:-1] for w in witness[witness.index(v)+1:])
        
        if is_clique:
            # Remove a random edge from the clique
            i, j = random.sample(witness, 2)
            if j in adj[i]:
                adj[i].remove(j)
                adj[j].remove(i)
        else:
            # Add a random edge to the independent set
            i, j = random.sample(witness, 2)
            if j not in adj[i]:
                adj[i].add(j)
                adj[j].add(i)
    
    return _is_valid(adj, n, k), adj, n

def _try_add_vertex(adj, n, k=5, deadline=None):
    """Try to add a new vertex to the graph.
    Returns (success, adj, n_new)
    """
    if deadline is not None:
        if time.time() > deadline:
            return False, adj, n
    
    # Create a new vertex with random connections
    best_adj = None
    best_n = n
    
    for attempt in range(10):
        if deadline is not None:
            if time.time() > deadline:
                break
        
        new_adj = [set(s) for s in adj]
        new_adj.append(set())
        
        # Randomly connect new vertex to existing vertices
        for i in range(n):
            if random.random() < 0.5:
                new_adj[i].add(n)
                new_adj[n].add(i)
        
        # Try local search to fix violations
        success, fixed_adj, new_n = _local_search(new_adj, n + 1, k, max_iters=2000, deadline=deadline)
        
        if success:
            return True, fixed_adj, new_n
    
    return False, adj, n

def construct():
    """Construct a (5,5)-free graph with as many vertices as possible."""
    deadline = time.time() + 280  # Safety margin
    
    # Start with Paley graph on 37 vertices
    n, adj, edges = _paley_graph(37)
    
    best_n = n
    best_adj = [set(s) for s in adj]
    
    # Try to grow the graph
    for grow_iter in range(50):
        if time.time() > deadline:
            break
        
        success, new_adj, new_n = _try_add_vertex(adj, n, k=5, deadline=deadline)
        
        if success and new_n > best_n:
            n = new_n
            adj = new_adj
            best_n = new_n
            best_adj = [set(s) for s in adj]
        else:
            # If we can't grow, try local search on current graph to find better configuration
            if time.time() > deadline:
                break
            success, improved_adj, improved_n = _local_search(adj, n, k=5, max_iters=10000, deadline=deadline)
            if success:
                adj = improved_adj
                best_adj = [set(s) for s in adj]
    
    # Convert back to edge list
    final_edges = []
    for i in range(best_n):
        for j in best_adj[i]:
            if i < j:
                final_edges.append((i, j))
    
    return best_n, final_edges

# EVOLVE-BLOCK-END
