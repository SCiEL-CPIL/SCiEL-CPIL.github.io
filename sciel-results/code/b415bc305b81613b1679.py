"""Local search to grow a (5,5)-free graph beyond the Paley seed."""
import time
import random
from itertools import combinations

SEED_TIME_LIMIT = 280  # seconds, safety margin before 300s kill


def _paley_graph(q):
    residues = {(x * x) % q for x in range(1, q)}
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    return q, edges


def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    return adj


def _has_k_clique_or_independent(n, adj, k=5):
    """Check if graph has a k-clique or k-independent set. Returns (is_valid, witness)."""
    # Check for k-clique
    for combo in combinations(range(n), k):
        is_clique = True
        for a, b in combinations(combo, 2):
            if b not in adj[a]:
                is_clique = False
                break
        if is_clique:
            return False, ('clique', combo)
    # Check for k-independent set
    for combo in combinations(range(n), k):
        is_indep = True
        for a, b in combinations(combo, 2):
            if b in adj[a]:
                is_indep = False
                break
        if is_indep:
            return False, ('independent', combo)
    return True, None


def _find_violation(n, adj, k=5):
    """Find a k-clique or k-independent set. Returns (type, set) or (None, None)."""
    for combo in combinations(range(n), k):
        is_clique = True
        is_indep = True
        for a, b in combinations(combo, 2):
            if b in adj[a]:
                is_indep = False
            else:
                is_clique = False
            if not is_clique and not is_indep:
                break
        if is_clique:
            return 'clique', combo
        if is_indep:
            return 'independent', combo
    return None, None


def _count_violations_fast(n, adj, k=5):
    """Count number of k-cliques and k-independent sets (for scoring)."""
    cliques = 0
    indeps = 0
    for combo in combinations(range(n), k):
        is_clique = True
        is_indep = True
        for a, b in combinations(combo, 2):
            if b in adj[a]:
                is_indep = False
            else:
                is_clique = False
            if not is_clique and not is_indep:
                break
        if is_clique:
            cliques += 1
        if is_indep:
            indeps += 1
    return cliques + indeps


def _try_add_vertex(n, adj, deadline, rng):
    """
    Try to add a new vertex (index n) to the graph.
    Returns True if successful, False if all attempts failed.
    """
    # Strategy: try random edge patterns, check validity
    # For each potential neighbor set, check if adding vertex creates violations
    
    # Precompute: for each 4-set, determine if it's a clique or independent
    # A new vertex connected to a 4-clique creates a 5-clique
    # A new vertex not connected to a 4-independent set creates a 5-independent set
    
    # We need: for every 4-clique, new vertex must NOT be connected to all 4
    # We need: for every 4-independent set, new vertex must be connected to at least 1
    
    # Build lists of 4-cliques and 4-independent sets
    four_cliques = []
    four_indeps = []
    for combo in combinations(range(n), 4):
        is_clique = True
        is_indep = True
        for a, b in combinations(combo, 2):
            if b in adj[a]:
                is_indep = False
            else:
                is_clique = False
            if not is_clique and not is_indep:
                break
        if is_clique:
            four_cliques.append(set(combo))
        if is_indep:
            four_indeps.append(set(combo))
    
    # Now find an assignment of edges from new vertex to existing vertices
    # such that:
    # - No 4-clique is fully connected to new vertex
    # - Every 4-independent set has at least one edge to new vertex
    
    # This is a constraint satisfaction problem. Try random assignments.
    for attempt in range(200):
        if time.time() > deadline:
            return False
        
        # Random assignment
        new_adj = set()
        for v in range(n):
            if rng.random() < 0.5:
                new_adj.add(v)
        
        # Check constraints
        valid = True
        for fc in four_cliques:
            if fc.issubset(new_adj):
                valid = False
                break
        if not valid:
            continue
        for fi in four_indeps:
            if not (fi & new_adj):
                valid = False
                break
        if not valid:
            continue
        
        # Valid! Add the vertex
        adj.append(new_adj)
        for v in new_adj:
            adj[v].add(n)
        return True
    
    return False


def _local_search_repair(n, adj, deadline, rng, max_iter=5000):
    """
    If the graph has violations, try to repair by flipping edges.
    Returns True if repaired, False otherwise.
    """
    for it in range(max_iter):
        if time.time() > deadline:
            return False
        
        vtype, witness = _find_violation(n, adj, 5)
        if vtype is None:
            return True
        
        # Try to fix by flipping one edge in the witness
        # If it's a clique, remove an edge from the witness
        # If it's independent, add an edge to the witness
        if vtype == 'clique':
            candidates = list(combinations(witness, 2))
            rng.shuffle(candidates)
            for a, b in candidates:
                # Try removing edge (a,b)
                adj[a].remove(b)
                adj[b].remove(a)
                # Check if this fixes things without creating new violations
                vtype2, witness2 = _find_violation(n, adj, 5)
                if vtype2 is None:
                    return True
                # Revert
                adj[a].add(b)
                adj[b].add(a)
        else:  # independent
            candidates = list(combinations(witness, 2))
            rng.shuffle(candidates)
            for a, b in candidates:
                # Try adding edge (a,b)
                adj[a].add(b)
                adj[b].add(a)
                vtype2, witness2 = _find_violation(n, adj, 5)
                if vtype2 is None:
                    return True
                # Revert
                adj[a].remove(b)
                adj[b].remove(a)
    
    return False


def construct():
    start_time = time.time()
    deadline = start_time + SEED_TIME_LIMIT
    
    rng = random.Random(42)
    
    # Start with Paley graph on 37 vertices
    n, edges = _paley_graph(37)
    adj = _build_adj(n, edges)
    
    # Verify seed is valid
    valid, _ = _has_k_clique_or_independent(n, adj, 5)
    assert valid, "Paley graph should be (5,5)-free"
    
    best_n = n
    best_adj = [set(s) for s in adj]
    
    # Try to grow the graph
    for target_n in range(n + 1, 47):
        if time.time() > deadline - 30:
            break
        
        success = _try_add_vertex(n, adj, deadline, rng)
        if success:
            n += 1
            # Verify
            valid, _ = _has_k_clique_or_independent(n, adj, 5)
            if valid:
                best_n = n
                best_adj = [set(s) for s in adj]
            else:
                # Try to repair
                if _local_search_repair(n, adj, deadline, rng):
                    valid, _ = _has_k_clique_or_independent(n, adj, 5)
                    if valid:
                        best_n = n
                        best_adj = [set(s) for s in adj]
                    else:
                        # Revert
                        adj.pop()
                        for v in range(n - 1):
                            adj[v].discard(n - 1)
                        n -= 1
                else:
                    # Revert
                    adj.pop()
                    for v in range(n - 1):
                        adj[v].discard(n - 1)
                    n -= 1
        else:
            # Try a few more times with different random seeds
            for retry in range(10):
                if time.time() > deadline - 30:
                    break
                success = _try_add_vertex(n, adj, deadline, rng)
                if success:
                    n += 1
                    valid, _ = _has_k_clique_or_independent(n, adj, 5)
                    if valid:
                        best_n = n
                        best_adj = [set(s) for s in adj]
                        break
                    else:
                        if _local_search_repair(n, adj, deadline, rng):
                            valid, _ = _has_k_clique_or_independent(n, adj, 5)
                            if valid:
                                best_n = n
                                best_adj = [set(s) for s in adj]
                                break
                        # Revert
                        adj.pop()
                        for v in range(n - 1):
                            adj[v].discard(n - 1)
                        n -= 1
    
    # Also try local search on the current graph to see if we can improve
    # (flip edges to allow further growth)
    if time.time() < deadline - 60 and best_n < 42:
        # Try flipping random edges to create more flexibility
        for flip_iter in range(20000):
            if time.time() > deadline - 60:
                break
            
            # Pick a random edge and flip it
            if len(best_adj) > 0:
                v = rng.randint(0, best_n - 1)
                if len(best_adj[v]) > 0:
                    u = rng.choice(list(best_adj[v]))
                    # Flip edge (v, u)
                    if u in best_adj[v]:
                        best_adj[v].remove(u)
                        best_adj[u].remove(v)
                    else:
                        best_adj[v].add(u)
                        best_adj[u].add(v)
                    
                    # Check validity
                    valid, _ = _has_k_clique_or_independent(best_n, best_adj, 5)
                    if not valid:
                        # Revert
                        if u in best_adj[v]:
                            best_adj[v].remove(u)
                            best_adj[u].remove(v)
                        else:
                            best_adj[v].add(u)
                            best_adj[u].add(v)
    
    # Build final edge list
    final_edges = []
    for i in range(best_n):
        for j in best_adj[i]:
            if i < j:
                final_edges.append((i, j))
    
    return best_n, final_edges


if __name__ == '__main__':
    n, edges = construct()
    print(f"Constructed graph with {n} vertices and {len(edges)} edges")
