"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys
from itertools import combinations

def _paley_graph(q):
    """Return adjacency matrix (list of sets) for Paley graph on q vertices."""
    residues = {(x * x) % q for x in range(1, q)}
    adj = [set() for _ in range(q)]
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i].add(j)
                adj[j].add(i)
    return adj

def _has_k5(adj, n):
    """Check if graph contains a K5. Returns the K5 if found, else None."""
    # For each vertex, check if it has a K4 in its neighborhood
    for v in range(n):
        nbrs = sorted(adj[v])
        # Check all K4s in the neighborhood
        for combo in combinations(nbrs, 4):
            a, b, c, d = combo
            if b in adj[a] and c in adj[a] and d in adj[a] and \
               c in adj[b] and d in adj[b] and \
               d in adj[c]:
                return [v, a, b, c, d]
    return None

def _has_independent_5(adj, n):
    """Check if graph contains an independent set of size 5."""
    # An independent set of size 5 in G is a K5 in complement
    # Check all 5-subsets (expensive but n is small)
    # Better: for each vertex, check if it has an independent 4-set in non-neighbors
    for v in range(n):
        non_nbrs = [u for u in range(n) if u != v and u not in adj[v]]
        # Check all 4-subsets of non_nbrs for independence
        for combo in combinations(non_nbrs, 4):
            a, b, c, d = combo
            if b not in adj[a] and c not in adj[a] and d not in adj[a] and \
               c not in adj[b] and d not in adj[b] and \
               d not in adj[c]:
                return [v, a, b, c, d]
    return None

def _is_valid(adj, n):
    """Check if graph is (5,5)-free."""
    if _has_k5(adj, n) is not None:
        return False
    if _has_independent_5(adj, n) is not None:
        return False
    return True

def _violation_count(adj, n):
    """Count number of K5s and independent 5-sets."""
    count = 0
    # Count K5s
    for v in range(n):
        nbrs = sorted(adj[v])
        for combo in combinations(nbrs, 4):
            a, b, c, d = combo
            if b in adj[a] and c in adj[a] and d in adj[a] and \
               c in adj[b] and d in adj[b] and \
               d in adj[c]:
                count += 1
    # Count independent 5-sets
    for v in range(n):
        non_nbrs = [u for u in range(n) if u != v and u not in adj[v]]
        for combo in combinations(non_nbrs, 4):
            a, b, c, d = combo
            if b not in adj[a] and c not in adj[a] and d not in adj[a] and \
               c not in adj[b] and d not in adj[b] and \
               d not in adj[c]:
                count += 1
    return count

def _try_add_vertex(adj, n):
    """Try to add a new vertex to the graph while maintaining (5,5)-free property."""
    # Try adding vertex n with various adjacency patterns
    # The new vertex must not create a K5 or independent 5-set
    # For K5: new vertex + 4 vertices forming K4 in its neighborhood
    # For independent 5: new vertex + 4 vertices forming independent 4-set in its non-neighborhood
    
    # Strategy: try adding with a random subset of existing vertices
    # that avoids creating violations
    
    # First, determine which subsets are "safe" for the new vertex
    # A subset S of existing vertices is safe if:
    # 1. S contains no K4 (else new vertex + K4 = K5)
    # 2. complement of S (in existing vertices) contains no independent 4-set
    
    # This is complex, so we use a simpler heuristic:
    # Try adding with a random pattern, check validity
    
    for attempt in range(100):
        new_adj = [set() for _ in range(n + 1)]
        for i in range(n):
            new_adj[i] = set(adj[i])
        new_adj[n] = set()
        
        # Randomly connect new vertex to existing vertices
        for i in range(n):
            if random.random() < 0.5:
                new_adj[i].add(n)
                new_adj[n].add(i)
        
        if _is_valid(new_adj, n + 1):
            return new_adj
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 10 second safety margin
    
    # Start with Paley graph on 37 vertices
    q = 37
    adj = _paley_graph(q)
    n = q
    
    # Verify initial graph is valid
    if not _is_valid(adj, n):
        # This should not happen, but handle it
        return n, [(i, j) for i in range(n) for j in range(i + 1, n) if j in adj[i]]
    
    best_n = n
    best_adj = [set(s) for s in adj]
    
    # Try to grow the graph by adding vertices
    while time.time() < deadline:
        if n >= 42:
            break
        
        new_adj = _try_add_vertex(adj, n)
        if new_adj is not None:
            adj = new_adj
            n += 1
            best_n = n
            best_adj = [set(s) for s in adj]
        else:
            # Try local search to find a valid configuration
            # that allows adding a vertex
            found = False
            for _ in range(50):
                if time.time() >= deadline:
                    break
                
                # Try flipping a random edge
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                
                if j in adj[i]:
                    adj[i].discard(j)
                    adj[j].discard(i)
                else:
                    adj[i].add(j)
                    adj[j].add(i)
                
                # Check if we can now add a vertex
                new_adj = _try_add_vertex(adj, n)
                if new_adj is not None:
                    adj = new_adj
                    n += 1
                    best_n = n
                    best_adj = [set(s) for s in adj]
                    found = True
                    break
                else:
                    # Revert the flip
                    if j in adj[i]:
                        adj[i].discard(j)
                        adj[j].discard(i)
                    else:
                        adj[i].add(j)
                        adj[j].add(i)
            
            if not found and n == best_n:
                # Stuck, try a different approach
                # Reset to best and try again with different random seed
                if random.random() < 0.1:
                    adj = [set(s) for s in best_adj]
                    n = best_n
    
    # Return the best graph found
    edges = [(i, j) for i in range(best_n) for j in range(i + 1, best_n) if j in best_adj[i]]
    return best_n, edges

# EVOLVE-BLOCK-END
