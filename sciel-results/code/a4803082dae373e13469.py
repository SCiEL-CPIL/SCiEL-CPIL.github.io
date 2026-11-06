import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 280  # 280s deadline, 20s margin
    
    # Start with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    
    def is_residue(d, q=q):
        return d % q in residues
    
    # Build adjacency matrix for Paley 37
    n_base = q
    adj = np.zeros((n_base, n_base), dtype=np.uint8)
    for i in range(n_base):
        for j in range(i+1, n_base):
            if is_residue(j - i):
                adj[i, j] = 1
                adj[j, i] = 1
    
    def has_k5(adj, n):
        """Check if graph has a clique of size 5."""
        # For each 4-clique, check if any vertex connects to all 4
        # More efficient: check all 5-subsets
        # This is expensive for large n, so use it carefully
        for subset in combinations(range(n), 5):
            a = subset[0]; b = subset[1]; c = subset[2]; d = subset[3]; e = subset[4]
            if (adj[a,b] and adj[a,c] and adj[a,d] and adj[a,e] and
                adj[b,c] and adj[b,d] and adj[b,e] and
                adj[c,d] and adj[c,e] and
                adj[d,e]):
                return True
        return False
    
    def has_independent_5(adj, n):
        """Check if graph has an independent set of size 5."""
        # Independent set = clique in complement
        comp = 1 - adj
        np.fill_diagonal(comp, 0)
        for subset in combinations(range(n), 5):
            a = subset[0]; b = subset[1]; c = subset[2]; d = subset[3]; e = subset[4]
            if (comp[a,b] and comp[a,c] and comp[a,d] and comp[a,e] and
                comp[b,c] and comp[b,d] and comp[b,e] and
                comp[c,d] and comp[c,e] and
                comp[d,e]):
                return True
        return False
    
    def is_valid_55_free(adj, n):
        return not has_k5(adj, n) and not has_independent_5(adj, n)
    
    # Verify base graph
    if not is_valid_55_free(adj, n_base):
        # Should not happen, but fallback
        return n_base, [(i, j) for i in range(n_base) for j in range(i+1, n_base) if adj[i,j]]
    
    best_n = n_base
    best_adj = adj.copy()
    
    # Try to add vertices one at a time
    # For each new vertex, try all possible connection patterns to existing vertices
    # A connection pattern is a subset of existing vertices to connect to
    
    # For efficiency, when adding vertex v, we need:
    # - v's neighbors among existing vertices should not contain a K4
    # - v's non-neighbors among existing vertices should not contain an independent 4-set
    
    def can_add_vertex(adj, n, neighbor_set):
        """Check if adding a vertex with given neighbors to adj (size n) keeps (5,5)-free."""
        # Check K5: neighbor_set must not contain a K4
        # Check independent 5: non-neighbors must not contain an independent 4-set
        
        # Get neighbor set as list
        neighbors = list(neighbor_set)
        non_neighbors = [i for i in range(n) if i not in neighbor_set]
        
        # Check if neighbors contain a K4
        for quad in combinations(neighbors, 4):
            a, b, c, d = quad
            if (adj[a,b] and adj[a,c] and adj[a,d] and
                adj[b,c] and adj[b,d] and
                adj[c,d]):
                return False  # Would create K5
        
        # Check if non-neighbors contain an independent 4-set
        # Independent 4-set in original = K4 in complement
        for quad in combinations(non_neighbors, 4):
            a, b, c, d = quad
            if (not adj[a,b] and not adj[a,c] and not adj[a,d] and
                not adj[b,c] and not adj[b,d] and
                not adj[c,d]):
                return False  # Would create independent 5-set
        
        return True
    
    # Precompute all 4-cliques and 4-independent-sets for faster checking
    # For each subset of 4 vertices, precompute if it's a K4 or independent 4-set
    
    def precompute_k4s(adj, n):
        """Return set of frozensets of 4-cliques."""
        k4s = set()
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if (adj[a,b] and adj[a,c] and adj[a,d] and
                adj[b,c] and adj[b,d] and
                adj[c,d]):
                k4s.add(quad)
        return k4s
    
    def precompute_ind4s(adj, n):
        """Return set of frozensets of independent 4-sets."""
        ind4s = set()
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if (not adj[a,b] and not adj[a,c] and not adj[a,d] and
                not adj[b,c] and not adj[b,d] and
                not adj[c,d]):
                ind4s.add(quad)
        return ind4s
    
    # Try to extend to 42 vertices
    target_n = 42
    
    for target in range(n_base + 1, target_n + 1):
        if time.time() > deadline:
            break
        
        found = False
        # Try randomized connection patterns
        np.random.seed(target * 1000)
        for attempt in range(5000):
            if time.time() > deadline:
                break
            
            # Randomly choose which existing vertices to connect to
            # Bias: connect to about half
            prob = 0.5
            new_neighbors = set()
            for i in range(n_base if target == n_base + 1 else best_n):
                if np.random.random() < prob:
                    new_neighbors.add(i)
            
            if can_add_vertex(best_adj, best_n, new_neighbors):
                # Add the vertex
                new_adj = np.zeros((best_n + 1, best_n + 1), dtype=np.uint8)
                new_adj[:best_n, :best_n] = best_adj
                for i in new_neighbors:
                    new_adj[best_n, i] = 1
                    new_adj[i, best_n] = 1
                
                best_adj = new_adj
                best_n = best_n + 1
                found = True
                break
        
        if not found and target > n_base + 1:
            # Couldn't extend, try a different approach or stop
            # Try with different random seeds
            for seed in range(100):
                if time.time() > deadline:
                    break
                np.random.seed(seed)
                for attempt in range(2000):
                    if time.time() > deadline:
                        break
                    prob = 0.3 + 0.4 * np.random.random()
                    new_neighbors = set()
                    for i in range(best_n):
                        if np.random.random() < prob:
                            new_neighbors.add(i)
                    if can_add_vertex(best_adj, best_n, new_neighbors):
                        new_adj = np.zeros((best_n + 1, best_n + 1), dtype=np.uint8)
                        new_adj[:best_n, :best_n] = best_adj
                        for i in new_neighbors:
                            new_adj[best_n, i] = 1
                            new_adj[i, best_n] = 1
                        best_adj = new_adj
                        best_n = best_n + 1
                        found = True
                        break
                if found:
                    break
        
        if not found:
            # Can't extend further with this approach
            break
    
    # Final verification
    if not is_valid_55_free(best_adj, best_n):
        # Fallback to base
        return n_base, [(i, j) for i in range(n_base) for j in range(i+1, n_base) if adj[i,j]]
    
    # Convert to edge list
    edges = [(i, j) for i in range(best_n) for j in range(i+1, best_n) if best_adj[i,j]]
    return best_n, edges
