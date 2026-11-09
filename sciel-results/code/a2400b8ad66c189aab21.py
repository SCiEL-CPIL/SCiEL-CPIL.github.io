import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

DEADLINE = time.time() + 270  # Safety margin

def _build_paley(q):
    """Return adjacency bitset list for Paley graph on q vertices (vertices 0..q-1)."""
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _has_k_clique(adj, n, k):
    """Check if graph has a clique of size k. Returns True if found."""
    # Simple backtracking search
    # For k=5, this is feasible for n=41
    
    def search(current_clique, candidates):
        if len(current_clique) == k:
            return True
        if len(candidates) < k - len(current_clique):
            return False
            
        for i, v in enumerate(candidates):
            # New candidates are those in candidates[i+1:] that are adjacent to v
            new_candidates = [u for u in candidates[i+1:] if (adj[v] & (1 << u))]
            if search(current_clique + [v], new_candidates):
                return True
        return False

    return search([], list(range(n)))

def _has_ind_clique(adj, n, k):
    """Check if graph has an independent set of size k. Returns True if found."""
    # Independent set in G is Clique in complement of G
    
    # Build complement adj
    comp_adj = [0] * n
    for i in range(n):
        # All bits set for n vertices
        all_bits = (1 << n) - 1
        # Remove self and neighbors
        comp_adj[i] = all_bits & ~(adj[i] | (1 << i))
        
    return _has_k_clique(comp_adj, n, k)

def _is_valid(adj, n, k=5):
    """Check if graph is (k,k)-free."""
    if _has_k_clique(adj, n, k):
        return False
    if _has_ind_clique(adj, n, k):
        return False
    return True

def construct():
    # Try Paley(41) first
    # If it is valid, it's a great improvement over 37
    q = 41
    adj41 = _build_paley(q)
    if _is_valid(adj41, q, 5):
        # Build edge list
        edges = []
        for i in range(q):
            for j in range(i + 1, q):
                if adj41[i] & (1 << j):
                    edges.append((i, j))
        return q, edges

    # Fallback to Paley(37) which is known to be valid
    q = 37
    adj37 = _build_paley(q)
    # Just in case, verify (though it should be)
    # if not _is_valid(adj37, q, 5):
    #     # This should never happen, but handle gracefully
    #     pass
    
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if adj37[i] & (1 << j):
                edges.append((i, j))
    return q, edges

# EVOLVE-BLOCK-END
