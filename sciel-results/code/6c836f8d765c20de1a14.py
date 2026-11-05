"""Improved Ramsey graph construction for R(5,5)."""
import time
import random
import sys
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    """Construct a graph with no K5 and no independent set of size 5.
    Returns (n, edges) where edges is a list of (i, j) pairs with 0 <= i < j < n.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    k = 5
    
    best_n = 0
    best_adj = None
    
    # Strategy 1: Start with Paley graph on 37 vertices (known to be (5,5)-free)
    best_adj = _paley_graph(37)
    best_n = 37
    
    # Verify it's valid
    if not _is_valid_fast(best_adj, best_n, k):
        # Fallback: try smaller Paley graphs
        for q in [29, 13, 5]:
            if q % 4 == 1:
                adj = _paley_graph(q)
                if _is_valid_fast(adj, q, k):
                    best_adj = adj
                    best_n = q
                    break
    
    # Strategy 2: Try to extend the Paley graph by adding vertices one at a time
    # Use a smarter approach: for each new vertex, find a connection pattern
    # that doesn't create a K5 or independent 5-set
    
    for target_n in range(best_n + 1, 47):
        if time.time() > deadline:
            break
        
        found = False
        # Try many random connection patterns for the new vertex
        for attempt in range(500):
            if time.time() > deadline:
                break
            
            candidate_adj = [s.copy() for s in best_adj]
            candidate_adj.append(set())
            new_v = best_n
            
            # Random connection pattern
            for i in range(best_n):
                if random.random() < 0.5:
                    candidate_adj[i].add(new_v)
                    candidate_adj[new_v].add(i)
            
            if _is_valid_fast(candidate_adj, target_n, k):
                best_adj = candidate_adj
                best_n = target_n
                found = True
                break
        
        if not found:
            # Try a more systematic approach: try all subsets of neighbors
            # that are "safe" (don't create K5 or independent 5-set)
            if _try_add_vertex_systematic(best_adj, best_n, k):
                best_n += 1
                found = True
            else:
                break
    
    # Strategy 3: Local search - try to improve by flipping edges
    if time.time() < deadline - 5:
        improved_adj, improved_n = _local_search_edges(best_adj, best_n, k, deadline)
        if improved_n > best_n:
            best_adj = improved_adj
            best_n = improved_n
    
    # Strategy 4: Try different starting points (other Paley graphs, random graphs)
    if time.time() < deadline - 10:
        for q in [29, 37, 41]:
            if time.time() > deadline:
                break
            if q % 4 != 1:
                continue
            paley_adj = _paley_graph(q)
            if _is_valid_fast(paley_adj, q, k):
                if q > best_n:
                    best_adj = paley_adj
                    best_n = q
                # Try to extend from this starting point
                for target_n in range(best_n + 1, 47):
                    if time.time() > deadline:
                        break
                    for attempt in range(200):
                        if time.time() > deadline:
                            break
                        candidate_adj = [s.copy() for s in best_adj]
                        candidate_adj.append(set())
                        new_v = best_n
                        for i in range(best_n):
                            if random.random() < 0.5:
                                candidate_adj[i].add(new_v)
                                candidate_adj[new_v].add(i)
                        if _is_valid_fast(candidate_adj, target_n, k):
                            best_adj = candidate_adj
                            best_n = target_n
                            break
                    else:
                        continue
                    break
    
    # Generate edge list
    edges = [(i, j) for i in range(best_n) for j in range(i + 1, best_n) if j in best_adj[i]]
    
    return best_n, edges


def _paley_graph(n):
    """Generate a Paley graph on n vertices (n must be a prime congruent to 1 mod 4)."""
    residues = {(x * x) % n for x in range(1, n)}
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n in residues:
                adj[i].add(j)
                adj[j].add(i)
    return adj


def _is_valid_fast(adj, n, k):
    """Check if graph is (k,k)-free using efficient backtracking."""
    # Check for clique of size k
    if _has_clique(adj, n, k):
        return False
    # Check for independent set of size k (clique in complement)
    if _has_independent(adj, n, k):
        return False
    return True


def _has_clique(adj, n, k):
    """Check if graph has a clique of size k using backtracking with pruning."""
    # Use bitmasks for efficiency
    adj_mask = [0] * n
    for i in range(n):
        for j in adj[i]:
            adj_mask[i] |= (1 << j)
    
    def backtrack(start, size, current_mask):
        if size == 0:
            return True
        if start >= n:
            return False
        # Pruning: not enough vertices left
        if n - start < size:
            return False
        
        # Find candidates: vertices from start to n-1 that are connected to all in current
        candidates = []
        for v in range(start, n):
            if (current_mask & ~adj_mask[v]) == 0:  # v is connected to all in current
                candidates.append(v)
        
        if len(candidates) < size:
            return False
        
        for v in candidates:
            new_mask = current_mask | (1 << v)
            if backtrack(v + 1, size - 1, new_mask):
                return True
        return False
    
    return backtrack(0, k, 0)


def _has_independent(adj, n, k):
    """Check if graph has an independent set of size k (clique in complement)."""
    # Build complement adjacency masks
    comp_mask = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and j not in adj[i]:
                comp_mask[i] |= (1 << j)
    
    def backtrack(start, size, current_mask):
        if size == 0:
            return True
        if start >= n:
            return False
        if n - start < size:
            return False
        
        candidates = []
        for v in range(start, n):
            if (current_mask & ~comp_mask[v]) == 0:  # v is non-adjacent to all in current
                candidates.append(v)
        
        if len(candidates) < size:
            return False
        
        for v in candidates:
            new_mask = current_mask | (1 << v)
            if backtrack(v + 1, size - 1, new_mask):
                return True
        return False
    
    return backtrack(0, k, 0)


def _try_add_vertex_systematic(adj, n, k):
    """Try to add a vertex by finding a valid connection pattern."""
    # For the new vertex, try different numbers of neighbors
    # Start with ~n/2 neighbors and adjust
    
    for num_neighbors in range(n // 2 - 2, n // 2 + 3):
        if num_neighbors < 0 or num_neighbors > n:
            continue
        
        # Try random subsets of that size
        for _ in range(50):
            neighbor_set = set(random.sample(range(n), num_neighbors))
            new_adj = [s.copy() for s in adj]
            new_adj.append(neighbor_set)
            for i in neighbor_set:
                new_adj[i].add(n)
            
            if _is_valid_fast(new_adj, n + 1, k):
                adj.clear()
                adj.extend(new_adj)
                return True
    
    return False


def _local_search_edges(adj, n, k, deadline):
    """Local search: try flipping edges to enable adding more vertices."""
    best_adj = [s.copy() for s in adj]
    best_n = n
    
    # Try to add vertices by first modifying the graph
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        
        # Try adding a vertex with various connection patterns
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            candidate_adj = [s.copy() for s in best_adj]
            candidate_adj.append(set())
            new_v = best_n
            
            # Random connection pattern
            for i in range(best_n):
                if random.random() < 0.5:
                    candidate_adj[i].add(new_v)
                    candidate_adj[new_v].add(i)
            
            if _is_valid_fast(candidate_adj, best_n + 1, k):
                best_adj = candidate_adj
                best_n += 1
                break
    
    return best_adj, best_n

# EVOLVE-BLOCK-END
