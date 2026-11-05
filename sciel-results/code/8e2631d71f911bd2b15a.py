import time
import random
from itertools import combinations

def _remaining_time(deadline):
    return deadline - time.time()

def _has_triangle_bitset(adj, n):
    """Check if graph has a triangle using bitsets."""
    for u in range(n):
        au = adj[u]
        # Only consider neighbors v > u to avoid duplicates
        v = u + 1
        while v < n:
            if au & (1 << v):
                common = au & adj[v]
                # Only consider w > v
                if common & ~((1 << (v + 1)) - 1):
                    return True
            v += 1
    return False

def _max_independent_set_size_bitset(adj, n, k):
    """
    Check if there exists an independent set of size k using bitmask backtracking.
    Returns True if found, False otherwise.
    Uses branch and bound with coloring heuristic for speed.
    """
    full_mask = (1 << n) - 1
    # non_neighbors[i] = vertices not adjacent to i (excluding i itself)
    non_neighbors = [0] * n
    for i in range(n):
        non_neighbors[i] = full_mask & ~adj[i] & ~(1 << i)
    
    found = [False]
    
    def dfs(candidates_mask, current_size):
        if found[0]:
            return
        if current_size == k:
            found[0] = True
            return
        
        # Prune: not enough candidates
        if candidates_mask == 0:
            return
        if bin(candidates_mask).count('1') < k - current_size:
            return
        
        # Get list of candidate vertices
        candidates = []
        mask = candidates_mask
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            candidates.append(v)
            mask ^= lsb
        
        # Heuristic: order by degree in candidate subgraph (ascending)
        # A vertex with few non-neighbors among candidates is more constrained
        # Actually, for finding independent set, we want to branch on vertices
        # that are most likely to be in the independent set.
        # Greedy coloring bound: color the candidate subgraph.
        # If max color < k - current_size, prune.
        
        # For speed, just use a simple heuristic: pick vertex with most neighbors
        # in candidates (i.e., fewest non-neighbors), so its branch prunes more.
        
        # Compute number of non-neighbors in candidates for each candidate
        best_v = None
        best_count = -1
        for v in candidates:
            nn_in_cands = bin(candidates_mask & non_neighbors[v]).count('1')
            if nn_in_cands > best_count:
                best_count = nn_in_cands
                best_v = v
        
        if best_v is None:
            return
            
        # Branch 1: Include best_v
        new_candidates = candidates_mask & non_neighbors[best_v] & ~(1 << best_v)
        dfs(new_candidates, current_size + 1)
        
        if found[0]:
            return
        
        # Branch 2: Exclude best_v
        dfs(candidates_mask & ~(1 << best_v), current_size)
    
    dfs(full_mask, 0)
    return found[0]

def _build_triangle_free_bitset(adj, n):
    """Remove edges to make the graph triangle-free."""
    for _ in range(10000):
        # Find a triangle
        tri_found = False
        for u in range(n):
            au = adj[u]
            v = u + 1
            while v < n:
                if au & (1 << v):
                    common = au & adj[v] & ~((1 << (v + 1)) - 1)
                    if common:
                        lsb = common & -common
                        w = lsb.bit_length() - 1
                        # Remove edge (u, v)
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        tri_found = True
                        break
                v += 1
            if tri_found:
                break
        if not tri_found:
            break

def _local_search_bitset(adj, n, target_alpha, deadline):
    """
    Try to reduce alpha below target_alpha by adding edges.
    Returns the final alpha estimate (if < target_alpha, it's valid).
    """
    for _ in range(5000):
        if time.time() > deadline:
            break
        
        # Check if there's an independent set of size target_alpha
        has_iso = _max_independent_set_size_bitset(adj, n, target_alpha)
        
        if not has_iso:
            return True  # Valid: alpha < target_alpha
        
        # Find an independent set of size target_alpha to add edges
        # We need to actually find the set, not just check existence
        # Let's modify to return the set
        
        iso = _find_independent_set_bitset(adj, n, target_alpha)
        if iso is None:
            return True
        
        # Try to add an edge between two vertices in iso that doesn't create a triangle
        improved = False
        random.shuffle(iso)
        for i in range(len(iso)):
            for j in range(i + 1, len(iso)):
                u, v = iso[i], iso[j]
                if not (adj[u] & (1 << v)):
                    common = adj[u] & adj[v]
                    if common == 0:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            # Try removing an edge that's in many independent sets?
            # Or just break and restart
            break
    
    # Final check
    return not _max_independent_set_size_bitset(adj, n, target_alpha)

def _find_independent_set_bitset(adj, n, k):
    """Find an independent set of size k. Returns list of vertices or None."""
    full_mask = (1 << n) - 1
    non_neighbors = [0] * n
    for i in range(n):
        non_neighbors[i] = full_mask & ~adj[i] & ~(1 << i)
    
    found = [False]
    result_set = []
    
    def dfs(candidates_mask, current_set):
        if found[0]:
            return
        if len(current_set) == k:
            result_set.extend(current_set)
            found[0] = True
            return
        if candidates_mask == 0:
            return
        if bin(candidates_mask).count('1') < k - len(current_set):
            return
        
        # Pick vertex with most non-neighbors in candidates
        mask = candidates_mask
        best_v = None
        best_count = -1
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            nn_count = bin(candidates_mask & non_neighbors[v]).count('1')
            if nn_count > best_count:
                best_count = nn_count
                best_v = v
            mask ^= lsb
        
        if best_v is None:
            return
        
        # Include
        new_cands = candidates_mask & non_neighbors[best_v] & ~(1 << best_v)
        dfs(new_cands, current_set + [best_v])
        
        if found[0]:
            return
        
        # Exclude
        dfs(candidates_mask & ~(1 << best_v), current_set)
    
    dfs(full_mask, [])
    if found[0]:
        return result_set
    return None

def _bitset_to_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges

def _circulant_graph(n, S):
    """
    Build a circulant graph on n vertices where vertex i is connected to
    i+s mod n and i-s mod n for each s in S.
    S should be a set of integers in {1, ..., n-1}.
    For symmetry, if s is in S, then n-s is automatically included.
    """
    adj = [0] * n
    S_symmetric = set()
    for s in S:
        s_mod = s % n
        if 0 < s_mod < n:
            S_symmetric.add(s_mod)
            S_symmetric.add((n - s_mod) % n)
    
    for i in range(n):
        for s in S_symmetric:
            j = (i + s) % n
            if j > i:  # Only add each edge once
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    return adj

def _enumerate_circulants(n, deadline, target_alpha=10):
    """
    Enumerate circulant graphs on n vertices and find one that is
    triangle-free with alpha < target_alpha.
    """
    # For circulant graphs, the graph is triangle-free iff
    # there's no triple (a, b, c) in S (symmetric closure) such that
    # a + b = c mod n or a + b + c = 0 mod n (with appropriate conditions).
    
    # Max n is 40 (since R(3,10) <= 41, max valid n is 40)
    # Try from n=40 down to n=20
    
    # For efficiency, only try S subsets of size up to n//2
    # But we need to be smart about which S to try.
    
    # For n even, S can contain n/2 (which creates a perfect matching component)
    # For n odd, S is just pairs {s, n-s}
    
    for n_try in range(40, 19, -1):
        if time.time() > deadline:
            break
        
        # Determine candidate S values
        # S is a subset of {1, 2, ..., n//2} (with n//2 special if even)
        max_s = n // 2
        
        # For small n, we can enumerate all subsets
        # For larger n, use heuristics
        
        # Heuristic: start with "dense" circulants that are likely triangle-free
        # A circulant C(n, S) is triangle-free if S is "sum-free" in a sense
        
        # Try all subsets of {1, 2, ..., min(n//2, 8)} first
        # This limits the search space
        
        search_range = min(n // 2, 12)
        
        # Generate all subsets of {1, ..., search_range}
        # But that's 2^12 = 4096 which is manageable
        # For larger n_try, we might need to be more selective
        
        if n_try <= 30:
            # Can enumerate more
            pass
        
        # Try a curated list of S values that are known to give good Ramsey properties
        # For triangle-free circulants, S should avoid {a, b, a+b mod n} patterns
        
        # Simple approach: try S = {1, 2, ..., k} for various k, and also
        # S = {a, 2a, 3a, ...} for various a
        
        # First, try arithmetic progressions
        for a in range(1, n_try // 2 + 1):
            for length in range(1, n_try // a + 1):
                S = set((a * i) % n_try for i in range(1, length + 1))
                S.discard(0)
                if not S:
                    continue
                
                adj = _circulant_graph(n_try, S)
                
                if _has_triangle_bitset(adj, n_try):
                    continue
                
                if not _max_independent_set_size_bitset(adj, n_try, target_alpha):
                    return {'n': n_try, 'edges': _bitset_to_edges(adj, n_try)}
        
        # Try other structured sets
        # S = {s1, s2, ..., sk} where si are spread out
        for k in range(1, min(n_try // 2, 6) + 1):
            if time.time() > deadline:
                break
            
            # Try random S of size k
            for attempt in range(100):
                S = set(random.sample(range(1, n_try // 2 + 1), min(k, n_try // 2)))
                adj = _circulant_graph(n_try, S)
                
                if _has_triangle_bitset(adj, n_try):
                    continue
                
                if not _max_independent_set_size_bitset(adj, n_try, target_alpha):
                    return {'n': n_try, 'edges': _bitset_to_edges(adj, n_try)}
        
        # Exhaustive enumeration for smaller n
        if n_try <= 20:
            # Enumerate all subsets of {1, ..., n_try//2}
            all_s = list(range(1, n_try // 2 + 1))
            for mask in range(1, 1 << len(all_s)):
                if time.time() > deadline:
                    break
                
                S = set()
                for i, s in enumerate(all_s):
                    if mask & (1 << i):
                        S.add(s)
                
                if not S:
                    continue
                
                adj = _circulant_graph(n_try, S)
                
                if _has_triangle_bitset(adj, n_try):
                    continue
                
                if not _max_independent_set_size_bitset(adj, n_try, target_alpha):
                    return {'n': n_try, 'edges': _bitset_to_edges(adj, n_try)}
        
        # If no circulant works for this n, try local search on random graphs
        if time.time() > deadline - 10:
            break
        
        # Random local search fallback
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            adj = [0] * n_try
            for i in range(n_try):
                for j in range(i + 1, n_try):
                    if random.random() < 0.4:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            _build_triangle_free_bitset(adj, n_try)
            
            if _local_search_bitset(adj, n_try, target_alpha, deadline):
                return {'n': n_try, 'edges': _bitset_to_edges(adj, n_try)}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit
    
    # First, try to find a high-n graph using circulant enumeration
    result = _enumerate_circulants(40, deadline)
    
    if result is not None:
        return result
    
    # Fallback: try local search on random graphs for various n
    for n in range(39, 19, -1):
        if time.time() > deadline:
            break
        
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.4:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            _build_triangle_free_bitset(adj, n)
            
            if _local_search_bitset(adj, n, 10, deadline):
                return {'n': n, 'edges': _bitset_to_edges(adj, n)}
    
    # Final fallback: known valid small graph
    # n=9, empty graph has alpha=9 < 10 and no triangles
    n = 9
    return {'n': n, 'edges': []}

if __name__ == "__main__":
    res = construct()
    print(f"n={res['n']}, edges={len(res['edges'])}")
