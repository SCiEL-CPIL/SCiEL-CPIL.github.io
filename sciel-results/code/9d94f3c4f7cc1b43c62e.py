import random
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a triangle-free graph with no independent set of size 10.
    Aims for n=40 (theoretical max).
    """
    deadline = time.time() + 110.0  # Safety margin for 120s limit
    
    # Start with the known valid graph from the seed as a fallback
    # Andrasfai graph And(9): Z_26
    n_seed = 26
    edges_seed = [[i, j] for i in range(n_seed) for j in range(i + 1, n_seed) if (j - i) % n_seed % 3 == 1]
    best_result = {'n': n_seed, 'edges': edges_seed}
    best_n = n_seed

    # Try to find a graph on n=40, then n=39, etc.
    # We allocate a portion of the time for each attempt.
    # Since n=40 is the target, we give it the most time.
    
    # Time allocation strategy:
    # Try 40 for 80s.
    # If fail, try 39 for 20s.
    # If fail, return best (which is 26).
    
    time_for_40 = 80.0
    time_for_39 = 20.0
    
    # Attempt n=40
    if time.time() < deadline:
        end_time_40 = min(deadline, time.time() + time_for_40)
        result = _local_search(40, end_time_40)
        if result is not None:
            return result # Found max n, return immediately

    # Attempt n=39
    if time.time() < deadline:
        end_time_39 = min(deadline, time.time() + time_for_39)
        result = _local_search(39, end_time_39)
        if result is not None:
            return result

    # Fallback to seed
    return best_result

def _count_triangles(n, edges):
    """Counts triangles in the graph."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    count = 0
    for u in range(n):
        for v in adj[u]:
            if u < v:
                count += len(adj[u] & adj[v])
    return count

def _has_independent_set_10(n, edges):
    """
    Checks if there is an independent set of size 10.
    Uses backtracking with bitsets.
    Returns True if found, False otherwise.
    """
    # Build complement adjacency bitsets
    # 1 if edge exists in G, 0 otherwise.
    # We want Independent Set in G -> Clique in G_bar.
    # G_bar edge (i,j) exists if G edge (i,j) does NOT exist.
    
    adj = [[0]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = 1
        adj[v][u] = 1
    
    comp_adj = [0] * n
    all_mask = (1 << n) - 1
    for i in range(n):
        mask = 0
        for j in range(n):
            if i != j and adj[i][j] == 0:
                mask |= (1 << j)
        comp_adj[i] = mask & ~(1 << i)

    # Backtracking to find clique of size 10 in comp_adj
    def find_clique(candidates, size):
        if size == 0:
            return True
        if candidates == 0:
            return False
        
        # Count bits
        # Python 3.10+ has int.bit_count()
        if candidates.bit_count() < size:
            return False
            
        # Get lowest set bit
        lsb = candidates & -candidates
        v = lsb.bit_length() - 1
        
        # Option 1: Exclude v
        # New candidates: remove v
        new_cand_excl = candidates & ~lsb
        
        if find_clique(new_cand_excl, size):
            return True
            
        # Option 2: Include v
        # New candidates: must be neighbors of v in comp_adj, and > v (to avoid duplicates/symmetry)
        # Actually, if we just intersect with neighbors, we might pick vertices < v.
        # To ensure we don't find the same clique, we restrict to vertices > v.
        higher_mask = all_mask & ~((1 << (v + 1)) - 1)
        new_cand_incl = (candidates & comp_adj[v] & higher_mask)
        
        if find_clique(new_cand_incl, size - 1):
            return True
            
        return False

    return find_clique(all_mask, 10)

def _local_search(n, deadline):
    """
    Runs randomized local search (Simulated Annealing) to find a valid graph on n vertices.
    Returns dict if found, None otherwise.
    """
    # Initial random graph
    edges_set = set()
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                edges_set.add((i, j))
    
    current_edges = list(edges_set)
    
    # Helper to calculate cost
    def get_cost(edge_list):
        t_count = _count_triangles(n, edge_list)
        if t_count > 0:
            # Prioritize removing triangles
            return t_count * 100
        
        if _has_independent_set_10(n, edge_list):
            # High penalty for independent set
            return 1000 + t_count
        
        return 0 # Valid!
        
    # If initial is valid, return
    if get_cost(current_edges) == 0:
        return {'n': n, 'edges': [list(e) for e in current_edges]}
    
    T = 1.0
    cooling_rate = 0.9995
    current_cost = get_cost(current_edges)
    
    while time.time() < deadline:
        # Pick a random edge to flip
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
            
        if (i, j) in edges_set:
            # Remove edge
            edges_set.remove((i, j))
            new_edges = list(edges_set)
        else:
            # Add edge
            edges_set.add((i, j))
            new_edges = list(edges_set)
            
        new_cost = get_cost(new_edges)
        
        if new_cost < current_cost:
            # Accept
            current_edges = new_edges
            current_cost = new_cost
            if current_cost == 0:
                return {'n': n, 'edges': [list(e) for e in current_edges]}
        else:
            # Accept with probability
            delta = new_cost - current_cost
            if T > 0.01:
                # delta is positive
                prob = pow(2.718, -delta / T)
                if random.random() < prob:
                    current_edges = new_edges
                    current_cost = new_cost
                    if current_cost == 0:
                        return {'n': n, 'edges': [list(e) for e in current_edges]}
            else:
                # If T is very low, we only accept improvements.
                # Since we didn't accept (new_cost >= current_cost), we revert.
                # But we already modified edges_set. We need to revert it.
                if (i, j) in edges_set:
                    edges_set.remove((i, j))
                else:
                    edges_set.add((i, j))
                # Note: In the "Accept" branch above, we didn't revert, so edges_set is updated.
                # In the "Reject" branch, we must revert edges_set to match current_edges.
                # Wait, if we reject, edges_set should be restored to its previous state.
                # The previous state was before the flip.
                # So if we added, remove. If we removed, add.
                # This logic is correct.
                
        T *= cooling_rate

    # If we exit loop, check if current is valid (just in case)
    if current_cost == 0:
        return {'n': n, 'edges': [list(e) for e in current_edges]}
        
    return None

# EVOLVE-BLOCK-END
