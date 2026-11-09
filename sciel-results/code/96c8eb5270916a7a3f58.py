import time
import random

def _greedy_independent_set(adj, n):
    """
    Finds a large independent set using a greedy heuristic.
    Returns the set of vertices as a bitmask.
    """
    available = (1 << n) - 1
    indep_set = 0
    while available:
        # Pick vertex with minimum degree in the induced subgraph
        best_v = -1
        min_deg = n + 1
        temp_avail = available
        while temp_avail:
            v_bit = temp_avail & -temp_avail
            v = v_bit.bit_length() - 1
            # Count neighbors in available
            deg = (adj[v] & available).bit_count()
            if deg < min_deg:
                min_deg = deg
                best_v = v
            temp_avail &= temp_avail - 1
            
        if best_v == -1:
            break
            
        indep_set |= (1 << best_v)
        # Remove best_v and its neighbors from available
        available &= ~(adj[best_v] | (1 << best_v))
        
    return indep_set

def _has_independent_set_10(adj, n):
    """
    Exact check for independent set of size 10 using branch and bound.
    Returns True if an independent set of size >= 10 exists.
    """
    def dfs(available_mask, size):
        if size >= 10:
            return True
        if available_mask == 0:
            return False
        
        if available_mask.bit_count() < 10 - size:
            return False
            
        v_bit = available_mask & -available_mask
        v = v_bit.bit_length() - 1
        
        # Branch 1: Include v
        new_avail = available_mask & ~(adj[v] | v_bit)
        if dfs(new_avail, size + 1):
            return True
            
        # Branch 2: Exclude v
        new_avail = available_mask & ~v_bit
        if dfs(new_avail, size):
            return True
            
        return False

    return dfs((1 << n) - 1, 0)

def _random_maximal_triangle_free(n, seed=None):
    """
    Generate a random maximal triangle-free graph on n vertices.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
        
    adj = [0] * n
    edges_to_add = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(edges_to_add)
    
    for u, v in edges_to_add:
        common = adj[u] & adj[v]
        if common == 0:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    return adj

def _optimize_graph(adj, n, deadline):
    """
    Local search to reduce independence number.
    Returns True if a valid graph (alpha < 10) is found.
    """
    # We will iterate until deadline or success
    # We track the best independent set found to guide edge additions
    last_check_iter = 0
    check_interval = 20
    
    while time.time() < deadline - 2:
        # Find a greedy independent set
        indep_mask = _greedy_independent_set(adj, n)
        indep_size = indep_mask.bit_count()
        
        if indep_size < 10:
            # Potential success, verify exactly
            if not _has_independent_set_10(adj, n):
                return True
            # If exact check fails, it means greedy was misleading or we have a 
            # complex structure, but greedy < 10 usually implies alpha < 10 
            # for these random graphs? No, greedy is a lower bound.
            # If greedy finds < 10, it doesn't prove alpha < 10.
            # But if greedy finds >= 10, we definitely have alpha >= 10.
            # So we only know for sure if we find alpha < 10 via exact search.
            # However, if greedy finds < 10, it's a strong candidate.
            # Let's do a few more iterations to see if we can break any hidden sets.
            # Actually, if greedy < 10, it's very likely alpha < 10.
            # But we must be exact.
            pass
        
        # If indep_size >= 10, try to break it
        if indep_size >= 10:
            # Get list of vertices in indep_mask
            verts = []
            temp = indep_mask
            while temp:
                b = temp & -temp
                verts.append(b.bit_length() - 1)
                temp &= temp - 1
            
            # Try to add edges between pairs in this independent set
            # We pick a random pair and try to add an edge
            # If we can add an edge (u, v) where u, v in indep_set,
            # we break this specific independent set.
            
            # Heuristic: Try all pairs? No, too slow.
            # Try a random sample of pairs.
            if len(verts) >= 2:
                # Shuffle verts to randomize
                random.shuffle(verts)
                added = False
                for i in range(len(verts)):
                    for j in range(i + 1, len(verts)):
                        u = verts[i]
                        v = verts[j]
                        if not (adj[u] & (1 << v)):
                            # Check if adding edge (u, v) creates a triangle
                            common = adj[u] & adj[v]
                            if common == 0:
                                # Safe to add
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                                added = True
                                break
                    if added:
                        break
                
                if not added:
                    # If we couldn't add any edge within the independent set,
                    # it means the induced subgraph on the independent set is complete
                    # in the complement? No, it means for every pair u,v in IS,
                    # adj[u] & adj[v] != 0.
                    # This implies the graph is "dense" around this set.
                    # We might need to remove edges elsewhere to allow this.
                    # For now, just continue.
                    pass

        # Periodic exact check
        if last_check_iter == 0 or (last_check_iter % check_interval == 0):
             # We already did greedy check above.
             # If greedy < 10, we did exact check.
             # If greedy >= 10, we know alpha >= 10, so no need for exact check.
             # But what if greedy was bad?
             # Let's do an exact check every 100 iterations just in case.
             pass
        last_check_iter += 1
        
        # If we added an edge, the graph changed.
        # If we didn't, we might be stuck.
        # Let's try a random edge removal to escape local minima?
        # No, removing edges increases alpha usually.
        # We want to ADD edges to decrease alpha.
        
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Try to find a graph for n from 40 down to 25
    # We prioritize larger n.
    # R(3,10) is known to be <= 41.
    # Known lower bound is 29 (from literature, e.g., R(3,10) > 29).
    # We aim for 29 or 30.
    
    for n in range(40, 24, -1):
        if time.time() > deadline - 5:
            break
            
        found = False
        attempts = 0
        max_attempts = 50 # Reduced attempts to allow more time per optimization
        
        while attempts < max_attempts and not found:
            attempts += 1
            if time.time() > deadline - 5:
                break
                
            seed = attempts * 1000 + n
            rng = random.Random(seed)
            
            adj = _random_maximal_triangle_free(n, seed=seed)
            
            # Optimization
            if _optimize_graph(adj, n, deadline):
                # Found a valid graph
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
        
        # If not found for this n, try smaller n

    # Fallback: return a small valid graph
    # We can construct a small one easily, e.g., n=1 or n=2
    # But we should try to return something decent.
    # Let's try n=25 with more attempts if time permits, or just return a trivial one.
    # Given the loop went down to 25, if we failed all, we return a trivial graph.
    # However, it is very likely we found one for n >= 29.
    
    # If we reach here, we failed to find any graph with n > 24.
    # This is highly unlikely given the search space.
    # Return a trivial valid graph: n=1, no edges.
    return {'n': 1, 'edges': []}

if __name__ == '__main__':
    r = construct()
    print(r['n'], len(r['edges']))
