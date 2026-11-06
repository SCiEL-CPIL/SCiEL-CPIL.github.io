"""
(5,5)-free graph construction for R(5,5).
Uses a deterministic local-search extension of the Paley graph on 37 vertices,
with a fallback to the Paley graph if no extension is found in time.
"""
import time
import itertools
import random

DEADLINE = 330.0
START_TIME = time.time()

def _time_left():
    return DEADLINE - time.time()

def _has_k5_or_ind5(adj, n):
    """Check if graph has a K5 or an independent set of size 5."""
    # Check for K5
    for quad in itertools.combinations(range(n), 4):
        # Check all 5-subsets containing this quad
        a, b, c, d = quad
        for e in range(n):
            if e in quad:
                continue
            # Check if {a,b,c,d,e} forms a K5
            if (adj[a][b] and adj[a][c] and adj[a][d] and adj[a][e] and
                adj[b][c] and adj[b][d] and adj[b][e] and
                adj[c][d] and adj[c][e] and
                adj[d][e]):
                return True
    # Check for independent set of size 5
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        for e in range(n):
            if e in quad:
                continue
            # Check if {a,b,c,d,e} forms an independent set
            if not (adj[a][b] or adj[a][c] or adj[a][d] or adj[a][e] or
                    adj[b][c] or adj[b][d] or adj[b][e] or
                    adj[c][d] or adj[c][e] or
                    adj[d][e]):
                return True
    return False

def _check_valid_fast(adj, n, k=5):
    """Faster check using bitmasks for small k."""
    # For k=5, we can be more efficient
    # Precompute neighbor bitmasks
    neighbors = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if i != j and adj[i][j]:
                mask |= (1 << j)
        neighbors[i] = mask
    
    # Check for K5: look for 5-cliques
    # For each pair (i,j) that are adjacent, check if there exist k,l,m such that
    # all form a clique with i,j
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i][j]:
                continue
            # Common neighbors of i and j
            common = neighbors[i] & neighbors[j]
            # Remove i and j from common
            common &= ~(1 << i)
            common &= ~(1 << j)
            # Check if common contains a triangle (3-clique)
            # For each edge (k,l) in common, check if k-l are adjacent and both adjacent to m
            k_list = []
            temp = common
            while temp:
                lowbit = temp & (-temp)
                k = lowbit.bit_length() - 1
                temp ^= lowbit
                k_list.append(k)
            for idx_k in range(len(k_list)):
                for idx_l in range(idx_k+1, len(k_list)):
                    k = k_list[idx_k]
                    l = k_list[idx_l]
                    if adj[k][l]:
                        # Check if there's a vertex m adjacent to i, j, k, l
                        # m must be in common neighbors of i,j,k,l
                        mask_m = neighbors[i] & neighbors[j] & neighbors[k] & neighbors[l]
                        mask_m &= ~(1 << i)
                        mask_m &= ~(1 << j)
                        mask_m &= ~(1 << k)
                        mask_m &= ~(1 << l)
                        if mask_m:
                            return True
    
    # Check for independent set of size 5
    # Equivalent to K5 in the complement
    comp_neighbors = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if i != j and not adj[i][j]:
                mask |= (1 << j)
        comp_neighbors[i] = mask
    
    for i in range(n):
        for j in range(i+1, n):
            if comp_neighbors[i] & (1 << j) == 0:
                continue
            common = comp_neighbors[i] & comp_neighbors[j]
            common &= ~(1 << i)
            common &= ~(1 << j)
            k_list = []
            temp = common
            while temp:
                lowbit = temp & (-temp)
                k = lowbit.bit_length() - 1
                temp ^= lowbit
                k_list.append(k)
            for idx_k in range(len(k_list)):
                for idx_l in range(idx_k+1, len(k_list)):
                    k = k_list[idx_k]
                    l = k_list[idx_l]
                    if comp_neighbors[k] & (1 << l):
                        mask_m = comp_neighbors[i] & comp_neighbors[j] & comp_neighbors[k] & comp_neighbors[l]
                        mask_m &= ~(1 << i)
                        mask_m &= ~(1 << j)
                        mask_m &= ~(1 << k)
                        mask_m &= ~(1 << l)
                        if mask_m:
                            return True
    
    return False

def _build_paley(q):
    """Build Paley graph on q vertices (q must be prime ≡ 1 mod 4)."""
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    adj = [[False] * n for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (j - i) % q in residues:
                adj[i][j] = True
                adj[j][i] = True
                edges.append((i, j))
    return n, adj, edges

def construct():
    """
    Return (n, edges) for a (5,5)-free graph on n vertices.
    Uses Paley graph on 37 as seed, then attempts local search extension.
    """
    if time.time() > DEADLINE - 30:
        # Not enough time, return Paley graph
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, edges
    
    # Start with Paley graph on 37 vertices
    n, adj, edges = _build_paley(37)
    
    # Verify base graph is valid
    if _check_valid_fast(adj, n):
        # Should not happen, but be safe
        pass
    else:
        return n, edges
    
    best_n = n
    best_edges = list(edges)
    best_adj = [row[:] for row in adj]
    
    # Attempt to extend by adding vertices
    # We'll try to add vertices one at a time, searching for a valid connection pattern
    random.seed(42)
    
    # For each new vertex, we need to decide adjacency to all existing vertices
    # such that no K5 or independent 5-set is created
    # This is a constraint satisfaction problem
    
    # Greedy approach: for each new vertex, try random connection patterns
    # and keep the first one that maintains validity
    
    target_n = 42
    attempts_per_vertex = 200
    
    current_n = n
    current_adj = [row[:] for row in adj]
    current_edges = list(edges)
    
    for new_v in range(target_n - current_n):
        if _time_left() < 30:
            break
        
        found = False
        for attempt in range(attempts_per_vertex):
            if _time_left() < 15:
                break
            
            # Random connection pattern
            new_connections = [random.random() < 0.5 for _ in range(current_n)]
            
            # Create new adjacency
            test_n = current_n + 1
            test_adj = [row[:] + [False] for row in current_adj]
            test_adj.append([False] * test_n)
            
            for i in range(current_n):
                test_adj[i][test_n - 1] = new_connections[i]
                test_adj[test_n - 1][i] = new_connections[i]
            
            # Check validity - this is expensive, so use a faster check
            # For efficiency, we can check only 5-sets involving the new vertex
            valid = True
            new_idx = test_n - 1
            
            # Check for K5 involving new vertex
            for quad in itertools.combinations(range(current_n), 4):
                a, b, c, d = quad
                # Check if new vertex is adjacent to all of a,b,c,d
                if all(new_connections[x] for x in quad):
                    # Check if a,b,c,d form a K4
                    if (test_adj[a][b] and test_adj[a][c] and test_adj[a][d] and
                        test_adj[b][c] and test_adj[b][d] and
                        test_adj[c][d]):
                        valid = False
                        break
            
            if valid:
                # Check for independent set of size 5 involving new vertex
                for quad in itertools.combinations(range(current_n), 4):
                    a, b, c, d = quad
                    # Check if new vertex is NOT adjacent to any of a,b,c,d
                    if not any(new_connections[x] for x in quad):
                        # Check if a,b,c,d form an independent set (no edges)
                        if not (test_adj[a][b] or test_adj[a][c] or test_adj[a][d] or
                                test_adj[b][c] or test_adj[b][d] or
                                test_adj[c][d]):
                            valid = False
                            break
            
            if valid:
                found = True
                current_n = test_n
                current_adj = test_adj
                current_edges = current_edges + [(i, new_idx) for i in range(current_n - 1) if new_connections[i]]
                break
        
        if not found:
            # Try with different random seeds
            random.seed(attempt * 1000 + 7)
            for attempt2 in range(attempts_per_vertex):
                if _time_left() < 15:
                    break
                new_connections = [random.random() < 0.5 for _ in range(current_n)]
                new_idx = current_n
                test_adj = [row[:] + [False] for row in current_adj]
                test_adj.append([False] * (current_n + 1))
                
                for i in range(current_n):
                    test_adj[i][new_idx] = new_connections[i]
                    test_adj[new_idx][i] = new_connections[i]
                
                valid = True
                for quad in itertools.combinations(range(current_n), 4):
                    if all(new_connections[x] for x in quad):
                        if (test_adj[quad[0]][quad[1]] and test_adj[quad[0]][quad[2]] and test_adj[quad[0]][quad[3]] and
                            test_adj[quad[1]][quad[2]] and test_adj[quad[1]][quad[3]] and
                            test_adj[quad[2]][quad[3]]):
                            valid = False
                            break
                
                if valid:
                    for quad in itertools.combinations(range(current_n), 4):
                        if not any(new_connections[x] for x in quad):
                            if not (test_adj[quad[0]][quad[1]] or test_adj[quad[0]][quad[2]] or test_adj[quad[0]][quad[3]] or
                                    test_adj[quad[1]][quad[2]] or test_adj[quad[1]][quad[3]] or
                                    test_adj[quad[2]][quad[3]]):
                                valid = False
                                break
                
                if valid:
                    found = True
                    current_n = current_n + 1
                    current_adj = test_adj
                    current_edges = current_edges + [(i, new_idx) for i in range(current_n - 1) if new_connections[i]]
                    break
        
        if not found:
            break
    
    # Final verification
    if _check_valid_fast(current_adj, current_n):
        # Invalid, return best valid
        return best_n, best_edges
    else:
        return current_n, current_edges
