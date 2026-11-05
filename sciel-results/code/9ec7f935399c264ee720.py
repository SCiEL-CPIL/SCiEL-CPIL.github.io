import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a C4-free graph on n vertices with min degree >= n - 39.
    Maximize n.
    """
    deadline = time.time() + 145  # Leave 5s safety margin

    # We know n=47 is possible (R(C4, K1,39) = 47).
    # Try 47 first.
    for n in [47, 46, 45, 44, 43, 42, 41, 40, 39]:
        if time.time() > deadline:
            break

        required_min_deg = n - 39

        # Try to construct for this n
        result = _robust_construct(n, required_min_deg, deadline)

        if result is not None:
            n_found, edges_found = result
            return {'n': n_found, 'edges': edges_found}

    # Fallback: return a valid small graph
    # n=39, min_deg=0 is always valid (empty graph)
    return {'n': 39, 'edges': []}

def _check_c4_add_sets(adj_sets, u, v):
    """
    Check if adding edge (u,v) creates a C4, using Python sets for adjacency.
    adj_sets[i] is a set of neighbors of vertex i.
    Returns True if a C4 is created, False otherwise.
    """
    # For C4: u-x-y-v-u where x in N(u)\{v}, y in N(v)\{u}, and (x,y) in E
    # Equivalent: exists x in N(u)\{v} such that N(x) intersects N(v)\{u} (excluding y==x case, but adj[x] doesn't contain x)
    nu = adj_sets[u] - {v}
    nv = adj_sets[v] - {u}
    
    for x in nu:
        # Check if N(x) has any common element with N(v)\{u}
        # i.e., if adj_sets[x] & nv is non-empty
        if adj_sets[x] & nv:
            return True
    return False

def _greedy_fill_sets(adj_sets, n, target, deadline):
    """
    Greedily add edges to increase min degree using set-based adjacency.
    Returns True if target min degree is reached.
    """
    while time.time() < deadline:
        degrees = [len(s) for s in adj_sets]
        min_deg = min(degrees)
        
        if min_deg >= target:
            return True
            
        # Find vertex with minimum degree
        min_vertices = [i for i in range(n) if degrees[i] == min_deg]
        # Pick one randomly to break symmetry
        u = random.choice(min_vertices)
        
        # Get non-neighbors of u
        all_vertices = set(range(n))
        non_neighbors = list(all_vertices - adj_sets[u] - {u})
        
        if len(non_neighbors) == 0:
            return False
            
        random.shuffle(non_neighbors)
        
        added = False
        for v in non_neighbors:
            if not _check_c4_add_sets(adj_sets, u, v):
                adj_sets[u].add(v)
                adj_sets[v].add(u)
                added = True
                break  # Restart loop to re-evaluate degrees
        
        if not added:
            # Stuck in greedy phase
            return False
            
    return False

def _local_search_swaps_sets(adj_sets, n, target, deadline):
    """
    Local search using edge swaps to increase min degree, using set-based adjacency.
    Assumes adj_sets is C4-free.
    Returns True if target min degree is reached.
    """
    for _ in range(2000):  # Max iterations per restart
        if time.time() > deadline:
            return False
            
        degrees = [len(s) for s in adj_sets]
        min_deg = min(degrees)
        
        if min_deg >= target:
            return True
            
        # Find vertex with minimum degree
        min_vertices = [i for i in range(n) if degrees[i] == min_deg]
        u = random.choice(min_vertices)
        
        # Get non-neighbors of u
        all_vertices = set(range(n))
        non_neighbors = list(all_vertices - adj_sets[u] - {u})
        
        if len(non_neighbors) == 0:
            return False
            
        random.shuffle(non_neighbors)
        
        improved = False
        for v in non_neighbors:
            if _check_c4_add_sets(adj_sets, u, v):
                # Try to find a swap: remove (x,y) where x in N(u)\{v}, y in N(v)\{u}, (x,y) in E
                # and such that after removing (x,y), adding (u,v) is safe
                nu = adj_sets[u] - {v}
                nv = adj_sets[v] - {u}
                
                # Find all blocking edges (x,y) where x in nu, y in nv, (x,y) in E
                blockers = []
                for x in nu:
                    common = adj_sets[x] & nv
                    for y in common:
                        blockers.append((x, y))
                
                for x, y in blockers:
                    # Tentatively remove (x,y) and add (u,v)
                    adj_sets[x].discard(y)
                    adj_sets[y].discard(x)
                    adj_sets[u].add(v)
                    adj_sets[v].add(u)
                    
                    # Check if the graph is still C4-free
                    # The only new C4s must involve (u,v)
                    # So check if adding (u,v) to graph-without-(x,y) creates a C4
                    # We need to check: does there exist x' in N(u)\{v} and y' in N(v)\{u} with (x',y') in E?
                    # Note: (x,y) is no longer in E, so it won't block.
                    nu_new = adj_sets[u] - {v}
                    nv_new = adj_sets[v] - {u}
                    
                    safe = True
                    for x_prime in nu_new:
                        if adj_sets[x_prime] & nv_new:
                            safe = False
                            break
                    
                    if safe:
                        # Keep the swap
                        improved = True
                        break
                    else:
                        # Revert
                        adj_sets[u].discard(v)
                        adj_sets[v].discard(u)
                        adj_sets[x].add(y)
                        adj_sets[y].add(x)
                        continue
                        
                if improved:
                    break
            else:
                # No C4, just add
                adj_sets[u].add(v)
                adj_sets[v].add(u)
                improved = True
                break
                
        if not improved:
            break
            
    return False

def _robust_construct(n, target, deadline):
    """
    Combines greedy and local search with multiple restarts using set-based adjacency.
    """
    # Try several random greedy starts
    for attempt in range(100):
        if time.time() > deadline:
            break
            
        adj_sets = [set() for _ in range(n)]
        
        # 1. Greedy Phase
        if _greedy_fill_sets(adj_sets, n, target, deadline):
            return (n, _to_edges_sets(adj_sets, n))
            
        # 2. Local Search Phase
        if _local_search_swaps_sets(adj_sets, n, target, deadline):
            return (n, _to_edges_sets(adj_sets, n))
            
    return None

def _to_edges_sets(adj_sets, n):
    edges = []
    for u in range(n):
        for v in adj_sets[u]:
            if u < v:
                edges.append([u, v])
    return edges

# EVOLVE-BLOCK-END
