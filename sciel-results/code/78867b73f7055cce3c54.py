"""Construct a large (5,5)-free graph via Local Search on fixed N."""

import time
import random
import sys

def _find_k5(adj, n):
    """
    Find a K5 in the graph. Returns the set of 5 vertices or None.
    adj: list of sets of neighbors.
    """
    for v in range(n):
        nbrs_v = list(adj[v])
        m = len(nbrs_v)
        if m < 4:
            continue
        
        for u in nbrs_v:
            common = adj[u] & adj[v]
            common_list = list(common)
            L = len(common_list)
            if L < 3:
                continue
            
            for i in range(L):
                x = common_list[i]
                for j in range(i + 1, L):
                    y = common_list[j]
                    if y in adj[x]:
                        candidates_z = adj[x] & adj[y] & common
                        if candidates_z:
                            z = next(iter(candidates_z))
                            return {v, u, x, y, z}
    return None

def _find_ind5(adj, n, non_adj=None):
    """
    Find an independent set of size 5.
    non_adj: optional precomputed list of non-neighbor sets.
    """
    if non_adj is None:
        non_adj = [set(range(n)) - adj[v] - {v} for v in range(n)]
    
    for v in range(n):
        non_nbrs_v = non_adj[v]
        m = len(non_nbrs_v)
        if m < 4:
            continue
        
        for u in non_nbrs_v:
            non_nbrs_u = non_adj[u]
            common = non_nbrs_v & non_nbrs_u
            common.discard(v)
            common.discard(u)
            
            if len(common) < 3:
                continue
                
            common_list = list(common)
            L = len(common_list)
            
            for i in range(L):
                x = common_list[i]
                for j in range(i + 1, L):
                    y = common_list[j]
                    if y not in adj[x]:
                        # Check for z in common that is non-adjacent to both x and y
                        found_z = False
                        for z in common_list[j+1:]:
                            if z not in adj[x] and z not in adj[y]:
                                found_z = True
                                break
                        if found_z:
                            return {v, u, x, y, z}
    return None

def _local_search(n, max_time=200, initial_adj=None):
    """
    Perform local search to find a (5,5)-free graph on n vertices.
    Returns (n, edges) if found, else None.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    if initial_adj is not None:
        adj = [set(s) for s in initial_adj]
    else:
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i].add(j)
                    adj[j].add(i)
    
    # Precompute non-adjacency for speed
    non_adj = [set(range(n)) - adj[v] - {v} for v in range(n)]
    
    def update_non_adj(i, j):
        """Update non_adj after flipping edge (i,j)."""
        if j in adj[i]:
            # Edge exists, so i and j are adjacent
            non_adj[i].discard(j)
            non_adj[j].discard(i)
        else:
            # No edge, so i and j are non-adjacent
            non_adj[i].add(j)
            non_adj[j].add(i)
    
    # Local search loop
    while time.time() < deadline:
        # Find a violation
        k5 = _find_k5(adj, n)
        if k5:
            # Break a random edge in the K5
            verts = list(k5)
            i, j = random.sample(verts, 2)
            if j in adj[i]:
                adj[i].discard(j)
                adj[j].discard(i)
                update_non_adj(i, j)
            else:
                adj[i].add(j)
                adj[j].add(i)
                update_non_adj(i, j)
            continue
        
        ind5 = _find_ind5(adj, n, non_adj)
        if ind5:
            # Break a random non-edge in the Ind5 (i.e., add an edge)
            verts = list(ind5)
            i, j = random.sample(verts, 2)
            if j in adj[i]:
                adj[i].discard(j)
                adj[j].discard(i)
                update_non_adj(i, j)
            else:
                adj[i].add(j)
                adj[j].add(i)
                update_non_adj(i, j)
            continue
        
        # No violations found
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
        
    return None

def _paley_graph(q):
    """Return adjacency list for the Paley graph on q vertices."""
    residues = {(x * x) % q for x in range(1, q)}
    adj = [set() for _ in range(q)]
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i].add(j)
                adj[j].add(i)
    return adj

def construct():
    deadline = time.time() + 280 # Safety margin
    
    # Strategy: Try local search from random init for n=42, 41, ...
    # Also try extending the Paley graph on 37 vertices
    
    # First, try direct local search for high n values
    for n in [42, 41, 40]:
        if time.time() > deadline:
            break
        
        budget = 80 if n == 42 else (60 if n == 41 else 40)
        
        # Try a few restarts
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            result = _local_search(n, max_time=budget / 3)
            if result:
                return result
            
    # Try extending the Paley graph on 37 vertices
    if time.time() < deadline - 30:
        paley_37 = _paley_graph(37)
        
        for target_n in [42, 41, 40, 39, 38]:
            if time.time() > deadline - 10:
                break
            
            # Start from Paley 37 and add vertices
            # We need to decide edges for new vertices
            budget = 50 if target_n == 42 else (40 if target_n == 41 else 30)
            
            # Initialize with Paley 37 plus random edges for new vertices
            adj = [set() for _ in range(target_n)]
            # Copy Paley 37
            for i in range(37):
                adj[i] = set(paley_37[i])
            
            # Add random edges for new vertices (37 to target_n-1)
            for i in range(37, target_n):
                for j in range(i):
                    if random.random() < 0.5:
                        adj[i].add(j)
                        adj[j].add(i)
            
            result = _local_search(target_n, max_time=budget, initial_adj=adj)
            if result:
                return result
            
    # Fallback: Return the Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    return q, edges

if __name__ == "__main__":
    n, edges = construct()
    print(f"Constructed graph with {n} vertices and {len(edges)} edges.")
