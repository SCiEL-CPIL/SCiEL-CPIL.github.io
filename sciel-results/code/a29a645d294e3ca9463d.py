"""
Triangle-free graphs without independent 10-sets (Ramsey number R(3,10)).

Strategy:
1. Start with the Andrasfai graph And(9) on 26 vertices (triangle-free, alpha=9).
2. Try to extend to n=27, 28, ..., up to 40 by adding vertices one at a time.
   For each new vertex, choose its neighborhood (subset of existing vertices)
   such that:
     - No triangle is created.
     - No independent set of size 10 is created.
   Use randomized local search / greedy heuristics to find valid neighborhoods.
3. Use a deadline guard to return the best result found before timeout.

Key insight: When adding vertex v to a triangle-free graph G, v's neighborhood N(v)
must be an independent set in G (otherwise a triangle forms). Also, for any
independent set I in G, if I ∩ N(v) = ∅ then I ∪ {v} is independent in G∪{v}.
So we need: for every independent set I of G with |I| = 9, we must have I ∩ N(v) ≠ ∅.
Equivalently, N(v) must hit all maximum independent sets of G.

This is a hitting set problem: find an independent set S in G that hits all
maximum independent sets of G. Then adding a vertex connected exactly to S
preserves triangle-freeness and keeps alpha ≤ 9.

We use randomized greedy to find such S.
"""

import time
import random
import sys

DEADLINE = 145.0  # safety margin before 150s kill

def _elapsed():
    return time.time() - _START

_START = time.time()

def _is_triangle_free(adj, n):
    """Check if graph with adjacency matrix adj (list of bytearrays) is triangle-free."""
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                # Check if any common neighbor k > j to avoid double counting
                # Actually just check all k
                for k in range(j + 1, n):
                    if adj[i][k] and adj[j][k]:
                        return False
    return True

def _max_independent_set_size(adj, n):
    """
    Compute the size of the maximum independent set using branch and bound.
    adj: list of bytearrays, n: number of vertices.
    Returns the size.
    """
    # Use a simple but effective branch and bound for alpha
    # Independent set = clique in complement. Let's work with complement for speed.
    # Actually, let's do branch and bound on the original graph for independent set.
    
    # Represent adjacency as bitmasks for speed
    masks = [0] * n
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                masks[i] |= (1 << j)
    
    best = [0]
    
    def bnb(candidates, current_size):
        if _elapsed() > DEADLINE - 5:
            return
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
        # Upper bound: current_size + len(candidates)
        if current_size + len(candidates) <= best[0]:
            return
        
        # Choose vertex with highest degree in candidate subgraph for branching
        # Actually for independent set, we want to branch on a vertex.
        # Pick a vertex v from candidates.
        v = candidates[0]
        v_bit = 1 << v
        
        # Branch 1: include v in independent set
        # Remove v and all its neighbors from candidates
        neighbors_mask = masks[v]
        new_candidates = [u for u in candidates if u != v and not (neighbors_mask & (1 << u))]
        bnb(new_candidates, current_size + 1)
        
        # Branch 2: exclude v
        new_candidates = candidates[1:]
        bnb(new_candidates, current_size)
    
    bnb(list(range(n)), 0)
    return best[0]

def _max_independent_sets(adj, n, limit=1000):
    """
    Find up to 'limit' maximum independent sets.
    Returns list of frozensets.
    """
    masks = [0] * n
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                masks[i] |= (1 << j)
    
    alpha = _max_independent_set_size(adj, n)
    
    found = set()
    
    def find_one(candidates, current, count):
        if _elapsed() > DEADLINE - 5:
            return
        if len(found) >= limit:
            return
        if not candidates:
            if count == alpha:
                found.add(frozenset(current))
            return
        if count + len(candidates) < alpha:
            return
        
        v = candidates[0]
        neighbors_mask = masks[v]
        
        # Include v
        new_candidates = [u for u in candidates if u != v and not (neighbors_mask & (1 << u))]
        find_one(new_candidates, current | {v}, count + 1)
        
        # Exclude v
        find_one(candidates[1:], current, count)
    
    find_one(list(range(n)), frozenset(), 0)
    return list(found)

def _andrasfai_9():
    """Construct Andrasfai graph And(9) on 26 vertices."""
    n = 26
    adj = [bytearray(n) for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n % 3 == 1:
                adj[i][j] = 1
                adj[j][i] = 1
                edges.append([i, j])
    return n, adj, edges

def _try_extend(n, adj, edges, target_n):
    """
    Try to extend the graph from n vertices to target_n vertices.
    For each new vertex, find a neighborhood S (independent set in current graph)
    that hits all maximum independent sets.
    Returns (new_n, new_adj, new_edges) or None if failed.
    """
    cur_n = n
    cur_adj = [row[:] for row in adj]
    cur_edges = list(edges)
    
    for new_v in range(cur_n, target_n):
        # Find all maximum independent sets of current graph
        mis = _max_independent_sets(cur_adj, cur_n)
        if _elapsed() > DEADLINE - 5:
            return None
        
        if not mis:
            return None
        
        # We need an independent set S in G such that S intersects every MIS.
        # S must be independent: for all u,v in S, not adj[u][v]
        # S must hit every mis in mis.
        
        # Greedy randomized approach:
        # Try many random independent sets and check if they hit all MIS.
        
        # First, let's get the set of all vertices that appear in some MIS.
        # Actually, let's just try random independent sets.
        
        # Build list of all independent sets of small size? No, too many.
        # Instead: randomly sample independent sets.
        
        # A better approach: 
        # The set S must be an independent set that is a hitting set for the family of MIS.
        # Let's try: for each vertex v, check if {v} hits all MIS. If yes, S={v} works.
        # More generally, try small independent sets.
        
        # Let's collect all MIS as sets of vertices.
        mis_list = [set(m) for m in mis]
        
        # Check single vertices
        for v in range(cur_n):
            if all(v in m for m in mis_list):
                # S = {v} is independent (trivially) and hits all MIS.
                # Add new vertex connected to v.
                for u in range(cur_n):
                    cur_adj[new_v] = cur_adj[new_v]  # already zero
                # Set adjacency
                cur_adj = [row[:] for row in cur_adj] + [bytearray(cur_n + 1)]
                cur_adj[new_v] = bytearray(cur_n + 1)
                for u in range(cur_n):
                    cur_adj[u][new_v] = 0
                    cur_adj[new_v][u] = 0
                cur_adj[v][new_v] = 1
                cur_adj[new_v][v] = 1
                cur_edges.append([v, new_v])
                cur_n = new_v + 1
                break
            else:
                continue
        else:
            # No single vertex works. Try pairs.
            found = False
            for _ in range(5000):  # random attempts
                if _elapsed() > DEADLINE - 5:
                    return None
                # Pick two non-adjacent vertices
                u = random.randint(0, cur_n - 1)
                v = random.randint(0, cur_n - 1)
                if u == v or cur_adj[u][v]:
                    continue
                # Check if {u, v} hits all MIS
                if all(u in m or v in m for m in mis_list):
                    # Add new vertex connected to u and v
                    cur_adj = [row[:] for row in cur_adj] + [bytearray(cur_n + 1)]
                    cur_adj[new_v] = bytearray(cur_n + 1)
                    for x in range(cur_n):
                        cur_adj[x][new_v] = 0
                        cur_adj[new_v][x] = 0
                    cur_adj[u][new_v] = 1
                    cur_adj[new_v][u] = 1
                    cur_adj[v][new_v] = 1
                    cur_adj[new_v][v] = 1
                    cur_edges.append([u, new_v])
                    cur_edges.append([v, new_v])
                    cur_n = new_v + 1
                    found = True
                    break
            if not found:
                # Try triples
                found = False
                for _ in range(5000):
                    if _elapsed() > DEADLINE - 5:
                        return None
                    u = random.randint(0, cur_n - 1)
                    v = random.randint(0, cur_n - 1)
                    w = random.randint(0, cur_n - 1)
                    if u == v or v == w or u == w:
                        continue
                    if cur_adj[u][v] or cur_adj[v][w] or cur_adj[u][w]:
                        continue
                    if all(u in m or v in m or w in m for m in mis_list):
                        cur_adj = [row[:] for row in cur_adj] + [bytearray(cur_n + 1)]
                        cur_adj[new_v] = bytearray(cur_n + 1)
                        for x in range(cur_n):
                            cur_adj[x][new_v] = 0
                            cur_adj[new_v][x] = 0
                        cur_adj[u][new_v] = 1
                        cur_adj[new_v][u] = 1
                        cur_adj[v][new_v] = 1
                        cur_adj[new_v][v] = 1
                        cur_adj[w][new_v] = 1
                        cur_adj[new_v][w] = 1
                        cur_edges.append([u, new_v])
                        cur_edges.append([v, new_v])
                        cur_edges.append([w, new_v])
                        cur_n = new_v + 1
                        found = True
                        break
                if not found:
                    return None
    
    return cur_n, cur_adj, cur_edges

def construct():
    global _START
    _START = time.time()
    
    # Start with And(9)
    n, adj, edges = _andrasfai_9()
    best_n = n
    best_edges = list(edges)
    
    # Try to extend
    for target in range(n + 1, 41):
        if _elapsed() > DEADLINE - 10:
            break
        result = _try_extend(n, adj, edges, target)
        if result is not None:
            n, adj, edges = result
            best_n = n
            best_edges = list(edges)
        else:
            break
    
    return {'n': best_n, 'edges': best_edges}
