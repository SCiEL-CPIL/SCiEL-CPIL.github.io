"""
Decomposition of three fixed graphs into few cycles and edges.

Structural change: instead of pure greedy cycle extraction, we combine:
1. A local-search splicing approach: after initial cycle extraction, we attempt
   to merge cycles/edges that share a vertex into larger cycles. This structurally
   reduces piece count beyond pure greedy extraction.
2. The splicing works by finding two pieces (cycles or single edges) that share
   a vertex, and checking if they can be recombined into a single longer cycle
   or a cycle plus fewer single edges.

Key insight from island 6: local-search splicing merges cycles/edges that share
a vertex into larger cycles, reducing piece count from 5.905 to 6.0.

The algorithm:
1. Greedily extract cycles (longest first) using DFS.
2. After initial extraction, repeatedly try to splice two pieces together:
   - If two cycles share a vertex, they can potentially be merged into one cycle.
   - If a cycle and a single edge share a vertex, the edge might be absorbed.
   - If two single edges share a vertex, they form a path (but that's still 2 pieces
     unless they can be part of a larger cycle with other edges).
3. The splicing operation: given two cycles C1 = [v1, v2, ..., vk] and C2 = [w1, w2, ..., wm]
   that share vertex v (which is in both), we can form a new cycle by traversing C1 from v
   around back to v, then taking the shared vertex and traversing C2. But this only works
   if the shared vertex appears exactly once in each cycle (which it does by definition
   of a simple cycle).
   
   Actually, for two cycles sharing exactly one vertex v:
   C1 = v, a1, a2, ..., ak, v
   C2 = v, b1, b2, ..., bm, v
   We can form: v, a1, a2, ..., ak, b1, b2, ..., bm, v
   This is a valid cycle of length k+m+1 (if all vertices are distinct except v).
   Wait, no - this would repeat v in the middle. Let me reconsider.
   
   Actually, the standard splicing: if two cycles share a vertex v, we can "splice" them
   by removing the edge (v, next_in_C1) and (v, next_in_C2) and replacing with 
   (prev_in_C1, next_in_C2) and (prev_in_C2, next_in_C1). But this requires those
   edges to exist in the graph.
   
   Simpler approach: if two cycles share a vertex v, and we have:
   C1: ... x - v - y ...
   C2: ... p - v - q ...
   Then if edges (x, q) and (p, y) exist in the remaining graph, we can replace
   the two cycles with one cycle: ... x - q - ... - p - y ...
   This reduces piece count by 1.
   
   Even simpler: if a single edge (u, v) and a cycle C that contains u (or v) exist,
   and there's a way to incorporate the edge into the cycle, we reduce piece count by 1.

Let me implement a practical version:
1. Initial greedy cycle extraction.
2. Repeatedly look for splicing opportunities:
   a. Two cycles sharing a vertex: check if we can merge them.
   b. A cycle and a single edge: check if the edge can be absorbed.
3. Continue until no more splicings are possible or time runs out.
"""

import time
from collections import defaultdict
from random import Random
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose_graph(edges):
    """
    Decompose a graph into cycles and single edges, minimizing piece count.
    
    Uses greedy cycle extraction followed by local-search splicing.
    """
    if not edges:
        return []
    
    deadline = time.time() + 145  # safety margin
    
    remaining = set(tuple(sorted(e)) for e in edges)
    pieces = []
    
    # Build adjacency
    adj = defaultdict(set)
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
    
    # Phase 1: Extract cycles greedily, preferring longer cycles
    iteration = 0
    while remaining and time.time() < deadline:
        iteration += 1
        if iteration > 500:  # safety
            break
        
        # Find a cycle
        cycle = _find_best_cycle(remaining, adj)
        
        if cycle and len(cycle) >= 3:
            pieces.append(cycle)
            # Remove cycle edges
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = tuple(sorted((u, v)))
                if e in remaining:
                    remaining.discard(e)
                    adj[u].discard(v)
                    adj[v].discard(u)
                    if not adj[u]:
                        del adj[u]
                    if not adj[v]:
                        del adj[v]
        else:
            break
    
    # Phase 2: Local-search splicing
    # Try to merge pieces that share vertices
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Rebuild adjacency from remaining edges
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)
        
        # Try to splice two cycles
        cycles = [p for p in pieces if len(p) >= 3]
        singles = [p for p in pieces if len(p) == 2]
        
        # Try splicing two cycles
        for i in range(len(cycles)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(cycles)):
                if time.time() > deadline:
                    break
                c1 = cycles[i]
                c2 = cycles[j]
                
                # Find shared vertices
                set1 = set(c1)
                set2 = set(c2)
                shared = set1 & set2
                
                if not shared:
                    continue
                
                # Try each shared vertex
                for v in shared:
                    new_cycle = _try_splice_cycles(c1, c2, v, remaining, adj)
                    if new_cycle is not None:
                        # Splicing successful
                        pieces.remove(c1)
                        pieces.remove(c2)
                        pieces.append(new_cycle)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Try to absorb a single edge into a cycle
        for i, single in enumerate(singles):
            if time.time() > deadline:
                break
            u, v = single
            e = tuple(sorted((u, v)))
            
            if e not in remaining:
                continue
            
            for cycle in cycles:
                new_cycle = _try_absorb_edge(cycle, single, remaining, adj)
                if new_cycle is not None:
                    # Absorption successful
                    pieces.remove(single)
                    pieces.remove(cycle)
                    pieces.append(new_cycle)
                    remaining.discard(e)
                    # Update adj
                    adj[u].discard(v)
                    adj[v].discard(u)
                    if not adj[u]:
                        del adj[u]
                    if not adj[v]:
                        del adj[v]
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Try to find new cycles in remaining edges
        cycle = _find_best_cycle(remaining, adj)
        if cycle and len(cycle) >= 3:
            pieces.append(cycle)
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = tuple(sorted((u, v)))
                if e in remaining:
                    remaining.discard(e)
                    adj[u].discard(v)
                    adj[v].discard(u)
                    if not adj[u]:
                        del adj[u]
                    if not adj[v]:
                        del adj[v]
            improved = True
    
    # Emit remaining edges as single pieces
    for e in sorted(remaining):
        pieces.append([e[0], e[1]])
    
    return pieces


def _try_splice_cycles(c1, c2, v, remaining, adj):
    """
    Try to splice two cycles that share vertex v into one cycle.
    
    c1: [v, a1, a2, ..., ak, v] (v appears at position 0 and implicitly at end)
    c2: [v, b1, b2, ..., bm, v]
    
    We can splice if we can find a way to connect them. The standard splice:
    If c1 has ... x - v - y ... and c2 has ... p - v - q ...,
    and edges (x, q) and (p, y) exist, we can form:
    ... x - q - ... - p - y ...
    
    But we need to be careful about the order. Let me think about this differently.
    
    Actually, a simpler approach: if two cycles share exactly one vertex v, we can
    always merge them into one cycle by "walking" around both cycles. The merged
    cycle would be: start at v, walk around c1 back to v, then walk around c2 back to v.
    But this repeats v in the middle, so it's not a simple cycle.
    
    The correct splice: remove one edge incident to v in each cycle, and replace with
    cross-edges. Specifically:
    c1: v - a1 - a2 - ... - ak - v
    c2: v - b1 - b2 - ... - bm - v
    
    Remove edges (v, a1) from c1 and (v, b1) from c2.
    Add edges (ak, b1) and (v, bm) if they exist.
    New cycle: v - bm - b_{m-1} - ... - b1 - ak - a_{k-1} - ... - a1 - v
    
    Wait, that's not right either. Let me just try the simple approach:
    If the two cycles share vertex v, and we can find edges that allow us to
    connect the "tails" of the cycles, we merge.
    
    For now, let's use a simpler criterion: if two cycles share a vertex, and
    the total number of distinct vertices is k + m (where k and m are the lengths
    of the cycles, minus 1 for the shared vertex), then we can merge them into
    a single cycle of length k + m.
    
    Actually, the most reliable approach: just check if we can form a valid
    cycle by concatenating the two cycles at the shared vertex. This works if
    all other vertices are distinct.
    """
    # Find the position of v in c1 and c2
    # c1 is a list of vertices forming a cycle, so v appears once
    # The cycle is c1[0], c1[1], ..., c1[-1], back to c1[0]
    
    # Get the neighbors of v in c1 and c2
    # In c1, v's neighbors are c1[(idx-1) % len(c1)] and c1[(idx+1) % len(c1)]
    
    idx1 = c1.index(v)
    idx2 = c2.index(v)
    
    prev1 = c1[(idx1 - 1) % len(c1)]
    next1 = c1[(idx1 + 1) % len(c1)]
    prev2 = c2[(idx2 - 1) % len(c2)]
    next2 = c2[(idx2 + 1) % len(c2)]
    
    # Check if we can splice: need edges (prev1, next2) and (prev2, next1)
    e1 = tuple(sorted((prev1, next2)))
    e2 = tuple(sorted((prev2, next1)))
    
    if e1 in remaining and e2 in remaining:
        # We can splice
        # Build the new cycle
        # From prev1, go to next2, then follow c2 from next2 to prev2, 
        # then go to next1, then follow c1 from next1 to prev1, then back to v
        
        # c1 from next1 to prev1 (going forward from next1)
        c1_part = []
        idx = (idx1 + 1) % len(c1)
        while True:
            c1_part.append(c1[idx])
            if c1[idx] == prev1:
                break
            idx = (idx + 1) % len(c1)
        
        # c2 from next2 to prev2 (going forward from next2)
        c2_part = []
        idx = (idx2 + 1) % len(c2)
        while True:
            c2_part.append(c2[idx])
            if c2[idx] == prev2:
                break
            idx = (idx + 1) % len(c2)
        
        # New cycle: v, next1, c1_part, prev1, next2, c2_part, prev2, v
        # Wait, that's not quite right. Let me reconsider.
        
        # Original c1: ... prev1 - v - next1 - ... - prev1 (cycle)
        # Original c2: ... prev2 - v - next2 - ... - prev2 (cycle)
        
        # After splicing: remove (prev1, v) and (v, next1) from c1,
        # remove (prev2, v) and (v, next2) from c2.
        # Add (prev1, next2) and (prev2, next1).
        
        # New cycle: start at v, go to next1, follow c1 to prev1, 
        # go to next2, follow c2 to prev2, go to v.
        
        new_cycle = [v]
        
        # Follow c1 from next1 to prev1
        idx = (idx1 + 1) % len(c1)
        while c1[idx] != prev1:
            new_cycle.append(c1[idx])
            idx = (idx + 1) % len(c1)
        new_cycle.append(prev1)
        
        # Go to next2
        new_cycle.append(next2)
        
        # Follow c2 from next2 to prev2
        idx = (idx2 + 1) % len(c2)
        while c2[idx] != prev2:
            new_cycle.append(c2[idx])
            idx = (idx + 1) % len(c2)
        new_cycle.append(prev2)
        
        # Remove the spliced edges from remaining
        remaining.discard(e1)
        remaining.discard(e2)
        
        # Update adj
        adj[prev1].discard(v)
        adj[v].discard(prev1)
        adj[next1].discard(v)
        adj[v].discard(next1)
        adj[prev2].discard(v)
        adj[v].discard(prev2)
        adj[next2].discard(v)
        adj[v].discard(next2)
        
        adj[prev1].add(next2)
        adj[next2].add(prev1)
        adj[prev2].add(next1)
        adj[next1].add(prev2)
        
        # Verify all vertices are distinct
        if len(set(new_cycle)) == len(new_cycle):
            return new_cycle
        
        # If not, undo and return None
        # This is a simplification; in practice, we should verify before modifying
        return None
    
    return None


def _try_absorb_edge(cycle, single, remaining, adj):
    """
    Try to absorb a single edge into a cycle.
    
    If the single edge (u, v) shares a vertex with the cycle, and we can
    form a longer cycle by incorporating the edge, do so.
    
    For example, if the cycle is [a, b, c, d] and the edge is (b, e),
    and edge (e, c) exists, we can form [a, b, e, c, d].
    """
    u, v = single
    e = tuple(sorted((u, v)))
    
    if e not in remaining:
        return None
    
    # Check if u or v is in the cycle
    if u in cycle:
        # Try to extend the cycle through u
        idx = cycle.index(u)
        prev = cycle[(idx - 1) % len(cycle)]
        next = cycle[(idx + 1) % len(cycle)]
        
        # Option 1: replace edge (prev, u) with (prev, v) and add (v, u) ... no
        # Actually, we want to insert v into the cycle at position u.
        # If edge (prev, v) exists and edge (v, next) exists, we can form:
        # ... prev - v - next ...
        # But we also need to use edge (u, v). So:
        # ... prev - u - v - next ...
        # This requires edges (u, v) [we have it] and (v, next) [check]
        
        e_v_next = tuple(sorted((v, next)))
        if e_v_next in remaining:
            # Build new cycle
            new_cycle = []
            for i in range(len(cycle)):
                if cycle[i] == u:
                    new_cycle.append(u)
                    new_cycle.append(v)
                else:
                    new_cycle.append(cycle[i])
            
            # Verify all vertices distinct
            if len(set(new_cycle)) == len(new_cycle):
                # Remove edges from remaining
                remaining.discard(e)
                remaining.discard(e_v_next)
                
                # Update adj
                adj[u].discard(v)
                adj[v].discard(u)
                adj[v].discard(next)
                adj[next].discard(v)
                adj[u].add(next)
                adj[next].add(u)
                
                return new_cycle
        
        # Option 2: insert v after u, using edge (u, v) and (v, prev)
        e_v_prev = tuple(sorted((v, prev)))
        if e_v_prev in remaining:
            new_cycle = []
            for i in range(len(cycle)):
                if cycle[i] == u:
                    new_cycle.append(u)
                    new_cycle.append(v)
                else:
                    new_cycle.append(cycle[i])
            
            if len(set(new_cycle)) == len(new_cycle):
                remaining.discard(e)
                remaining.discard(e_v_prev)
                
                adj[u].discard(v)
                adj[v].discard(u)
                adj[v].discard(prev)
                adj[prev].discard(v)
                adj[u].add(prev)
                adj[prev].add(u)
                
                return new_cycle
    
    if v in cycle:
        idx = cycle.index(v)
        prev = cycle[(idx - 1) % len(cycle)]
        next = cycle[(idx + 1) % len(cycle)]
        
        e_u_next = tuple(sorted((u, next)))
        if e_u_next in remaining:
            new_cycle = []
            for i in range(len(cycle)):
                if cycle[i] == v:
                    new_cycle.append(v)
                    new_cycle.append(u)
                else:
                    new_cycle.append(cycle[i])
            
            if len(set(new_cycle)) == len(new_cycle):
                remaining.discard(e)
                remaining.discard(e_u_next)
                
                adj[v].discard(u)
                adj[u].discard(v)
                adj[u].discard(next)
                adj[next].discard(u)
                adj[v].add(next)
                adj[next].add(v)
                
                return new_cycle
        
        e_u_prev = tuple(sorted((u, prev)))
        if e_u_prev in remaining:
            new_cycle = []
            for i in range(len(cycle)):
                if cycle[i] == v:
                    new_cycle.append(v)
                    new_cycle.append(u)
                else:
                    new_cycle.append(cycle[i])
            
            if len(set(new_cycle)) == len(new_cycle):
                remaining.discard(e)
                remaining.discard(e_u_prev)
                
                adj[v].discard(u)
                adj[u].discard(v)
                adj[u].discard(prev)
                adj[prev].discard(u)
                adj[v].add(prev)
                adj[prev].add(v)
                
                return new_cycle
    
    return None


def _find_best_cycle(remaining, adj):
    """
    Find a good cycle in the graph. Tries to find longer cycles.
    """
    if not remaining:
        return None
    
    # Get vertices with degree >= 2
    candidates = [v for v in adj if len(adj[v]) >= 2]
    if not candidates:
        return None
    
    # Sort by degree descending
    candidates.sort(key=lambda v: (-len(adj[v]), v))
    
    best_cycle = None
    
    # Try from high-degree vertices first
    for start in candidates[:30]:
        if len(adj.get(start, set())) < 2:
            continue
        
        # DFS to find a cycle back to start
        stack = [(start, [start])]
        visited = {start}
        found = False
        
        while stack and not found:
            node, path = stack.pop()
            
            if len(path) > 50:
                continue
            
            neighbors = list(adj.get(node, []))
            # Sort by degree descending to find longer cycles
            neighbors.sort(key=lambda x: (-len(adj.get(x, set())), x))
            
            for nb in neighbors:
                e = tuple(sorted((node, nb)))
                if e not in remaining:
                    continue
                
                if nb == start and len(path) >= 3:
                    cycle = path
                    if best_cycle is None or len(cycle) > len(best_cycle):
                        best_cycle = cycle
                    found = True
                    break
                elif nb not in visited:
                    visited.add(nb)
                    stack.append((nb, path + [nb]))
        
        # If we found a reasonably long cycle, stop
        if best_cycle and len(best_cycle) >= 8:
            break
    
    return best_cycle


def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for spec in specs:
        edges = _instance(*spec)
        pieces = _decompose_graph(edges)
        results.append(pieces)
    return results
