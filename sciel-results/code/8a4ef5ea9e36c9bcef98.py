"""Construct a K2,3-free graph on 20 vertices maximizing edges."""
import time
import random
from itertools import combinations

N = 20
DEADLINE = 110.0  # safety margin before 120s limit


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    # Total possible edges in K20
    all_edges = list(combinations(range(N), 2))
    m_total = len(all_edges)  # 190

    # Edge index for quick lookup
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}

    # Adjacency representation: adj[v] is a set of neighbors
    # We'll use bitmasks for speed: adj_mask[v] is an integer bitmask of neighbors
    adj_mask = [0] * N
    # Track which edges are present
    present = [False] * m_total

    def get_neighbors(v):
        """Return set of neighbors of v."""
        return [u for u in range(N) if (adj_mask[v] >> u) & 1]

    def can_add_edge(u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free.
        After adding (u,v), for every pair of vertices (a,b), the number of common
        neighbors must be at most 2.
        Adding (u,v) increases common neighbor counts:
        - common(u, v) increases by 1 (they now share... wait no)
        Actually, adding edge (u,v):
        - For any vertex w adjacent to both u and v: w is a common neighbor of u and v.
          So common(u,v) increases by 1 for each w that is neighbor of both u and v.
          Wait, that's not right either. Let me think more carefully.
        
        When we add edge (u,v):
        - The pair (u,v) now has as common neighbors: N(u) ∩ N(v) (the set of vertices adjacent to both).
          Before adding, u and v weren't connected, but they could still have common neighbors.
          After adding, the common neighbors of u and v are still N(u) ∩ N(v) (same set, since
          adding edge (u,v) doesn't change who is adjacent to u or v except that u becomes
          adjacent to v and v becomes adjacent to u).
          
          Actually: after adding (u,v), N(u) gains v and N(v) gains u.
          So common neighbors of u and v after = (N(u) ∪ {v}) ∩ (N(v) ∪ {u})
          = (N(u) ∩ N(v)) ∪ (N(u) ∩ {u}) ∪ ({v} ∩ N(v)) ∪ ({v} ∩ {u})
          = N(u) ∩ N(v)  (since u not in N(u) for simple graph, v not in N(v))
          Wait: N(u) ∩ {u} = ∅ (no self-loops), {v} ∩ N(v) = ∅.
          And {v} ∩ {u} = ∅.
          So common(u,v) after = N(u) ∩ N(v) (the original intersection, before adding).
          
          Hmm, but that means adding (u,v) doesn't change common(u,v)? That seems wrong.
          
          Let me reconsider. Before adding (u,v): u and v are not adjacent.
          N(u) and N(v) are their neighbor sets.
          Common neighbors of u and v = N(u) ∩ N(v).
          
          After adding (u,v): u is now adjacent to v, so N(u) becomes N(u) ∪ {v}.
          v is now adjacent to u, so N(v) becomes N(v) ∪ {u}.
          Common neighbors of u and v = (N(u) ∪ {v}) ∩ (N(v) ∪ {u})
          = (N(u) ∩ N(v)) ∪ (N(u) ∩ {u}) ∪ ({v} ∩ N(v)) ∪ ({v} ∩ {u})
          = N(u) ∩ N(v) ∪ ∅ ∪ ∅ ∪ ∅
          = N(u) ∩ N(v)
          
          So indeed, common(u,v) doesn't change when we add edge (u,v). Good.
        
        - But for other pairs (a, b) where a ≠ u and b ≠ v, or one of them is u or v:
          If a = u and b = some other vertex w ≠ v:
          common(u, w) after = N(u) ∩ N(w) where N(u) now includes v.
          So common(u, w) after = (N(u)_old ∪ {v}) ∩ N(w) = (N(u)_old ∩ N(w)) ∪ ({v} ∩ N(w))
          = old_common(u,w) + (1 if v ∈ N(w) else 0)
          
          Similarly for common(v, w) where w ≠ u:
          common(v, w) after = old_common(v,w) + (1 if u ∈ N(w) else 0)
        
        So the constraint violations that could be introduced:
        1. For each w adjacent to v (w ≠ u): common(u, w) increases by 1.
           If common(u, w) was already 2, this would make it 3 -> violation.
        2. For each w adjacent to u (w ≠ v): common(v, w) increases by 1.
           If common(v, w) was already 2, this would make it 3 -> violation.
        """
        # Check condition 1: for each w in N(v) (w ≠ u), common(u,w) must be < 2
        mask_u = adj_mask[u]
        mask_v = adj_mask[v]
        
        # For each w that is a neighbor of v (and w != u):
        # common(u, w) = popcount(mask_u & adj_mask[w])
        # We need this to be <= 1 (since it will increase by 1)
        w = 0
        temp = mask_v
        while temp:
            if temp & 1:
                if w != u:
                    common = bin(mask_u & adj_mask[w]).count('1')
                    if common >= 2:
                        return False
            temp >>= 1
            w += 1
        
        # For each w that is a neighbor of u (and w != v):
        # common(v, w) = popcount(mask_v & adj_mask[w])
        # We need this to be <= 1
        w = 0
        temp = mask_u
        while temp:
            if temp & 1:
                if w != v:
                    common = bin(mask_v & adj_mask[w]).count('1')
                    if common >= 2:
                        return False
            temp >>= 1
            w += 1
        
        return True

    def add_edge(u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        present[edge_to_idx[(min(u,v), max(u,v))]] = True

    def remove_edge(u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        present[edge_to_idx[(min(u,v), max(u,v))]] = False

    # Initialize with a good starting structure: 5 copies of K4
    # Actually, let's start with a greedy construction
    # Greedy: try to add edges in order of some heuristic
    
    # Start with empty graph and greedily add edges
    # Use a randomized greedy approach with multiple restarts
    
    best_edges = 0
    best_adj = None
    
    def greedy_build():
        """Greedy construction with random edge ordering."""
        nonlocal adj_mask, present
        adj_mask = [0] * N
        present = [False] * m_total
        
        order = list(range(m_total))
        random.shuffle(order)
        
        count = 0
        for idx in order:
            u, v = all_edges[idx]
            if can_add_edge(u, v):
                add_edge(u, v)
                count += 1
        
        return count, adj_mask[:]

    # First, do a few greedy builds
    for i in range(20):
        if time.time() > deadline - 10:
            break
        count, adj_copy = greedy_build()
        if count > best_edges:
            best_edges = count
            best_adj = adj_copy[:]

    # Now use simulated annealing / local search to improve
    # Start from best found so far
    if best_adj is not None:
        adj_mask = best_adj[:]
        # Rebuild present array
        present = [False] * m_total
        for u in range(N):
            mask = adj_mask[u]
            w = 0
            temp = mask
            while temp:
                if temp & 1:
                    if w > u:
                        present[edge_to_idx[(u, w)]] = True
                temp >>= 1
                w += 1

    # Simulated annealing style local search
    # Moves: remove an edge, add a different edge
    current_edges = best_edges
    current_adj = adj_mask[:]
    
    temperature = 2.0
    cooling = 0.999
    
    while time.time() < deadline - 5:
        # Try a move: remove one edge, try to add another
        # Pick a random edge to remove
        edge_indices = [i for i in range(m_total) if present[i]]
        if not edge_indices:
            break
        
        remove_idx = random.choice(edge_indices)
        ru, rv = all_edges[remove_idx]
        
        # Save state
        saved_adj = current_adj[:]
        
        remove_edge(ru, rv)
        current_edges -= 1
        
        # Try to add a random non-present edge
        non_present = [i for i in range(m_total) if not present[i]]
        if not non_present:
            # Restore
            current_adj = saved_adj
            adj_mask = current_adj
            present = [False] * m_total
            for u in range(N):
                mask = adj_mask[u]
                w = 0
                temp = mask
                while temp:
                    if temp & 1:
                        if w > u:
                            present[edge_to_idx[(u, w)]] = True
                    temp >>= 1
                    w += 1
            current_edges = best_edges
            break
        
        add_idx = random.choice(non_present)
        au, av = all_edges[add_idx]
        
        if can_add_edge(au, av):
            add_edge(au, av)
            current_edges += 1
        
        # Check if improved
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = current_adj[:]
            # Reset temperature on improvement
            temperature = 2.0
        else:
            # Accept with probability based on temperature
            delta = current_edges - best_edges
            if delta < 0:
                # Compute acceptance probability
                # Simple: accept if random < temperature
                if random.random() > temperature:
                    # Reject: restore
                    current_adj = saved_adj
                    adj_mask = current_adj
                    present = [False] * m_total
                    for u in range(N):
                        mask = adj_mask[u]
                        w = 0
                        temp = mask
                        while temp:
                            if temp & 1:
                                if w > u:
                                    present[edge_to_idx[(u, w)]] = True
                            temp >>= 1
                            w += 1
                    current_edges = best_edges
        
        temperature *= cooling
        if temperature < 0.1:
            temperature = 0.1

    # Final polish: try to add any remaining edges greedily
    adj_mask = best_adj[:]
    present = [False] * m_total
    for u in range(N):
        mask = adj_mask[u]
        w = 0
        temp = mask
        while temp:
            if temp & 1:
                if w > u:
                    present[edge_to_idx[(u, w)]] = True
            temp >>= 1
            w += 1
    
    # Try to add more edges
    for i in range(3):
        if time.time() > deadline - 2:
            break
        improved = True
        while improved:
            improved = False
            for idx in range(m_total):
                if not present[idx]:
                    u, v = all_edges[idx]
                    if can_add_edge(u, v):
                        add_edge(u, v)
                        best_edges += 1
                        improved = True

    # Build final edge list
    edges = []
    for u in range(N):
        mask = best_adj[u]
        w = 0
        temp = mask
        while temp:
            if temp & 1:
                if w > u:
                    edges.append([u, w])
            temp >>= 1
            w += 1

    return edges
