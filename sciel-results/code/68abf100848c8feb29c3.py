"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    d = 7
    n_vertices = 1 << d
    # Precompute all edges
    all_edges = []
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y))
    
    # Precompute for each edge, the 6 "witness" edges that would form a C4 with it
    # Edge (x, y) differs in bit i. For each other bit j, we get a 4-cycle:
    # x - y, x^(1<<j) - y^(1<<j), x - x^(1<<j), y - y^(1<<j)
    # So adding edge (x,y) is forbidden if any of these 6 edges are all present.
    # Actually, we need the other 3 edges to be present for the C4 to be completed.
    # The 4-cycle has edges: (x,y), (x,c), (y,e), (c,e) where c=x^(1<<j), e=y^(1<<j)
    # So if (x,c), (y,e), (c,e) are all in chosen, we can't add (x,y).
    
    edge_witnesses = {}
    for (x, y) in all_edges:
        # Find which bit differs
        diff = x ^ y
        i = diff.bit_length() - 1
        witnesses = []
        for j in range(d):
            if j == i:
                continue
            c = x ^ (1 << j)
            e = y ^ (1 << j)
            # The 3 other edges in the 4-cycle
            e1 = (min(x, c), max(x, c))
            e2 = (min(y, e), max(y, e))
            e3 = (min(c, e), max(c, e))
            witnesses.append((e1, e2, e3))
        edge_witnesses[(x, y)] = witnesses
    
    # Precompute edge index for quick lookup
    edge_set = set(all_edges)
    
    def greedy_randomized(seed):
        rng = random.Random(seed)
        chosen = set()
        # Random order of edges
        edges = all_edges[:]
        rng.shuffle(edges)
        
        for (x, y) in edges:
            ok = True
            for (e1, e2, e3) in edge_witnesses[(x, y)]:
                if e1 in chosen and e2 in chosen and e3 in chosen:
                    ok = False
                    break
            if ok:
                chosen.add((x, y))
        
        return list(chosen)
    
    def local_improve(edge_list):
        """Try to add edges that were rejected, by removing conflicting edges."""
        chosen = set(edge_list)
        improved = True
        while improved:
            improved = False
            # Find edges not in chosen
            not_chosen = [e for e in all_edges if e not in chosen]
            random.shuffle(not_chosen)
            for (x, y) in not_chosen:
                # Check if adding (x,y) would create a C4
                conflicts = []
                for (e1, e2, e3) in edge_witnesses[(x, y)]:
                    if e1 in chosen and e2 in chosen and e3 in chosen:
                        # We need to remove one of e1, e2, e3 to add (x,y)
                        # Try removing each and see if it's beneficial
                        for rem_edge in [e1, e2, e3]:
                            # Remove rem_edge and add (x,y), check if still valid
                            # For simplicity, just try: remove rem_edge, add (x,y)
                            # This is a simple 1-swap
                            new_chosen = set(chosen)
                            new_chosen.discard(rem_edge)
                            new_chosen.add((x, y))
                            # Verify no C4 was created by adding (x,y)
                            valid = True
                            for (f1, f2, f3) in edge_witnesses[(x, y)]:
                                if f1 in new_chosen and f2 in new_chosen and f3 in new_chosen:
                                    valid = False
                                    break
                            if valid:
                                # Also check that removing rem_edge didn't break anything
                                # (it can't, since removing an edge can't create a C4)
                                chosen = new_chosen
                                improved = True
                                break
                        if improved:
                            break
                if not improved:
                    continue
        return list(chosen)
    
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    best = None
    best_size = -1
    seed = 0
    
    while time.time() < deadline:
        edges = greedy_randomized(seed)
        size = len(edges)
        
        # Try local improvement if we have time
        if time.time() < deadline - 5:
            edges = local_improve(edges)
            size = len(edges)
        
        if size > best_size:
            best_size = size
            best = edges
        
        seed += 1
    
    if best is None:
        # Fallback: deterministic greedy
        chosen = set()
        for (x, y) in all_edges:
            ok = True
            for (e1, e2, e3) in edge_witnesses[(x, y)]:
                if e1 in chosen and e2 in chosen and e3 in chosen:
                    ok = False
                    break
            if ok:
                chosen.add((x, y))
        best = list(chosen)
    
    return best
# EVOLVE-BLOCK-END
