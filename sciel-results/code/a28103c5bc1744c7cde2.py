"""Rewritten program: randomized greedy + local improvement for C4-free edge sets in Q7."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    d = 7
    n_vertices = 1 << d
    deadline = time.time() + 140  # safety margin before 150s kill

    # Precompute all edges as (u, v) with u < v
    all_edges = []
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y))
    total_edges = len(all_edges)  # 448

    # Precompute for each edge, the list of other edges that would form a C4 with it
    # Edge (x, y) where y = x ^ (1<<i). A C4 is formed if there exists j != i such that
    # edges (x, x^j), (y, y^j), and (x^j, y^j) are all present.
    # For each edge, precompute the set of "conflict pairs" - for each j != i,
    # the two edges (x, x^j) and (y, y^j) such that if both are present, adding (x,y) creates a C4.
    # Actually, adding (x,y) creates a C4 if for some j != i, both (x, x^j) and (y, y^j) are in the set.
    # The third edge (x^j, y^j) is automatically determined.
    
    # Let's precompute: for each edge e = (x, y), for each j != i, the pair of edges that conflict.
    # edge_map: maps edge tuple to index
    edge_index = {}
    for idx, e in enumerate(all_edges):
        edge_index[e] = idx
    
    # For each edge, store list of (edge_a, edge_b) pairs such that if both edge_a and edge_b are chosen,
    # then choosing this edge would create a C4.
    conflict_pairs = [[] for _ in range(total_edges)]
    
    for idx, (x, y) in enumerate(all_edges):
        # Determine which bit differs
        diff = x ^ y
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            c = x ^ (1 << j)
            e2 = y ^ (1 << j)
            # Edge (x, c) and edge (y, e2)
            ea = (min(x, c), max(x, c))
            eb = (min(y, e2), max(y, e2))
            if ea in edge_index and eb in edge_index:
                conflict_pairs[idx].append((edge_index[ea], edge_index[eb]))
    
    def run_greedy(seed):
        """Run randomized greedy with given seed. Returns list of edges."""
        rng = random.Random(seed)
        order = list(range(total_edges))
        rng.shuffle(order)
        
        chosen = [False] * total_edges
        count = 0
        
        for idx in order:
            x, y = all_edges[idx]
            # Check if adding this edge creates any C4
            ok = True
            for (ia, ib) in conflict_pairs[idx]:
                if chosen[ia] and chosen[ib]:
                    ok = False
                    break
            if ok:
                chosen[idx] = True
                count += 1
        
        return [list(all_edges[i]) for i in range(total_edges) if chosen[i]]
    
    def local_improve(edges, seed):
        """Try to improve a solution by removing and re-adding edges."""
        rng = random.Random(seed)
        # Build chosen set
        chosen_set = set(tuple(e) for e in edges)
        chosen_indices = set()
        for i, e in enumerate(all_edges):
            if tuple(e) in chosen_set:
                chosen_indices.add(i)
        
        count = len(chosen_indices)
        best_count = count
        best_set = set(chosen_indices)
        
        # Try removing random edges and re-adding greedily
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Pick a random edge to remove
            if not chosen_indices:
                break
            remove_idx = rng.choice(list(chosen_indices))
            new_set = set(chosen_indices)
            new_set.discard(remove_idx)
            
            # Try to re-add the removed edge and other edges greedily
            # First, try to re-add remove_idx
            can_add = True
            for (ia, ib) in conflict_pairs[remove_idx]:
                if ia in new_set and ib in new_set:
                    can_add = False
                    break
            if can_add:
                new_set.add(remove_idx)
            
            # Try to add other edges that weren't in the set
            for idx in range(total_edges):
                if idx in new_set:
                    continue
                ok = True
                for (ia, ib) in conflict_pairs[idx]:
                    if ia in new_set and ib in new_set:
                        ok = False
                        break
                if ok:
                    new_set.add(idx)
            
            if len(new_set) > best_count:
                best_count = len(new_set)
                best_set = set(new_set)
                chosen_indices = set(new_set)
        
        return [list(all_edges[i]) for i in sorted(best_set)]
    
    # Run multiple randomized greedy attempts
    best_edges = None
    best_count = 0
    seed = 42
    
    while time.time() < deadline - 10:
        edges = run_greedy(seed)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            # Try local improvement
            if time.time() < deadline - 15:
                improved = local_improve(edges, seed + 1000000)
                if len(improved) > best_count:
                    best_count = len(improved)
                    best_edges = improved
        
        seed += 1
        if seed % 100 == 0:
            # Occasionally try a different strategy: start from best and do more local search
            pass
    
    return best_edges if best_edges else []
# EVOLVE-BLOCK-END
