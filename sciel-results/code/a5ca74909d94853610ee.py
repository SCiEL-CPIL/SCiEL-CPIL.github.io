import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    d = 7
    n_vertices = 1 << d
    all_edges = []
    edge_to_idx = {}
    
    # Precompute all edges and their indices
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_to_idx[(x, y)] = idx
    
    m_edges = len(all_edges)
    
    # Precompute conflicts: for each edge, which other edges form a C4 with it
    # A C4 is formed by two parallel edges (same dimension, different position) 
    # Wait, no. A 2D face in Q7 consists of 4 vertices: x, x^i, x^j, x^i^j
    # Edges: (x, x^i), (x, x^j), (x^i, x^i^j), (x^j, x^i^j)
    # So for an edge (u, v) where v = u ^ (1<<i), the C4s containing it involve
    # any j != i. The other three edges are:
    # (u ^ (1<<j), u ^ (1<<i) ^ (1<<j))  -> parallel to (u, v)
    # (u, u ^ (1<<j))
    # (v, v ^ (1<<j))
    
    # conflicts[e] = set of edge indices that, if all three others are present, 
    # adding e would complete a C4. Actually, we need to check if adding e completes a C4.
    # That means the other 3 edges of the face are already in the set.
    
    # Let's precompute for each edge e, the list of "conflict triples" - sets of 3 edges 
    # that together with e form a C4. But that's too big.
    # Better: for each edge e, and for each dimension j != dim(e), the face is determined.
    # The other 3 edges are fixed. So conflicts[e] is a list of tuples of 3 edge indices.
    
    conflicts = [[] for _ in range(m_edges)]
    
    for idx, (u, v) in enumerate(all_edges):
        # Determine dimension i
        diff = u ^ v
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            # The face is {u, v, u^(1<<j), v^(1<<j)}
            # Other edges:
            e1 = (min(u, u ^ (1<<j)), max(u, u ^ (1<<j)))
            e2 = (min(v, v ^ (1<<j)), max(v, v ^ (1<<j)))
            e3 = (min(u ^ (1<<j), v ^ (1<<j)), max(u ^ (1<<j), v ^ (1<<j)))
            
            idx1 = edge_to_idx[e1]
            idx2 = edge_to_idx[e2]
            idx3 = edge_to_idx[e3]
            
            conflicts[idx].append((idx1, idx2, idx3))
    
    best_count = 0
    best_edges = None
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Randomized greedy
        edge_order = list(range(m_edges))
        rng.shuffle(edge_order)
        
        chosen = [False] * m_edges
        count = 0
        
        for e_idx in edge_order:
            # Check if adding e_idx completes any C4
            ok = True
            for (a, b, c) in conflicts[e_idx]:
                if chosen[a] and chosen[b] and chosen[c]:
                    ok = False
                    break
            if ok:
                chosen[e_idx] = True
                count += 1
        
        # Local search: try to remove one edge and add two others? Or just swap?
        # Simpler: try to add edges that were skipped, by removing one conflicting edge.
        # Or: random walk - remove a random edge, then greedily add back as many as possible.
        
        # Let's do a simple local search: 
        # 1. Identify edges not in chosen.
        # 2. For each, check if removing one of its conflict edges allows it to be added, 
        #    and if that leads to a net gain (unlikely with 1-for-1, but maybe with re-add).
        
        # Better local search: 
        # Try to find an edge e not in chosen such that there exists a conflict triple (a,b,c)
        # where at most 2 of a,b,c are in chosen. If exactly 2 are in chosen, we can remove one 
        # and add e. Net change: -1 + 1 = 0. No gain.
        # To get a gain, we need to remove 1 and add >1.
        
        # Alternative: 2-opt style. Remove two edges, add three?
        # This is getting complex. Let's stick to the randomized greedy with many restarts.
        # The key insight from the prompt is that randomized greedy with precomputed checks 
        # allows thousands of restarts.
        
        if count > best_count:
            best_count = count
            best_edges = [all_edges[i] for i in range(m_edges) if chosen[i]]
            
            # Optional: quick local improvement on the best found
            # Try to add any edge that can be added without conflict (shouldn't exist if greedy was maximal)
            # But greedy is maximal in the sense that no single edge can be added.
            # So we need to break maximality to improve.
            
            # Let's try a simple "kick": remove a random edge from chosen, then greedily re-add.
            if time.time() < deadline - 5:
                chosen_copy = chosen[:]
                # Remove a random edge
                chosen_indices = [i for i in range(m_edges) if chosen_copy[i]]
                if chosen_indices:
                    rem_idx = rng.choice(chosen_indices)
                    chosen_copy[rem_idx] = False
                    count_copy = count - 1
                    
                    # Greedily add back
                    for e_idx in edge_order:
                        if not chosen_copy[e_idx]:
                            ok = True
                            for (a, b, c) in conflicts[e_idx]:
                                if chosen_copy[a] and chosen_copy[b] and chosen_copy[c]:
                                    ok = False
                                    break
                            if ok:
                                chosen_copy[e_idx] = True
                                count_copy += 1
                    
                    if count_copy > best_count:
                        best_count = count_copy
                        best_edges = [all_edges[i] for i in range(m_edges) if chosen_copy[i]]
    
    return [list(e) for e in best_edges] if best_edges else []
