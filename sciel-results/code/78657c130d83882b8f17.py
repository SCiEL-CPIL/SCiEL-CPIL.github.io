import time
import random
import numpy as np

def construct():
    d = 7
    n = 2 ** d
    deadline = time.time() + 115.0

    edges = []
    edge_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_idx[(x, y)] = idx
                edge_idx[(y, x)] = idx

    m = len(edges)  # 448

    # Build conflict graph
    conflict = [set() for _ in range(m)]
    
    for b in range(d):
        for c in range(b + 1, d):
            for a in range(n):
                if (a & (1 << b)) or (a & (1 << c)):
                    continue
                v0 = a
                v1 = a ^ (1 << b)
                v2 = a ^ (1 << c)
                v3 = a ^ (1 << b) ^ (1 << c)
                e0 = edge_idx[(min(v0, v1), max(v0, v1))]
                e1 = edge_idx[(min(v0, v2), max(v0, v2))]
                e2 = edge_idx[(min(v1, v3), max(v1, v3))]
                e3 = edge_idx[(min(v2, v3), max(v2, v3))]
                # Add conflicts for all pairs in this face
                face_edges = [e0, e1, e2, e3]
                for i in range(4):
                    for j in range(i+1, 4):
                        conflict[face_edges[i]].add(face_edges[j])
                        conflict[face_edges[j]].add(face_edges[i])
    
    # Convert to bitmasks
    conflict_masks = [0] * m
    for i in range(m):
        mask = 0
        for j in conflict[i]:
            mask |= (1 << j)
        conflict_masks[i] = mask
        
    # non_conflict_masks[i] contains all vertices j != i such that i and j do NOT conflict
    all_mask = (1 << m) - 1
    non_conflict_masks = [0] * m
    for i in range(m):
        # Vertices that do not conflict with i, excluding i itself
        # conflict_masks[i] has bits for j where i conflicts with j.
        # ~conflict_masks[i] has bits for j where i does NOT conflict with j.
        # We need to mask out i itself.
        non_conflict_masks[i] = (all_mask ^ conflict_masks[i]) ^ (1 << i)

    best_size = 0
    best_iset = []
    
    # 1. Greedy Initialization
    def greedy_init(seed):
        rng = random.Random(seed)
        order = list(range(m))
        rng.shuffle(order)
        selected = 0
        count = 0
        for v in order:
            if (selected >> v) & 1:
                continue
            if selected & conflict_masks[v]:
                continue
            selected |= (1 << v)
            count += 1
        return count, selected

    # 2. Local Search
    def local_search(iset_mask, max_iters=2000):
        nonlocal best_size, best_iset
        current = iset_mask
        current_size = bin(current).count('1')
        
        for iteration in range(max_iters):
            if time.time() >= deadline - 5.0:
                break
            
            improved = False
            
            # 1-flip: Add a single edge
            # Find addable edges
            addable = 0
            # Iterate through all edges not in current
            # This is O(m) per check, which is fine for m=448
            for v in range(m):
                if not ((current >> v) & 1):
                    if not (current & conflict_masks[v]):
                        addable |= (1 << v)
            
            if addable:
                # Pick a random addable edge
                bits = []
                am = addable
                while am:
                    lsb = am & (-am)
                    bits.append(lsb.bit_length() - 1)
                    am ^= lsb
                rng = random.Random(iteration)
                pick = rng.choice(bits)
                new_mask = current | (1 << pick)
                new_size = current_size + 1
                if new_size > best_size:
                    best_size = new_size
                    best_iset = [i for i in range(m) if (new_mask >> i) & 1]
                current = new_mask
                current_size = new_size
                improved = True
            
            # 2-swap: Remove one, add two
            if not improved:
                sel_list = [i for i in range(m) if (current >> i) & 1]
                if sel_list:
                    rng = random.Random(iteration + 10000)
                    # Try a few random removals
                    for _ in range(10):
                        ri = rng.choice(sel_list)
                        temp_mask = current ^ (1 << ri)
                        
                        # Find addable edges for temp_mask
                        addable_list = []
                        for v in range(m):
                            if not ((temp_mask >> v) & 1):
                                if not (temp_mask & conflict_masks[v]):
                                    addable_list.append(v)
                        
                        if len(addable_list) >= 2:
                            # Try to find two non-conflicting addable edges
                            # Sample pairs
                            for _ in range(50):
                                i, j = rng.sample(addable_list, 2)
                                if not ((conflict_masks[i] >> j) & 1):
                                    new_mask = temp_mask | (1 << i) | (1 << j)
                                    new_size = current_size - 1 + 2
                                    if new_size > best_size:
                                        best_size = new_size
                                        best_iset = [k for k in range(m) if (new_mask >> k) & 1]
                                    current = new_mask
                                    current_size = new_size
                                    improved = True
                                    break
                        if improved:
                            break
            
            if not improved:
                break
        
        return current

    # Run multi-start
    for seed in range(50):
        if time.time() >= deadline - 10.0:
            break
        count, selected = greedy_init(seed)
        if count > best_size:
            best_size = count
            best_iset = [i for i in range(m) if (selected >> i) & 1]
        
        if time.time() < deadline - 5.0:
            improved_mask = local_search(selected, max_iters=500)
            improved_size = bin(improved_mask).count('1')
            if improved_size > best_size:
                best_size = improved_size
                best_iset = [i for i in range(m) if (improved_mask >> i) & 1]

    # 3. Branch and Bound MIS on the current best solution's complement? 
    # Or run B&B globally with a time limit.
    
    # Let's try a simple B&B to see if we can beat the local search.
    # We will run it on the full graph, but prune aggressively.
    
    global_best_size = best_size
    global_best_mask = 0
    for i in best_iset:
        global_best_mask |= (1 << i)
        
    def mis_bnb(candidates, current_size, current_mask):
        nonlocal global_best_size, global_best_mask
        if time.time() >= deadline - 2.0:
            return
        
        if candidates == 0:
            if current_size > global_best_size:
                global_best_size = current_size
                global_best_mask = current_mask
            return
        
        # Upper bound: current_size + number of bits in candidates
        # If this is <= global_best_size, prune
        if current_size + bin(candidates).count('1') <= global_best_size:
            return
            
        # Pick a vertex v from candidates
        # Heuristic: pick a vertex with high degree in the induced subgraph
        # To keep it simple, pick the first one, or a random one?
        # Let's pick the one with the most conflicts within candidates.
        # This is expensive to compute for every node.
        # Simpler: pick the first set bit.
        lsb = candidates & (-candidates)
        v = lsb.bit_length() - 1
        candidates ^= lsb
        
        # Branch 1: Include v
        # New candidates: candidates that don't conflict with v
        new_candidates = candidates & non_conflict_masks[v]
        mis_bnb(new_candidates, current_size + 1, current_mask | (1 << v))
        
        # Branch 2: Exclude v
        mis_bnb(candidates, current_size, current_mask)
        
    # Run B&B from scratch? Or from the greedy solution?
    # Running from scratch with B&B might find a better global optimum.
    # But it might be slow.
    # Let's try running it with a small time budget.
    
    if time.time() < deadline - 10.0:
        # Start with an empty set? No, start with a good lower bound.
        # We already have best_size.
        # Let's run B&B on the full graph.
        # To speed up, we can order vertices by degree.
        # But for now, let's just try it.
        # We need to be careful with recursion depth. m=448, depth can be 448.
        # Python recursion limit is usually 1000, so it's fine.
        
        # Optimization: Use a stack-based approach or just hope it's fast.
        # Given the sparsity, B&B should be very effective.
        
        # Let's run it.
        mis_bnb(all_mask, 0, 0)
        
        if global_best_size > best_size:
            best_size = global_best_size
            best_iset = [i for i in range(m) if (global_best_mask >> i) & 1]

    out = []
    for ei in best_iset:
        x, y = edges[ei]
        out.append([x, y])
    
    return out
