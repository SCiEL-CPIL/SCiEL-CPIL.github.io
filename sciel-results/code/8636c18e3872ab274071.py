import time
import random
import numpy as np

# EVOLVE-BLOCK-START
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

    # Build conflict graph: two edges conflict if they share a face
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
                for i in range(4):
                    for j in range(i+1, 4):
                        conflict[e0 if i==0 else e1 if i==1 else e2 if i==2 else e3].add(
                            e0 if j==0 else e1 if j==1 else e2 if j==2 else e3)
                        conflict[e0 if j==0 else e1 if j==1 else e2 if j==2 else e3].add(
                            e0 if i==0 else e1 if i==1 else e2 if i==2 else e3)
    
    # Convert to bitmasks for fast bit-parallel operations
    conflict_masks = [0] * m
    for i in range(m):
        mask = 0
        for j in conflict[i]:
            mask |= (1 << j)
        conflict_masks[i] = mask
    
    # Complement masks: non-conflicting edges (excluding self)
    all_mask = (1 << m) - 1
    complement_masks = [0] * m
    for i in range(m):
        complement_masks[i] = all_mask ^ conflict_masks[i] ^ (1 << i)
    
    best_iset = []
    best_size = 0
    
    # Multi-start greedy with different orderings
    def greedy_order(order, seed):
        """Greedy independent set using given vertex order."""
        selected = 0
        count = 0
        for v in order:
            if (selected >> v) & 1:
                continue
            # Check if v conflicts with any selected
            if selected & conflict_masks[v]:
                continue
            selected |= (1 << v)
            count += 1
        return count, selected
    
    # Try multiple random orderings
    for seed in range(200):
        if time.time() >= deadline - 10.0:
            break
        rng = random.Random(seed)
        order = list(range(m))
        rng.shuffle(order)
        count, selected = greedy_order(order, seed)
        if count > best_size:
            best_size = count
            best_iset = [i for i in range(m) if (selected >> i) & 1]
    
    # Local search: 1-flip (add one edge) and 2-swap (remove one, add two)
    def local_search(iset_mask, max_iters=2000):
        """Improve independent set via local search."""
        nonlocal best_size, best_iset
        current = iset_mask
        current_size = bin(current).count('1')
        
        for iteration in range(max_iters):
            if time.time() >= deadline - 5.0:
                break
            
            improved = False
            
            # Try 1-flip: add a single edge
            # Find all edges that don't conflict with current set
            addable_mask = 0
            for v in range(m):
                if not ((current >> v) & 1):
                    if not (current & conflict_masks[v]):
                        addable_mask |= (1 << v)
            
            if addable_mask:
                # Pick a random addable edge
                bits = []
                am = addable_mask
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
            
            # Try 2-swap: remove one edge, add two non-conflicting edges
            if not improved:
                # Get list of selected edges
                sel_list = [i for i in range(m) if (current >> i) & 1]
                if sel_list:
                    rng = random.Random(iteration + 10000)
                    for _ in range(min(20, len(sel_list))):
                        ri = rng.choice(sel_list)
                        temp_mask = current ^ (1 << ri)
                        temp_size = current_size - 1
                        
                        # Find addable edges for temp_mask
                        addable = []
                        for v in range(m):
                            if not ((temp_mask >> v) & 1):
                                if not (temp_mask & conflict_masks[v]):
                                    addable.append(v)
                        
                        if len(addable) >= 2:
                            # Try to find two non-conflicting addable edges
                            for _ in range(30):
                                i, j = rng.sample(addable, 2)
                                if not ((conflict_masks[i] >> j) & 1):
                                    new_mask = temp_mask | (1 << i) | (1 << j)
                                    new_size = temp_size + 2
                                    if new_size > best_size:
                                        best_size = new_size
                                        best_iset = [k for k in range(m) if (new_mask >> k) & 1]
                                    current = new_mask
                                    current_size = new_size
                                    improved = True
                                    break
            
            if not improved:
                break
        
        return current
    
    # Run local search from best solution
    if best_iset and time.time() < deadline - 15.0:
        best_mask = 0
        for i in best_iset:
            best_mask |= (1 << i)
        improved_mask = local_search(best_mask, max_iters=5000)
        improved_size = bin(improved_mask).count('1')
        if improved_size > best_size:
            best_size = improved_size
            best_iset = [i for i in range(m) if (improved_mask >> i) & 1]
    
    # Additional multi-start local search
    for seed in range(50):
        if time.time() >= deadline - 5.0:
            break
        rng = random.Random(seed + 1000)
        order = list(range(m))
        rng.shuffle(order)
        count, selected = greedy_order(order, seed + 1000)
        if count > best_size:
            best_size = count
            best_iset = [i for i in range(m) if (selected >> i) & 1]
        
        if time.time() < deadline - 5.0:
            improved_mask = local_search(selected, max_iters=1000)
            improved_size = bin(improved_mask).count('1')
            if improved_size > best_size:
                best_size = improved_size
                best_iset = [i for i in range(m) if (improved_mask >> i) & 1]
    
    out = []
    for ei in best_iset:
        x, y = edges[ei]
        out.append([x, y])
    
    return out
# EVOLVE-BLOCK-END
