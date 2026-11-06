"""Optimized search for graphs on 16 vertices with 65 edges minimizing max triangle degree sum."""
import time
import random

def construct():
    deadline = time.time() + 145.0  # safety margin before 150s kill
    n = 16
    target_edges = 65
    
    def make_bitmasks(edges):
        masks = [0] * n
        for u, v in edges:
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
        return masks
    
    def edges_from_masks(masks):
        edges = []
        for u in range(n):
            m = masks[u]
            for v in range(u + 1, n):
                if (m >> v) & 1:
                    edges.append([u, v])
        return edges
    
    def compute_score(masks):
        degs = [bin(m).count('1') for m in masks]
        max_score = 0
        for u in range(n):
            mu = masks[u]
            for v in range(u + 1, n):
                if (mu >> v) & 1:
                    common = mu & masks[v]
                    while common:
                        lsb = common & (-common)
                        w = lsb.bit_length() - 1
                        s = degs[u] + degs[v] + degs[w]
                        if s > max_score:
                            max_score = s
                        common ^= lsb
        return max_score
    
    def local_search(masks, deadline, max_iters=200000):
        """Local search maintaining edge list for efficiency."""
        # Build edge list
        edge_set = set()
        edge_list = []
        for u in range(n):
            m = masks[u]
            for v in range(u + 1, n):
                if (m >> v) & 1:
                    edge_set.add((u, v))
                    edge_list.append((u, v))
        
        current_score = compute_score(masks)
        best_score = current_score
        best_masks = masks[:]
        degs = [bin(m).count('1') for m in masks]
        
        it = 0
        while it < max_iters:
            it += 1
            if time.time() > deadline - 5.0:
                break
            
            # Pick random edge to remove
            a, b = random.choice(edge_list)
            
            # Pick random non-edge to add (sample a few times)
            found = False
            for _ in range(20):
                c = random.randint(0, n - 1)
                d = random.randint(0, n - 1)
                if c == d:
                    continue
                if c > d:
                    c, d = d, c
                if (c, d) in edge_set:
                    continue
                if (c, d) == (a, b):
                    continue
                found = True
                break
            
            if not found:
                continue
            
            # Apply swap
            masks[a] ^= (1 << b)
            masks[b] ^= (1 << a)
            masks[c] |= (1 << d)
            masks[d] |= (1 << c)
            
            edge_set.discard((a, b))
            edge_set.add((c, d))
            edge_list.remove((a, b))
            edge_list.append((c, d))
            
            new_score = compute_score(masks)
            
            if new_score <= best_score:
                best_score = new_score
                best_masks = masks[:]
            else:
                # Revert
                masks[c] ^= (1 << d)
                masks[d] ^= (1 << c)
                masks[a] |= (1 << b)
                masks[b] |= (1 << a)
                edge_set.discard((c, d))
                edge_set.add((a, b))
                edge_list.remove((c, d))
                edge_list.append((a, b))
        
        return best_masks, best_score
    
    # Start with K_{8,8} + 1
    edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    masks = make_bitmasks(edges)
    best_score = compute_score(masks)
    best_masks = masks[:]
    
    # Local search from seed
    best_masks, best_score = local_search(masks[:], deadline)
    
    # Multiple restarts with perturbations
    while time.time() < deadline - 10.0:
        start_masks = best_masks[:]
        # Perturb with random swaps
        edge_set = set()
        for u in range(n):
            m = start_masks[u]
            for v in range(u + 1, n):
                if (m >> v) & 1:
                    edge_set.add((u, v))
        
        for _ in range(random.randint(5, 20)):
            cur_edges = list(edge_set)
            if len(cur_edges) < 2:
                break
            a, b = random.choice(cur_edges)
            found = False
            for _ in range(20):
                c = random.randint(0, n - 1)
                d = random.randint(0, n - 1)
                if c == d or c > d:
                    continue
                if (c, d) in edge_set:
                    continue
                if (c, d) == (a, b):
                    continue
                found = True
                break
            if not found:
                continue
            
            start_masks[a] ^= (1 << b)
            start_masks[b] ^= (1 << a)
            start_masks[c] |= (1 << d)
            start_masks[d] |= (1 << c)
            edge_set.discard((a, b))
            edge_set.add((c, d))
        
        new_masks, new_score = local_search(start_masks[:], deadline, max_iters=80000)
        if new_score < best_score:
            best_score = new_score
            best_masks = new_masks
    
    final_edges = edges_from_masks(best_masks)
    assert len(final_edges) == target_edges
    
    return {'n': n, 'edges': final_edges}
