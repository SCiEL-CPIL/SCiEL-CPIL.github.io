"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import defaultdict
from itertools import combinations

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    trees = instance_trees(20, 611)
    n = 20
    
    # Edge index mapping
    edge_to_idx = {}
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            edge_to_idx[(i, j)] = len(edge_list)
            edge_list.append((i, j))
    num_edges = len(edge_list)  # 190
    
    # Precompute tree data
    tree_data = []
    for t_idx in range(20):
        tree_edges = trees[t_idx]
        k = t_idx + 1
        if k == 1:
            tree_data.append(None)
            continue
        
        tree_verts = set()
        for u, v in tree_edges:
            tree_verts.add(u)
            tree_verts.add(v)
        tree_verts = sorted(tree_verts)
        
        tree_adj = {v: [] for v in tree_verts}
        for u, v in tree_edges:
            tree_adj[u].append(v)
            tree_adj[v].append(u)
            
        ordered_tv = sorted(tree_verts, key=lambda v: (-len(tree_adj[v]), v))
        
        tree_data.append({
            'edges': tree_edges,
            'verts': tree_verts,
            'adj': tree_adj,
            'ordered': ordered_tv,
            'k': k
        })
    
    # Sort tree indices: largest trees first
    tree_order = list(range(20))
    tree_order.sort(key=lambda idx: (-(len(trees[idx])), -idx))
    
    def embed_tree(t_idx, available_mask, rng, order_variant=0):
        """
        Attempts to embed tree t_idx using available edges (bitmask).
        Returns a list of host vertices for tree vertices 0..k-1, or None.
        """
        if t_idx == 0:
            return [rng.randint(0, n - 1)]
        
        data = tree_data[t_idx]
        k = data['k']
        tree_adj = data['adj']
        ordered_tv = data['ordered']
        tree_edges = data['edges']
        
        # Build adjacency from available mask
        host_adj_mask = [0] * n
        for ei in range(num_edges):
            if (available_mask >> ei) & 1:
                a, b = edge_list[ei]
                host_adj_mask[a] |= (1 << b)
                host_adj_mask[b] |= (1 << a)
        
        all_mask = (1 << n) - 1
        host_map = {}
        used_mask = 0
        
        # Precompute degrees for sorting candidates
        deg = [bin(host_adj_mask[h]).count('1') for h in range(n)]
        
        # Try different orderings based on variant
        if order_variant == 1:
            # Order by degree ascending (least constrained first)
            ordered_tv = sorted(ordered_tv, key=lambda v: (len(tree_adj[v]), v))
        elif order_variant == 2:
            # Randomized ordering
            ordered_tv = ordered_tv[:]
            rng.shuffle(ordered_tv)
        
        def backtrack(pos):
            nonlocal used_mask
            if pos == len(ordered_tv):
                for u, v in tree_edges:
                    hu, hv = host_map[u], host_map[v]
                    if not (host_adj_mask[hu] >> hv) & 1:
                        return False
                return True
            
            tv = ordered_tv[pos]
            tv_neighbors = tree_adj[tv]
            
            assigned_neighbors = [(nb, host_map[nb]) for nb in tv_neighbors if nb in host_map]
            
            if assigned_neighbors:
                cand_mask = host_adj_mask[assigned_neighbors[0][1]]
                for _, hn in assigned_neighbors[1:]:
                    cand_mask &= host_adj_mask[hn]
            else:
                cand_mask = all_mask
                
            cand_mask &= ~used_mask
            
            if cand_mask == 0:
                return False
            
            candidates = []
            cm = cand_mask
            while cm:
                h = (cm & (-cm)).bit_length() - 1
                cm &= cm - 1
                candidates.append(h)
            
            # Sort by degree, with tie-breaking by index for determinism
            candidates.sort(key=lambda h: (-deg[h], h))
            
            # Limit candidates for speed, but try more if we have time
            max_cands = 8
            if len(candidates) > max_cands:
                # Keep top candidates, shuffle the rest
                candidates = candidates[:max_cands]
                rng.shuffle(candidates[4:])
            
            for h in candidates:
                host_map[tv] = h
                new_used = used_mask | (1 << h)
                
                valid = True
                # Forward checking: ensure all unassigned neighbors can still be placed
                for w in tv_neighbors:
                    if w not in host_map:
                        other_assigned = [host_map[nb] for nb in tree_adj[w] if nb in host_map and nb != tv]
                        
                        if other_assigned:
                            inter = host_adj_mask[h]
                            for oh in other_assigned:
                                inter &= host_adj_mask[oh]
                        else:
                            inter = host_adj_mask[h]
                        
                        inter &= ~new_used
                        if inter == 0:
                            valid = False
                            break
                            
                if valid:
                    used_mask = new_used
                    if backtrack(pos + 1):
                        return True
                    used_mask = new_used & ~(1 << h)
                    
                del host_map[tv]
                
            return False
            
        if backtrack(0):
            return [host_map[i] for i in range(k)]
        return None

    best_result = None
    best_edges = 0
    
    # Phase 1: Randomized greedy with backtracking
    # Use multiple order variants for diversity
    for attempt in range(3000):
        if time.time() > deadline - 10:
            break
            
        rng = Random(attempt * 104729 + 1)
        order_variant = attempt % 3  # Cycle through different orderings
        
        # For larger attempts, try different tree embeddings
        if attempt % 10 == 0:
            # Try embedding in natural order for some attempts
            local_order = list(range(20))
        else:
            local_order = tree_order[:]
        
        available_mask = (1 << num_edges) - 1
        result = [None] * 20
        success = True
        edge_count = 0
        
        for t_idx in local_order:
            if time.time() > deadline - 10:
                success = False
                break
                
            k = t_idx + 1
            if k == 1:
                result[t_idx] = [rng.randint(0, n - 1)]
                continue
                
            embedded = embed_tree(t_idx, available_mask, rng, order_variant)
            
            if embedded is None:
                success = False
                break
                
            result[t_idx] = embedded
            edge_count += len(trees[t_idx])
            for u, v in trees[t_idx]:
                hu, hv = embedded[u], embedded[v]
                ei = edge_to_idx[(min(hu, hv), max(hu, hv))]
                available_mask &= ~(1 << ei)
                
        if success and edge_count == 190:
            best_result = result
            break
        elif edge_count > best_edges:
            best_edges = edge_count
            best_result = result

    # Phase 2: If we have a perfect solution, verify and return
    if best_result and best_edges == 190:
        final = []
        for entry in best_result:
            if entry is None:
                final.append(None)
            else:
                final.append([int(x) for x in entry])
        return final
    
    # Phase 3: Local search / hill climbing to improve partial solution
    if best_result is not None:
        # Try to fill in missing trees or rearrange
        for attempt in range(1500):
            if time.time() > deadline - 5:
                break
            rng = Random(attempt * 777777 + 3)
            
            # Copy current result
            curr = [x[:] if x is not None else None for x in best_result]
            curr_edges = best_edges
            
            # Find which trees are missing
            missing = [i for i in range(20) if curr[i] is None]
            if not missing:
                continue
            
            # Rebuild available mask from current embeddings
            used_edge_set = set()
            for t_idx in range(20):
                if curr[t_idx] is None:
                    continue
                for u, v in trees[t_idx]:
                    hu, hv = curr[t_idx][u], curr[t_idx][v]
                    used_edge_set.add((min(hu, hv), max(hu, hv)))
            
            available_mask = (1 << num_edges) - 1
            for e in used_edge_set:
                ei = edge_to_idx[e]
                available_mask &= ~(1 << ei)
            
            # Try to embed missing trees, using different order variants
            for t_idx in missing:
                k = t_idx + 1
                if k == 1:
                    curr[t_idx] = [rng.randint(0, n - 1)]
                    continue
                # Try multiple order variants
                for ov in range(3):
                    embedded = embed_tree(t_idx, available_mask, rng, ov)
                    if embedded is not None:
                        curr[t_idx] = embedded
                        curr_edges += len(trees[t_idx])
                        for u, v in trees[t_idx]:
                            hu, hv = embedded[u], embedded[v]
                            ei = edge_to_idx[(min(hu, hv), max(hu, hv))]
                            available_mask &= ~(1 << ei)
                        break
            
            if curr_edges > best_edges:
                best_edges = curr_edges
                best_result = curr
                if best_edges == 190:
                    break
    
    # Final fallback
    if best_result is None:
        # Last resort: just embed trees greedily without worrying about perfect packing
        rng = Random(42)
        available_mask = (1 << num_edges) - 1
        result = [None] * 20
        for t_idx in range(20):
            k = t_idx + 1
            if k == 1:
                result[t_idx] = [rng.randint(0, n - 1)]
                continue
            embedded = embed_tree(t_idx, available_mask, rng)
            if embedded is not None:
                result[t_idx] = embedded
                for u, v in trees[t_idx]:
                    hu, hv = embedded[u], embedded[v]
                    ei = edge_to_idx[(min(hu, hv), max(hu, hv))]
                    available_mask &= ~(1 << ei)
        best_result = result
    
    final = []
    for entry in best_result:
        if entry is None:
            final.append(None)
        else:
            final.append([int(x) for x in entry])
    return final

# EVOLVE-BLOCK-END
