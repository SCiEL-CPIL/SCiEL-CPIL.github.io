import time
import random
from collections import deque

def instance_trees(n=20, seed=611):
    rng = random.Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    trees = instance_trees(20, 611)
    
    # Precompute tree properties
    tree_info = []
    for i in range(20):
        edges = trees[i]
        k = i + 1
        adj = [[] for _ in range(k)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        tree_info.append({
            'k': k,
            'edges': edges,
            'adj': adj,
            'num_edges': len(edges)
        })
    
    # Sort by size descending
    order = sorted(range(20), key=lambda i: -tree_info[i]['k'])
    
    def try_pack_greedy(attempt_seed):
        local_rng = random.Random(attempt_seed)
        used_edges = set()
        mappings = [None] * 20
        
        # We pack in the pre-sorted order (largest first)
        # But we can also randomize the order slightly to escape local minima?
        # Let's stick to largest first for determinism in the greedy choice, 
        # but randomize the vertex mapping.
        
        for t_idx in order:
            info = tree_info[t_idx]
            k = info['k']
            adj_t = info['adj']
            
            if k == 1:
                # Tree with 1 vertex, 0 edges. Can map to any vertex.
                # But we need distinct vertices for the tree itself? 
                # "list of k distinct int host vertices"
                # So for k=1, we need 1 vertex.
                # It doesn't use edges, so it doesn't conflict with edge usage.
                # But it uses a vertex. Does vertex usage matter?
                # "Embedded trees must be edge-disjoint in K20."
                # The constraint is only on edges. Vertices can be shared?
                # "list of k distinct int host vertices in 0..19"
                # This implies the mapping is injective for that tree.
                # It does NOT say the mappings of different trees must be vertex-disjoint.
                # So vertices CAN be shared across different trees.
                # Only edges must be disjoint.
                
                # For k=1, we can pick any vertex, say 0.
                mappings[t_idx] = [0]
                continue
            
            # For k > 1, we need to map k distinct vertices.
            # We use a randomized BFS/DFS embedding.
            
            # Pick a random root in the tree
            root_t = local_rng.randint(0, k - 1)
            
            # Try to find a mapping
            # We'll use a simple backtracking for the single tree embedding
            # to ensure we don't get stuck if the greedy path fails.
            # However, for small trees in K20, greedy usually works.
            # Let's try a few greedy attempts for each tree.
            
            mapping_found = False
            for _ in range(50): # 50 random tries for this tree
                mapping = [-1] * k
                used_hosts = set()
                
                # Initialize BFS queue
                queue = deque([root_t])
                mapping[root_t] = local_rng.randint(0, 19)
                used_hosts.add(mapping[root_t])
                
                success = True
                while queue:
                    u = queue.popleft()
                    hu = mapping[u]
                    for v in adj_t[u]:
                        if mapping[v] != -1:
                            continue
                        # Find a host for v
                        # Must be distinct from used_hosts
                        # And edge (hu, hv) must be unused
                        candidates = []
                        for hv in range(20):
                            if hv in used_hosts:
                                continue
                            e = (min(hu, hv), max(hu, hv))
                            if e in used_edges:
                                continue
                            candidates.append(hv)
                        
                        if not candidates:
                            success = False
                            break
                        
                        hv = local_rng.choice(candidates)
                        mapping[v] = hv
                        used_hosts.add(hv)
                        queue.append(v)
                
                if success:
                    # Verify all edges are valid and unused
                    valid = True
                    for u, v in info['edges']:
                        hu = mapping[u]
                        hv = mapping[v]
                        e = (min(hu, hv), max(hu, hv))
                        if e in used_edges:
                            valid = False
                            break
                    if valid:
                        mappings[t_idx] = mapping
                        # Mark edges as used
                        for u, v in info['edges']:
                            hu = mapping[u]
                            hv = mapping[v]
                            e = (min(hu, hv), max(hu, hv))
                            used_edges.add(e)
                        mapping_found = True
                        break
            
            if not mapping_found:
                return None, 0 # Failed to pack this tree
        
        # Calculate score
        score = 0
        for i in range(20):
            if mappings[i] is not None:
                score += tree_info[i]['num_edges']
        
        return mappings, score

    best_mappings = [None] * 20
    best_score = 0
    
    attempt = 0
    while time.time() < deadline:
        mappings, score = try_pack_greedy(attempt)
        if mappings is not None and score > best_score:
            best_score = score
            best_mappings = mappings[:]
            if best_score == 190:
                break
        attempt += 1
    
    # Ensure proper types
    result = []
    for i in range(20):
        if best_mappings[i] is not None:
            result.append([int(v) for v in best_mappings[i]])
        else:
            result.append(None)
    
    return result

if __name__ == "__main__":
    start = time.time()
    result = construct()
    elapsed = time.time() - start
    print(f"Time: {elapsed:.2f}s")
    
    # Verify
    trees = instance_trees(20, 611)
    used_edges = set()
    total_edges = 0
    valid = True
    for i in range(20):
        if result[i] is None:
            continue
        mapping = result[i]
        k = i + 1
        if len(mapping) != k:
            print(f"Tree {i}: wrong length {len(mapping)} vs {k}")
            valid = False
            break
        if len(set(mapping)) != k:
            print(f"Tree {i}: duplicate host vertices")
            valid = False
            break
        for v in mapping:
            if not isinstance(v, int) or v < 0 or v > 19:
                print(f"Tree {i}: invalid vertex {v}")
                valid = False
                break
        if not valid:
            break
        for u, v in trees[i]:
            hu = mapping[u]
            hv = mapping[v]
            e = (min(hu, hv), max(hu, hv))
            if e in used_edges:
                print(f"Tree {i}: edge conflict {e}")
                valid = False
                break
            used_edges.add(e)
            total_edges += 1
        if not valid:
            break
    
    print(f"Valid: {valid}")
    print(f"Total edges: {total_edges}")
    if valid:
        print(f"Score: {total_edges}")
    else:
        print("INVALID RESULT")
