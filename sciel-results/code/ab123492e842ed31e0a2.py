"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
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
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    deadline = time.time() + 145
    trees = instance_trees(20, 611)
    
    # Precompute tree data
    tree_adjs = []
    tree_orders = []
    for idx in range(20):
        k = idx + 1
        edges = trees[idx]
        adj = [[] for _ in range(k)]
        deg = [0] * k
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
        tree_adjs.append(adj)
        # Order vertices by degree descending for better backtracking
        verts = list(range(k))
        verts.sort(key=lambda v: -deg[v])
        tree_orders.append(verts)

    # All edges in K20
    all_edges = set()
    for i in range(20):
        for j in range(i+1, 20):
            all_edges.add((i, j))
    
    # Order of processing: largest trees first
    process_order = list(range(19, -1, -1))
    
    best_result = [None] * 20
    best_edges = 0
    
    for trial in range(5000):
        if time.time() > deadline:
            break
            
        rng = Random(trial)
        used_edges = set()
        result = [None] * 20
        success = True
        total_edges = 0
        
        for idx in process_order:
            k = idx + 1
            if k == 1:
                result[0] = [0]
                continue
            
            adj = tree_adjs[idx]
            v_order = tree_orders[idx]
            assignment = [-1] * k
            
            def backtrack(pos):
                if time.time() > deadline:
                    return False
                if pos == k:
                    return True
                
                tv = v_order[pos]
                
                # Try host vertices 0..19 in random order
                hosts = list(range(20))
                rng.shuffle(hosts)
                
                for h in hosts:
                    # Check if h is used by other tree vertices
                    if h in assignment:
                        continue
                    
                    # Check edges to already assigned neighbors
                    valid = True
                    for nb in adj[tv]:
                        if assignment[nb] != -1:
                            e = (min(assignment[nb], h), max(assignment[nb], h))
                            if e in used_edges:
                                valid = False
                                break
                    
                    if valid:
                        assignment[tv] = h
                        if backtrack(pos + 1):
                            return True
                        assignment[tv] = -1
                        
                return False
            
            if backtrack(0):
                result[idx] = assignment[:]
                for u, v in trees[idx]:
                    hu, hv = assignment[u], assignment[v]
                    e = (min(hu, hv), max(hu, hv))
                    used_edges.add(e)
                total_edges += k - 1
            else:
                success = False
                break
                
        if success and total_edges > best_edges:
            best_edges = total_edges
            best_result = result[:]
            if best_edges == 190:
                break # Optimal found
            
    if best_result[0] is None:
        best_result[0] = [0]
        
    return best_result
# EVOLVE-BLOCK-END
