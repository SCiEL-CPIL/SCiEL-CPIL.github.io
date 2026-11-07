import time
import random

def construct():
    trees = instance_trees(20, 611)
    n = 20
    start_time = time.time()
    timeout = 110
    
    best_assignment = [None] * n
    best_edges_count = 0
    
    # Order: T20 (19), T19 (18), ..., T2 (1), T1 (0)
    # We process indices 19 down to 1. Index 0 is T1.
    
    def bt(idx, used_edges, assignment):
        nonlocal best_assignment, best_edges_count
        
        if time.time() - start_time > timeout:
            return False
            
        if idx == 1:
            # Only T1 left (index 0)
            assignment[0] = None
            total = sum(len(trees[i]) for i in range(1, n))
            if total > best_edges_count:
                best_edges_count = total
                best_assignment = assignment[:]
            return True
        
        # Current tree index is idx
        # We are processing trees in order: 19, 18, ..., 1
        # So idx represents the tree index directly?
        # Let's say we call bt(19, ...), then bt(18, ...), etc.
        # Base case: idx == 0 (T1)
        
        tree_edges = trees[idx]
        k = idx + 1
        
        # Try to embed
        for _ in range(50):
            if time.time() - start_time > timeout:
                return False
                
            verts = random.sample(range(n), k)
            perm = list(range(k))
            random.shuffle(perm)
            mapping = {i: verts[perm[i]] for i in range(k)}
            
            valid = True
            new_used = set(used_edges)
            for u, v in tree_edges:
                hu = mapping[u]
                hv = mapping[v]
                edge = (min(hu, hv), max(hu, hv))
                if edge in new_used:
                    valid = False
                    break
                new_used.add(edge)
            
            if valid:
                assignment[idx] = [mapping[i] for i in range(k)]
                if bt(idx - 1, new_used, assignment):
                    return True
                assignment[idx] = None
        
        return False
    
    # Run multiple trials
    for t in range(20):
        if time.time() - start_time > timeout:
            break
        random.seed(t)
        bt(19, set(), [None]*n)
        
    if best_edges_count > 0:
        result = []
        for i in range(n):
            if best_assignment[i] is None:
                result.append(None)
            else:
                result.append([int(x) for x in best_assignment[i]])
        return result
        
    return [None] * n
