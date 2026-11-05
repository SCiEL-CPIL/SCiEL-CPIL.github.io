import time
import random
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Checks if the hypergraph is 2-colorable.
    Returns True if 2-colorable, False otherwise.
    """
    colors = [-1] * n
    edge_list = [set(e) for e in edges]
    
    # Sort vertices by degree to improve backtracking? 
    # For simplicity, we stick to index order but propagate.
    
    def propagate():
        changed = True
        while changed:
            changed = False
            for e in edge_list:
                c0 = 0
                c1 = 0
                unassigned = []
                for v in e:
                    if colors[v] == 0:
                        c0 += 1
                    elif colors[v] == 1:
                        c1 += 1
                    else:
                        unassigned.append(v)
                
                if c0 == 4 and len(unassigned) == 1:
                    v = unassigned[0]
                    if colors[v] == 1:
                        return False
                    colors[v] = 1
                    changed = True
                elif c1 == 4 and len(unassigned) == 1:
                    v = unassigned[0]
                    if colors[v] == 0:
                        return False
                    colors[v] = 0
                    changed = True
                elif c0 == 5 or c1 == 5:
                    return False
        return True

    def backtrack(idx):
        if idx == n:
            # All assigned. Check validity
            for e in edge_list:
                c0 = 0
                c1 = 0
                for v in e:
                    if colors[v] == 0:
                        c0 += 1
                    else:
                        c1 += 1
                if c0 == 0 or c1 == 0:
                    return False
            return True
            
        # Find next unassigned
        v = -1
        for i in range(idx, n):
            if colors[i] == -1:
                v = i
                break
        
        if v == -1:
            return backtrack(n)
            
        # Try 0
        old = colors[v]
        colors[v] = 0
        if propagate():
            if backtrack(idx + 1):
                return True
        colors[v] = old
        
        # Try 1
        colors[v] = 1
        if propagate():
            if backtrack(idx + 1):
                return True
        colors[v] = old
        
        return False

    return backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    best_edges = None
    best_n = 9
    best_count = 126
    
    for n in range(9, 14): # Limit n to keep search fast
        if time.time() > deadline:
            break
        
        all_edges = list(combinations(range(n), 5))
        
        # Try a few random instances
        for _ in range(20):
            if time.time() > deadline:
                break
            
            # Pick a random subset of edges
            # Start with a high density to likely be non-2-colorable
            m = len(all_edges) // 2
            random.shuffle(all_edges)
            current_edges = all_edges[:m]
            
            if not is_2_colorable(n, current_edges):
                # Prune
                pruned = True
                while pruned and time.time() < deadline:
                    pruned = False
                    # Try removing edges
                    # Iterate through a copy to allow modification
                    edges_copy = list(current_edges)
                    random.shuffle(edges_copy)
                    for i in range(len(edges_copy)):
                        if time.time() > deadline:
                            break
                        # Try removing edges_copy[i]
                        # We need to check if the rest is still non-2-colorable
                        candidate = edges_copy[:i] + edges_copy[i+1:]
                        if not is_2_colorable(n, candidate):
                            current_edges = candidate
                            pruned = True
                            break # Restart with new list
                            
                if len(current_edges) < best_count:
                    best_count = len(current_edges)
                    best_n = n
                    best_edges = [list(e) for e in current_edges]
    
    if best_edges is None:
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    else:
        return {'n': best_n, 'edges': best_edges}
