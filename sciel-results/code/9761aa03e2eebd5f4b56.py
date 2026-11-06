import time
import random
from itertools import combinations

def is_2_colorable(n, edges):
    if not edges:
        return True
    
    colors = [None] * n
    
    def propagate():
        changed = True
        while changed:
            changed = False
            for e in edges:
                assigned_vals = []
                unassigned_verts = []
                for v in e:
                    if colors[v] is not None:
                        assigned_vals.append(colors[v])
                    else:
                        unassigned_verts.append(v)
                
                if not unassigned_verts:
                    if len(assigned_vals) > 0:
                        if all(c == 0 for c in assigned_vals) or all(c == 1 for c in assigned_vals):
                            return False
                else:
                    if assigned_vals:
                        if all(c == 0 for c in assigned_vals):
                            for v in unassigned_verts:
                                if colors[v] is not None:
                                    if colors[v] != 1:
                                        return False
                                else:
                                    colors[v] = 1
                                    changed = True
                        elif all(c == 1 for c in assigned_vals):
                            for v in unassigned_verts:
                                if colors[v] is not None:
                                    if colors[v] != 0:
                                        return False
                                else:
                                    colors[v] = 0
                                    changed = True
        return True

    def backtrack(idx):
        if idx == n:
            return True
        
        colors[idx] = 0
        if propagate():
            if backtrack(idx + 1):
                return True
        
        colors[idx] = 1
        if propagate():
            if backtrack(idx + 1):
                return True
        
        colors[idx] = None
        return False

    return backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_edges = None
    best_n = 0
    best_count = float('inf')
    
    # Try different n values
    for n in [11, 12, 13, 14, 15]:
        if time.time() > deadline:
            break
        
        attempts = 0
        while time.time() < deadline and attempts < 20:
            attempts += 1
            
            # Generate random edges
            all_combos = list(combinations(range(n), 5))
            random.shuffle(all_combos)
            
            # Target edge count for initial graph
            # Higher n might need more edges to be non-2-colorable?
            # But we want a small core.
            # Let's try a range.
            k = random.randint(int(n * 5), int(n * 10))
            edges = [list(e) for e in all_combos[:k]]
            
            # Ensure distinct
            edges = [list(e) for e in set([tuple(e) for e in edges])]
            
            if not is_2_colorable(n, edges):
                # Minimize
                edge_tuples = [tuple(e) for e in edges]
                
                changed = True
                while changed and time.time() < deadline:
                    changed = False
                    # Shuffle to avoid bias?
                    # random.shuffle(edge_tuples) -- might help find smaller cores
                    for i in range(len(edge_tuples)):
                        test_edges = edge_tuples[:i] + edge_tuples[i+1:]
                        if not is_2_colorable(n, test_edges):
                            edge_tuples = test_edges
                            changed = True
                            break
                
                if len(edge_tuples) < best_count:
                    best_edges = [list(e) for e in edge_tuples]
                    best_n = n
                    best_count = len(edge_tuples)
                    
                if best_count <= 30:
                    return {'n': best_n, 'edges': best_edges}
                    
    if best_edges is None:
        # Fallback
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
        
    return {'n': best_n, 'edges': best_edges}
