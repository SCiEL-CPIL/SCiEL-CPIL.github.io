import random
import time
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Returns True if the hypergraph is 2-colorable, False otherwise.
    Uses backtracking with unit propagation.
    """
    if not edges:
        return True
        
    # Map vertex to list of edge indices
    vertex_to_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_to_edges[v].append(i)
            
    colors = [-1] * n
    
    def backtrack(assigned_count):
        if assigned_count == n:
            return True
            
        # 1. Unit Propagation / Conflict Detection
        forced_var = -1
        forced_val = -1
        
        for e in edges:
            unassigned = []
            zeros = 0
            ones = 0
            for v in e:
                if colors[v] == -1:
                    unassigned.append(v)
                elif colors[v] == 0:
                    zeros += 1
                else:
                    ones += 1
            
            if not unassigned:
                if zeros == 5 or ones == 5:
                    return False
            elif len(unassigned) == 1:
                v = unassigned[0]
                if zeros == 4:
                    if forced_var != -1:
                        if forced_var == v:
                            if forced_val != 1:
                                return False
                        else:
                            return False
                    forced_var = v
                    forced_val = 1
                elif ones == 4:
                    if forced_var != -1:
                        if forced_var == v:
                            if forced_val != 0:
                                return False
                        else:
                            return False
                    forced_var = v
                    forced_val = 0
        
        if forced_var != -1:
            colors[forced_var] = forced_val
            if backtrack(assigned_count + 1):
                return True
            colors[forced_var] = -1
            return False
            
        # 2. Variable Selection
        var_to_assign = -1
        max_degree = -1
        for v in range(n):
            if colors[v] == -1:
                deg = len(vertex_to_edges[v])
                if deg > max_degree:
                    max_degree = deg
                    var_to_assign = v
                    
        if var_to_assign == -1:
            return True
            
        # 3. Branch
        colors[var_to_assign] = 0
        valid = True
        for i in vertex_to_edges[var_to_assign]:
            e = edges[i]
            all_zero = True
            for u in e:
                if u != var_to_assign:
                    if colors[u] != 0:
                        all_zero = False
                        break
            if all_zero:
                valid = False
                break
                
        if valid:
            if backtrack(assigned_count + 1):
                return True
                
        colors[var_to_assign] = 1
        valid = True
        for i in vertex_to_edges[var_to_assign]:
            e = edges[i]
            all_one = True
            for u in e:
                if u != var_to_assign:
                    if colors[u] != 1:
                        all_one = False
                        break
            if all_one:
                valid = False
                break
                
        if valid:
            if backtrack(assigned_count + 1):
                return True
                
        colors[var_to_assign] = -1
        return False
        
    return backtrack(0)

def construct():
    import random
    import time
    
    start_time = time.time()
    timeout = 110
    
    best_edges = None
    best_n = 0
    best_score = float('inf')
    
    for n in range(15, 27):
        if time.time() - start_time > timeout:
            break
            
        all_edges = list(combinations(range(n), 5))
        total_possible = len(all_edges)
        
        found_for_n = False
        # We want to find a small core.
        # Start with small m? No, small m is likely 2-colorable.
        # Start with m that is likely non-2-colorable, then minimize.
        # Threshold is around 50-60 for n=26.
        
        for m in range(45, min(total_possible, 100) + 1):
            if time.time() - start_time > timeout:
                break
            if found_for_n:
                break
                
            for _ in range(3):
                if time.time() - start_time > timeout:
                    break
                    
                edges = [list(e) for e in random.sample(all_edges, m)]
                
                if not is_2_colorable(n, edges):
                    found_for_n = True
                    current_edges = edges[:]
                    
                    # Minimize
                    changed = True
                    passes = 0
                    while changed and passes < 5 and time.time() - start_time < timeout:
                        changed = False
                        passes += 1
                        random.shuffle(current_edges)
                        for i in range(len(current_edges) - 1, -1, -1):
                            if time.time() - start_time > timeout:
                                break
                            
                            temp_edges = current_edges[:i] + current_edges[i+1:]
                            
                            if not is_2_colorable(n, temp_edges):
                                current_edges.pop(i)
                                changed = True
                                break
                                
                    if len(current_edges) < best_score:
                        best_score = len(current_edges)
                        best_edges = current_edges
                        best_n = n
                        
    if best_edges is None:
        n = 26
        all_edges = list(combinations(range(n), 5))
        edges = [list(e) for e in random.sample(all_edges, 100)]
        while is_2_colorable(n, edges) and len(edges) < len(all_edges):
            edges.append(list(random.choice(all_edges)))
        return {'n': n, 'edges': edges}
        
    return {'n': best_n, 'edges': best_edges}
