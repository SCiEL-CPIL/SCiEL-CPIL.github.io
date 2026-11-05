import time
import sys
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # 5 seconds margin for 60s limit

    N = 256
    # Precompute perfect squares up to 2*N
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
    
    # Identify vertices that cannot be in the set (2*v is square)
    forbidden = set()
    for v in range(1, N + 1):
        if 2 * v in squares:
            forbidden.add(v)
            
    valid_vertices = [v for v in range(1, N + 1) if v not in forbidden]
    M = len(valid_vertices)
    vertex_map = valid_vertices # i -> v
    
    # Build adjacency list for valid vertices
    adj = [set() for _ in range(M)]
    
    for i in range(M):
        vi = vertex_map[i]
        for j in range(i + 1, M):
            vj = vertex_map[j]
            if vi + vj in squares:
                adj[i].add(j)
                adj[j].add(i)
                
    # Initial lower bound using greedy
    best_size = 0
    best_set_indices = set()
    
    def greedy_random(seed):
        random.seed(seed)
        order = list(range(M))
        random.shuffle(order)
        selected = set()
        for v in order:
            conflict = False
            for u in adj[v]:
                if u in selected:
                    conflict = True
                    break
            if not conflict:
                selected.add(v)
        return selected

    # Run several greedy restarts
    for i in range(100):
        s = greedy_random(i)
        if len(s) > best_size:
            best_size = len(s)
            best_set_indices = s
            
    def greedy_deterministic():
        verts = sorted(range(M), key=lambda x: len(adj[x]))
        selected = set()
        for v in verts:
            if v in selected: continue
            conflict = False
            for u in adj[v]:
                if u in selected:
                    conflict = True
                    break
            if not conflict:
                selected.add(v)
        return selected
        
    s = greedy_deterministic()
    if len(s) > best_size:
        best_size = len(s)
        best_set_indices = s

    # Local Search Improvement
    def local_search(current_set):
        nonlocal best_size, best_set_indices
        improved = True
        current_set = set(current_set)
        while improved and (time.time() - start_time < DEADLINE):
            improved = False
            # Try removing each vertex and filling
            for v in list(current_set):
                new_set = set(current_set)
                new_set.remove(v)
                
                # Try to add vertices from complement
                # Candidates are those not in new_set and not adjacent to any in new_set
                candidates = []
                for u in range(M):
                    if u in new_set:
                        continue
                    # Check if u conflicts with any in new_set
                    conflict = False
                    for w in new_set:
                        if w in adj[u]:
                            conflict = True
                            break
                    if not conflict:
                        candidates.append(u)
                        
                # Greedily add candidates
                # Sort candidates by degree? Or just random?
                # Let's try adding all that don't conflict with each other
                added = set()
                for u in candidates:
                    if u in added:
                        continue
                    # Check conflict with already added in this step
                    conflict_added = False
                    for w in added:
                        if w in adj[u]:
                            conflict_added = True
                            break
                    if not conflict_added:
                        # Check conflict with new_set (already checked in candidates, but new_set hasn't changed)
                        # So we can just add
                        new_set.add(u)
                        added.add(u)
                        
                if len(new_set) > len(current_set):
                    current_set = new_set
                    improved = True
                    # Update global best if better
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set_indices = set(current_set)
        return current_set

    # Run local search on the best found so far
    # We can run it on a few random restarts too
    for i in range(10):
        s = greedy_random(1000 + i)
        s = local_search(s)
        if len(s) > best_size:
            best_size = len(s)
            best_set_indices = s

    # B&B with Coloring Bound
    sys.setrecursionlimit(10000)
    
    def greedy_coloring(P):
        if not P:
            return 0
        
        degs = {v: 0 for v in P}
        for v in P:
            for u in adj[v]:
                if u in P:
                    degs[v] += 1
        
        ordered_P = sorted(P, key=lambda x: degs[x], reverse=True)
        
        colors = {}
        num_colors = 0
        for v in ordered_P:
            used_colors = set()
            for u in adj[v]:
                if u in colors:
                    used_colors.add(colors[u])
            
            c = 0
            while c in used_colors:
                c += 1
            colors[v] = c
            if c >= num_colors:
                num_colors = c + 1
                
        return num_colors

    def solve(P, current_size, current_set):
        nonlocal best_size, best_set_indices
        
        if time.time() - start_time > DEADLINE:
            return
            
        if not P:
            if current_size > best_size:
                best_size = current_size
                best_set_indices = set(current_set)
            return
            
        ub = greedy_coloring(P)
        if current_size + ub <= best_size:
            return
            
        max_deg = -1
        v_choice = None
        for v in P:
            deg = 0
            for u in adj[v]:
                if u in P:
                    deg += 1
            if deg > max_deg:
                max_deg = deg
                v_choice = v
                if deg == len(P) - 1:
                    break
                    
        if v_choice is None:
            return
            
        new_P = set(P)
        new_P.remove(v_choice)
        for u in adj[v_choice]:
            if u in new_P:
                new_P.remove(u)
        
        new_set = set(current_set)
        new_set.add(v_choice)
        
        solve(new_P, current_size + 1, new_set)
        
        if time.time() - start_time > DEADLINE:
            return
            
        new_P2 = set(P)
        new_P2.remove(v_choice)
        
        solve(new_P2, current_size, current_set)

    initial_P = set(range(M))
    initial_set = set()
    
    try:
        solve(initial_P, 0, initial_set)
    except RecursionError:
        pass

    result_indices = best_set_indices
    result_values = [vertex_map[i] for i in result_indices]
    result_values.sort()
    
    return result_values

if __name__ == "__main__":
    res = construct()
    print(f"Found set size: {len(res)}")
