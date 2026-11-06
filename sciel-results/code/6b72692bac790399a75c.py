import math
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    n = 256
    max_sum = 2 * n  # 512
    
    # Perfect squares in [2, 512]
    squares = set()
    i = 1
    while i * i <= max_sum:
        if i * i >= 2:
            squares.add(i * i)
        i += 1
    
    # Diagonal constraint: 2a not in squares
    candidates = [a for a in range(1, n + 1) if 2 * a not in squares]
    cand_set = set(candidates)
    num_cand = len(candidates)
    
    # Build conflict graph: conflict[a] = set of b in candidates where a+b is a square
    conflict = {a: set() for a in candidates}
    for idx, a in enumerate(candidates):
        for b in candidates[idx+1:]:
            if a + b in squares:
                conflict[a].add(b)
                conflict[b].add(a)
    
    # Greedy coloring for upper bound on independent set
    def greedy_color(nodes, adj):
        """Color nodes greedily, return list of (color, nodes_with_that_color)"""
        uncolored = set(nodes)
        colors = []
        while uncolored:
            # Pick a node, color it, remove it and its neighbors from uncolored
            # Use a simple strategy: pick node with smallest degree in uncolored
            best_node = None
            best_deg = float('inf')
            for node in uncolored:
                deg = len(adj[node] & uncolored)
                if deg < best_deg:
                    best_deg = deg
                    best_node = node
            color_nodes = {best_node}
            uncolored.discard(best_node)
            # Remove neighbors
            for nb in adj[best_node] & uncolored:
                uncolored.discard(nb)
            colors.append(color_nodes)
        return colors
    
    # Upper bound: number of colors needed
    def upper_bound(nodes, adj):
        if not nodes:
            return 0
        colors = greedy_color(nodes, adj)
        return len(colors)
    
    # Branch and bound for maximum independent set
    def branch_and_bound(nodes, current_set, best_size):
        if time.time() - start_time > deadline - 2.0:
            return current_set, best_size
        
        nodes = set(nodes)
        if not nodes:
            if len(current_set) > best_size:
                best_size = len(current_set)
            return current_set, best_size
        
        # Upper bound
        ub = len(current_set) + upper_bound(nodes, conflict)
        if ub <= best_size:
            return current_set, best_size
        
        # Pick a vertex to branch on
        # Use heuristic: pick vertex with highest degree
        best_v = None
        best_deg = -1
        for v in nodes:
            deg = len(conflict[v] & nodes)
            if deg > best_deg:
                best_deg = deg
                best_v = v
        
        # Branch 1: include best_v
        new_nodes = nodes - {best_v} - conflict[best_v]
        new_set = current_set | {best_v}
        result_set, result_size = branch_and_bound(new_nodes, new_set, best_size)
        best_size = max(best_size, result_size)
        
        # Branch 2: exclude best_v
        nodes.discard(best_v)
        result_set, result_size = branch_and_bound(nodes, current_set, best_size)
        best_size = max(best_size, result_size)
        
        return result_set, best_size
    
    # Greedy initial solution
    def greedy_solution(pool):
        pool = set(pool)
        selected = set()
        while pool:
            min_deg_node = None
            min_deg = float('inf')
            for node in sorted(pool):
                deg = len(conflict[node] & pool)
                if deg < min_deg:
                    min_deg = deg
                    min_deg_node = node
                    if min_deg == 0:
                        break
            if min_deg_node is None:
                break
            selected.add(min_deg_node)
            pool.discard(min_deg_node)
            pool.difference_update(conflict[min_deg_node])
        return selected
    
    # Try to add more elements
    def try_add(solution):
        sol_set = set(solution)
        changed = True
        while changed:
            changed = False
            for a in candidates:
                if a not in sol_set and not (conflict[a] & sol_set):
                    sol_set.add(a)
                    changed = True
        return sol_set
    
    # Local search with add and swap moves
    def local_search(solution, max_iters=5000):
        sol_set = set(solution)
        best_set = set(sol_set)
        best_size = len(sol_set)
        
        for it in range(max_iters):
            if time.time() - start_time > deadline - 3.0:
                break
            
            improved = False
            
            # Try to add
            for a in candidates:
                if a not in sol_set and not (conflict[a] & sol_set):
                    sol_set.add(a)
                    improved = True
                    if len(sol_set) > best_size:
                        best_size = len(sol_set)
                        best_set = set(sol_set)
                    break
            
            if improved:
                continue
            
            # Try swap: remove one, add one or more
            # For each element in sol_set, try removing it and see if we can add more
            for remove_candidate in list(sol_set):
                temp_set = sol_set - {remove_candidate}
                # Try to add as many as possible
                added = set()
                for a in candidates:
                    if a not in temp_set and a != remove_candidate and not (conflict[a] & temp_set):
                        temp_set.add(a)
                        added.add(a)
                
                if len(temp_set) > len(sol_set):
                    sol_set = temp_set
                    if len(sol_set) > best_size:
                        best_size = len(sol_set)
                        best_set = set(sol_set)
                    improved = True
                    break
            
            if not improved:
                break
        
        return best_set
    
    # Start with greedy
    best_solution = greedy_solution(candidates)
    best_solution = try_add(best_solution)
    best_size = len(best_solution)
    
    # Randomized restarts
    for seed in range(200):
        if time.time() - start_time > deadline - 10.0:
            break
        random.seed(seed)
        pool = set(candidates)
        selected = set()
        while pool:
            degrees = {}
            for node in pool:
                degrees[node] = len(conflict[node] & pool)
            min_deg = min(degrees.values())
            min_nodes = [node for node, deg in degrees.items() if deg == min_deg]
            node = random.choice(min_nodes)
            selected.add(node)
            pool.discard(node)
            pool.difference_update(conflict[node])
        
        selected = try_add(selected)
        if len(selected) > best_size:
            best_size = len(selected)
            best_solution = selected
    
    # Local search on best solution
    best_solution = local_search(best_solution)
    best_size = len(best_solution)
    
    # Try B&B from a reduced problem
    # Remove elements that are "forced" (no conflicts with remaining)
    # Actually, let's try B&B on the complement: find minimum vertex cover
    # Or just do B&B on the independent set problem with a time limit
    
    # Let's try a different approach: use the fact that we can do B&B
    # but with a tighter time budget
    
    remaining_time = deadline - 5.0 - (time.time() - start_time)
    if remaining_time > 5.0:
        # Try B&B
        nodes = set(candidates) - best_solution
        # We want to find if we can add more to best_solution
        # B&B to find max independent set in nodes that can be added
        # But we need to consider conflicts with best_solution too
        
        # Actually, let's just try to find a better solution via B&B
        # Start B&B from scratch with time limit
        bbb_start = time.time()
        
        def bbb(nodes, current_set, best_size_ref):
            if time.time() - bbb_start > remaining_time - 2.0:
                return current_set, best_size_ref[0]
            
            nodes = set(nodes)
            if not nodes:
                if len(current_set) > best_size_ref[0]:
                    best_size_ref[0] = len(current_set)
                return current_set, best_size_ref[0]
            
            ub = len(current_set) + upper_bound(nodes, conflict)
            if ub <= best_size_ref[0]:
                return current_set, best_size_ref[0]
            
            # Pick vertex with highest degree
            best_v = None
            best_deg = -1
            for v in nodes:
                deg = len(conflict[v] & nodes)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            # Include best_v
            new_nodes = nodes - {best_v} - conflict[best_v]
            new_set = current_set | {best_v}
            rs, rsize = bbb(new_nodes, new_set, best_size_ref)
            best_size_ref[0] = max(best_size_ref[0], rsize)
            
            # Exclude best_v
            nodes.discard(best_v)
            rs, rsize = bbb(nodes, current_set, best_size_ref)
            best_size_ref[0] = max(best_size_ref[0], rsize)
            
            return rs, best_size_ref[0]
        
        best_size_ref = [best_size]
        bbb_solution, bbb_size = bbb(set(candidates), set(), best_size_ref)
        if bbb_size > best_size:
            best_solution = bbb_solution
            best_size = bbb_size
    
    # Final local search
    best_solution = local_search(best_solution, max_iters=10000)
    best_size = len(best_solution)
    
    # Final verification
    best_set = set(best_solution)
    for a in best_set:
        if 2 * a in squares:
            return []
    for i, a in enumerate(best_set):
        for b in best_set:
            if a < b and a + b in squares:
                return []
    
    return list(best_set)
