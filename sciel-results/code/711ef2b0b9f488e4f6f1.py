import time
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    timeout = 55.0
    
    n = 256
    adj = [set() for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                adj[a].add(b)
                adj[b].add(a)
    
    # Precompute neighbor bitmasks for fast intersection checks
    # Using Python ints as bitsets (bit i represents vertex i+1)
    adj_mask = [0] * (n + 1)
    for v in range(1, n + 1):
        mask = 0
        for u in adj[v]:
            mask |= (1 << u)
        adj_mask[v] = mask
    
    # Greedy coloring using bitmask for efficiency
    def greedy_color(P):
        """Color P greedily. Returns (colors_list, max_color, color_to_vertices)."""
        if not P:
            return [], 0, {}
        
        colors = [0] * len(P)
        color_map = {}
        # Process in order; use bitmask to quickly check colored neighbors
        colored_mask = 0  # bitmask of all colored vertices so far
        
        for i, v in enumerate(P):
            # Find colors used by already-colored neighbors of v
            neighbor_colors = set()
            # Check which neighbors of v are already colored
            v_neighbors_mask = adj_mask[v] & colored_mask
            # For each colored neighbor, add its color
            temp = v_neighbors_mask
            while temp:
                # Get lowest set bit
                lsb = temp & (-temp)
                u = lsb.bit_length() - 1
                if u in color_map:
                    neighbor_colors.add(color_map[u])
                temp ^= lsb
            
            c = 1
            while c in neighbor_colors:
                c += 1
            colors[i] = c
            color_map[v] = c
            colored_mask |= (1 << v)
        
        max_c = max(colors) if colors else 0
        return colors, max_c, color_map

    def greedy_mis_on_subset(vertices, adj):
        """Greedy MIS on a subset of vertices using minimum degree heuristic."""
        remaining = set(vertices)
        result = []
        while remaining:
            min_v = None
            min_deg = float('inf')
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            result.append(min_v)
            remaining.discard(min_v)
            remaining -= adj[min_v]
        return result

    def initial_solution():
        """Build a good initial solution: all odds + compatible evens."""
        odds = [i for i in range(1, n + 1) if i % 2 == 1]
        compatible_evens = []
        for e in range(2, n + 1, 2):
            compatible = True
            for o in odds:
                if e in adj[o]:
                    compatible = False
                    break
            if compatible:
                compatible_evens.append(e)
        
        even_mis = greedy_mis_on_subset(compatible_evens, adj)
        return odds + even_mis

    def local_search_swap(sol, max_iters=5000):
        """Try to improve solution by swapping or adding vertices."""
        sol_set = set(sol)
        best = sol[:]
        best_size = len(sol)
        
        # Try adding vertices not in solution
        for v in range(1, n + 1):
            if v not in sol_set:
                # Check if v can be added
                if all(v not in adj[u] for u in sol_set):
                    new_sol = sol + [v]
                    if len(new_sol) > best_size:
                        best = new_sol
                        best_size = len(new_sol)
                        sol = new_sol
                        sol_set = set(sol)
        
        # Try swaps: remove one, add one
        changed = True
        while changed and time.time() - start_time < timeout - 5:
            changed = False
            for i, v_out in enumerate(sol):
                if time.time() - start_time > timeout - 5:
                    break
                trial_sol = sol[:i] + sol[i+1:]
                trial_set = set(trial_sol)
                for v_in in range(1, n + 1):
                    if v_in in trial_set or v_in == v_out:
                        continue
                    if all(v_in not in adj[u] for u in trial_set):
                        new_sol = trial_sol + [v_in]
                        if len(new_sol) > best_size:
                            best = new_sol
                            best_size = len(new_sol)
                            sol = new_sol
                            sol_set = set(sol)
                            changed = True
                            break
                if changed:
                    break
        
        return best, best_size

    # Get initial solution
    init_sol = initial_solution()
    best_solution = init_sol[:]
    best_size = len(init_sol)
    
    # Try local search to improve initial solution
    improved_sol, improved_size = local_search_swap(init_sol)
    if improved_size > best_size:
        best_solution = improved_sol
        best_size = improved_size
    
    # Tomita's Branch and Bound
    def expand(P, sol):
        nonlocal best_solution, best_size
        
        if time.time() - start_time > timeout:
            return
            
        if not P:
            if len(sol) > best_size:
                best_size = len(sol)
                best_solution = sol.copy()
            return
            
        # Compute UB using greedy coloring
        colors, max_color, _ = greedy_color(P)
        
        if len(sol) + max_color <= best_size:
            return
            
        # Order P by color (ascending) then by degree (descending)
        P_with_info = [(P[i], colors[i], len(adj[P[i]])) for i in range(len(P))]
        P_with_info.sort(key=lambda x: (x[1], -x[2]))
        P = [v for v, c, d in P_with_info]
        
        # Track remaining color count for tighter bounds
        remaining_colors = max_color
        
        for i in range(len(P)):
            if time.time() - start_time > timeout:
                return
                
            v = P[i]
            
            # Include v: remove v and its neighbors from remaining P
            v_mask = 1 << v
            neighbor_mask = adj_mask[v]
            
            new_P = []
            for j in range(i + 1, len(P)):
                u = P[j]
                if u != v and u not in adj[v]:
                    new_P.append(u)
            
            # Recurse with v included
            expand(new_P, sol + [v])
            
            # Exclude v: remove v from P
            # Update remaining colors: v had color colors[i], but we need to recompute
            # Simple approach: just continue; the bound will be checked at next level
            # For tighter bound, we could decrement remaining_colors if v was the last of its color
            # But that's complex; rely on re-coloring at next level
            
            # Check if remaining P can beat best
            if i + 1 < len(P):
                # Quick bound: max possible from remaining is at most remaining_colors
                # But we removed v, so it might be less. Use conservative bound.
                if len(sol) + remaining_colors <= best_size:
                    break
    
    # Start B&B from the full vertex set, but use the initial solution for pruning
    initial_P = list(range(1, n + 1))
    expand(initial_P, [])
    
    # Final local search polish on the best solution found
    if time.time() - start_time < timeout - 3:
        polished_sol, polished_size = local_search_swap(best_solution, max_iters=20000)
        if polished_size > best_size:
            best_solution = polished_sol
            best_size = polished_size
    
    if not best_solution:
        best_solution = greedy_mis_on_subset(range(1, n + 1), adj)

    return list(best_solution)
