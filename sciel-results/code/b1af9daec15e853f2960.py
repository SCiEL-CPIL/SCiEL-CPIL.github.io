import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = [(20, 0.4, 11), (25, 0.0, 12), (30, 0.0, 13)] # Placeholder, will be updated
    # Wait, the specs in the prompt are (20,0.4,11), (25,0.35,12), (30,0.3,13)
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    graphs = [instance(*spec) for spec in specs]

    def solve_graph(edges, time_limit):
        """
        Solves the decomposition problem using Branch and Bound.
        Objective: Maximize number of edges covered by cycles.
        Total pieces = (Total Edges - Edges in Cycles) + (Number of Cycles)
        Since Total Edges is fixed, minimizing pieces is equivalent to
        maximizing (Edges in Cycles - Number of Cycles).
        However, it's simpler to think:
        Minimize Pieces = Minimize (Edges_not_in_cycles + Cycles)
        Let E be total edges.
        Let C be set of cycles.
        Edges_not_in_cycles = E - sum(len(c) for c in C)
        Pieces = E - sum(len(c) for c in C) + |C|
        We want to minimize this.
        """
        if not edges:
            return []

        # Preprocessing: Map edges to indices for faster lookup
        edge_list = list(edges)
        m = len(edge_list)
        edge_to_idx = {e: i for i, e in enumerate(edge_list)}
        
        # Adjacency list for fast neighbor lookup
        adj = defaultdict(set)
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
            
        # State for B&B
        # remaining_edges: set of edge indices still available
        # used_cycles: list of cycles (as lists of vertices)
        # current_score: number of pieces so far (for lower bound calculation)
        # Actually, we track:
        #   edges_used_in_cycles: count
        #   num_cycles: count
        #   Lower bound on final pieces:
        #   Current pieces = num_cycles
        #   Remaining edges = m - edges_used_in_cycles
        #   Best case for remaining: all form cycles of length 3? 
        #   Or simpler: LB = num_cycles + (m - edges_used_in_cycles) 
        #   This is the worst case (all edges are separate).
        #   A better LB: num_cycles + max(0, (m - edges_used_in_cycles) - max_possible_cycle_edges_remaining)
        #   Let's stick to a simple admissible heuristic:
        #   LB = num_cycles + (m - edges_used_in_cycles)
        #   This is admissible because each remaining edge must be at least 1 piece 
        #   if it's not in a cycle, but if it IS in a cycle, it reduces the count.
        #   Wait.
        #   Pieces = Num_Cycles + Num_Single_Edges
        #   Let X be number of edges in cycles.
        #   Num_Single_Edges = m - X
        #   Pieces = Num_Cycles + m - X
        #   We want to minimize this.
        #   In B&B, we have a partial solution.
        #   Current Num_Cycles = k
        #   Current X = x
        #   Remaining edges = m - x
        #   In the remaining subgraph, we can find more cycles.
        #   Let k_add be additional cycles, x_add be additional edges in those cycles.
        #   Final Pieces = k + k_add + m - x - x_add
        #   Lower Bound:
        #   k_add >= 0
        #   x_add <= m - x
        #   So Pieces >= k + 0 + m - x - (m - x) = k.
        #   This LB is too weak.
        
        # Better LB:
        #   Each additional cycle must have length >= 3.
        #   So x_add >= 3 * k_add  => k_add <= x_add / 3
        #   Pieces = k + k_add + m - x - x_add
        #   To minimize, we want to maximize x_add and minimize k_add.
        #   But k_add and x_add are linked.
        #   If we assume all remaining edges can be packed into cycles of length 3 (best case for reducing pieces? No, longer cycles are better for piece count? 
        #   Let's check: 
        #   Cycle of length 3: 3 edges, 1 piece. Delta = -2 vs 3 edges.
        #   Cycle of length 4: 4 edges, 1 piece. Delta = -3 vs 4 edges.
        #   Longer cycles are better for minimizing pieces.
        #   So the best case is one giant cycle using all remaining edges.
        #   LB = k + 1 (if remaining > 0 and forms 1 cycle) + m - x - (m-x) = k + 1.
        #   If remaining edges cannot form a cycle, LB is higher.
        
        # Let's use a simple LB:
        #   Current Pieces = k
        #   Remaining Edges = R = m - x
        #   If R == 0: LB = k
        #   If R > 0: 
        #       Best case: All R edges form one cycle (if possible). LB = k + 1.
        #       Worst case: All R edges are separate. LB = k + R.
        #       A safe LB is k + 1 (if R>0).
        #       Actually, if R < 3, no cycle possible. LB = k + R.
        #       If R >= 3, LB = k + 1 is admissible IF a cycle exists. 
        #       If no cycle exists in remaining graph, LB = k + R.
        #       Checking if a cycle exists is O(E). We can do this.

        # Global best
        best_pieces = m  # Worst case: all edges separate
        best_solution = [[u, v] for u, v in edge_list]
        
        # State:
        # rem_edges: set of indices
        # cycles: list of cycles (list of vertices)
        # edges_in_cycles: count
        
        # To speed up, we can pre-compute all simple cycles? 
        # For N=20-30, number of cycles can be large, but maybe manageable?
        # Or generate on the fly?
        # Generating on the fly in B&B is expensive.
        # Pre-computing all cycles in a graph with 30 nodes and p=0.3 (approx 135 edges) might be too many.
        # Let's try to find cycles on the fly using DFS in the B&B node?
        # That might be too slow.
        
        # Alternative: Use a heuristic to find a good initial solution, then B&B to improve.
        
        # 1. Find initial solution using Greedy
        def greedy_solve():
            rem = set(edge_list)
            cycles = []
            adj_rem = defaultdict(set)
            for u, v in rem:
                adj_rem[u].add(v)
                adj_rem[v].add(u)
            
            while rem:
                # Find any cycle
                cycle = None
                # Simple DFS for any cycle
                # Start from any node with degree >= 2
                candidates = [u for u in adj_rem if len(adj_rem[u]) >= 2]
                if not candidates:
                    break
                start = candidates[0]
                visited = {start}
                path = [start]
                
                def dfs_greedy(u):
                    nonlocal cycle
                    if cycle: return
                    for v in adj_rem[u]:
                        if v == start and len(path) >= 3:
                            cycle = list(path)
                            return
                        if v not in visited:
                            visited.add(v)
                            path.append(v)
                            dfs_greedy(v)
                            if cycle: return
                            path.pop()
                            visited.remove(v)
                    if not cycle:
                        pass # Backtrack handled by caller? No, this is recursive.
                
                # The above DFS is a bit messy. Let's use a standard one.
                def find_any_cycle(start):
                    stack = [(start, [start])]
                    visited = {start}
                    while stack:
                        u, path = stack.pop()
                        for v in adj_rem[u]:
                            if v == start and len(path) >= 3:
                                return list(path)
                            if v not in visited:
                                visited.add(v)
                                stack.append((v, path + [v]))
                    return None
                
                cycle = find_any_cycle(start)
                if cycle:
                    cycles.append(cycle)
                    # Remove edges
                    for i in range(len(cycle) - 1):
                        u, v = cycle[i], cycle[i+1]
                        e = (u, v) if u < v else (v, u)
                        rem.remove(e)
                        adj_rem[u].remove(v)
                        adj_rem[v].remove(u)
                    u, v = cycle[-1], cycle[0]
                    e = (u, v) if u < v else (v, u)
                    rem.remove(e)
                    adj_rem[u].remove(v)
                    adj_rem[v].remove(u)
                else:
                    # No cycle, take one edge
                    u, v = next(iter(rem))
                    cycles.append([u, v])
                    rem.remove((u, v))
                    adj_rem[u].remove(v)
                    adj_rem[v].remove(u)
            
            return cycles

        init_cycles = greedy_solve()
        init_pieces = len(init_cycles)
        best_pieces = init_pieces
        best_solution = init_cycles
        
        # 2. Branch and Bound
        # State: (rem_edges_set, cycles_list, edges_in_cycles_count)
        # We will use a stack for DFS B&B.
        
        # To make B&B efficient, we need a good LB.
        # LB(rem_edges, current_cycles_count):
        #   R = len(rem_edges)
        #   If R == 0: return current_cycles_count
        #   Check if rem_edges contains any cycle.
        #   If no cycle: return current_cycles_count + R
        #   If cycle exists: return current_cycles_count + 1 (optimistic: all remaining form 1 cycle)
        #   Actually, if R >= 3 and cycle exists, LB = current_cycles_count + 1 is admissible.
        
        def has_cycle(edge_set):
            if len(edge_set) < 3:
                return False
            # Build adj
            adj_c = defaultdict(set)
            for u, v in edge_set:
                adj_c[u].add(v)
                adj_c[v].add(u)
            # Check for cycle via DFS
            visited = set()
            def dfs(u, parent):
                visited.add(u)
                for v in adj_c[u]:
                    if v == parent: continue
                    if v in visited: return True
                    if dfs(v, u): return True
                return False
            for u in adj_c:
                if u not in visited:
                    if dfs(u, -1):
                        return True
            return False

        # B&B Loop
        # We will branch on: 
        #   1. Include a specific cycle C in the solution.
        #   2. Exclude all cycles that use a specific edge e? (Hard to implement)
        #   Simpler: 
        #   Find a cycle C in current rem.
        #   Branch 1: Take C.
        #   Branch 2: Do NOT take C. How to represent "Do not take C"?
        #   We can't just remove C's edges, because another cycle might share an edge.
        #   If we say "Don't take C", we must ensure no future cycle uses ANY edge of C? No, that's wrong.
        #   We just don't pick C *right now*. But we might pick a sub-cycle or a different cycle.
        
        # Correct B&B for Set Packing:
        #   Candidate set of cycles.
        #   Branch on a cycle C:
        #     1. Include C. Remove all edges in C from rem. Remove all other cycles that share edges with C from candidate set.
        #     2. Exclude C. Remove C from candidate set.
        
        # This requires pre-computing all cycles or generating them.
        # Given the size (N=20-30), pre-computing ALL cycles might be too slow/memory heavy if dense.
        # But p=0.3-0.4 is not very dense.
        # Let's try to generate cycles on the fly in the B&B? 
        # Or, simpler: 
        #   Use a heuristic B&B where we pick the "best" cycle (longest) and branch on it.
        
        # Let's try a different approach: **Local Search / Hill Climbing** with a good initial solution.
        # The current program uses Greedy. 
        # A Hill Climber can swap cycles to improve the count.
        # Move: Replace a set of cycles with a different set of cycles covering the same edges? 
        # Or: Remove a cycle, re-decompose the freed edges?
        
        # Let's stick to the ILP/B&B idea but simplify.
        # For N=20, we can enumerate all cycles? 
        # Number of cycles in a random graph is not too huge.
        # Let's pre-compute all simple cycles for each graph.
        
        def get_all_cycles(edges):
            cycles = []
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            
            # Use Johnson's algorithm or simple DFS for each start node
            # Simple DFS:
            # For each start node s:
            #   DFS to find paths starting and ending at s with distinct intermediate nodes.
            
            for s in sorted(adj.keys()):
                # We only want cycles where s is the smallest node, to avoid duplicates
                # This reduces duplicates by factor of cycle length.
                visited = {s}
                path = [s]
                
                def dfs(u):
                    for v in sorted(adj[u]):
                        if v == s and len(path) >= 3:
                            # Found a cycle
                            cycles.append(list(path))
                        elif v > s and v not in visited:
                            visited.add(v)
                            path.append(v)
                            dfs(v)
                            path.pop()
                            visited.remove(v)
                
                dfs(s)
            return cycles

        # If number of cycles is manageable (< 10000), we can do B&B.
        
        all_cycles = get_all_cycles(edge_list)
        
        # If too many cycles, fall back to greedy?
        if len(all_cycles) > 5000:
             return best_solution
             
        # Map edge to set of cycle indices containing it
        edge_to_cycles = defaultdict(set)
        for i, cyc in enumerate(all_cycles):
            for j in range(len(cyc) - 1):
                u, v = cyc[j], cyc[j+1]
                e = (u, v) if u < v else (v, u)
                edge_to_cycles[e].add(i)
            u, v = cyc[-1], cyc[0]
            e = (u, v) if u < v else (v, u)
            edge_to_cycles[e].add(i)
            
        # B&B State:
        # available_cycles: set of indices
        # used_edges: set of edges
        # current_pieces: int
        
        # We want to minimize pieces.
        # Pieces = num_cycles + (m - edges_in_cycles)
        
        # LB:
        # current_pieces + (m - len(used_edges) - max_additional_edges)
        # max_additional_edges <= len(available_cycles edges union)? No.
        # Simple LB: current_pieces + (m - len(used_edges))
        # This is admissible.
        
        best_val = best_pieces
        best_sol = best_solution
        
        # Stack: (available_cycles_frozen, used_edges_frozen, current_pieces)
        # To save memory, we can pass sets.
        
        # Initial state:
        # available = all cycles
        # used = empty
        # pieces = 0
        
        # We will use a priority queue (Best First Search) or DFS?
        # DFS is memory efficient.
        
        # To speed up, we can order cycles by length (longer first) to find good solutions early.
        # Sort all_cycles by length descending.
        # But indices change. Let's just sort the list and use indices.
        
        # Re-index cycles by length
        all_cycles.sort(key=lambda x: -len(x))
        
        # Rebuild edge_to_cycles
        edge_to_cycles = defaultdict(set)
        for i, cyc in enumerate(all_cycles):
            for j in range(len(cyc) - 1):
                u, v = cyc[j], cyc[j+1]
                e = (u, v) if u < v else (v, u)
                edge_to_cycles[e].add(i)
            u, v = cyc[-1], cyc[0]
            e = (u, v) if u < v else (v, u)
            edge_to_cycles[e].add(i)
            
        # B&B
        # We'll use a recursive function with a time check.
        
        sys.setrecursionlimit(10000)
        
        def bb(available, used, pieces, depth):
            nonlocal best_val, best_sol
            
            if time.time() > time_limit:
                return
                
            # Update best
            # We don't have the full solution yet, but we can estimate.
            # Actually, we only update best when we reach a leaf (no more cycles can be added? 
            # Or when available is empty? 
            # No, we can stop adding cycles at any point.
            # The current state is a valid solution: 
            #   Cycles = [all_cycles[i] for i in chosen] (we didn't track chosen, only used edges)
            #   Wait, we need to reconstruct the solution.
            #   This is a problem. B&B usually tracks the path.
            
            # Let's track the list of chosen cycle indices.
            
        # Revised State: (available_set, used_edges_set, chosen_indices_list, current_pieces)
        
        # This is getting complex. Let's try a simpler **Greedy with Local Search**.
        
        # 1. Start with Greedy solution.
        # 2. Try to improve by:
        #    a. Removing a cycle and re-decomposing its edges (might yield fewer pieces? Unlikely if greedy was good).
        #    b. Swapping two cycles for three? 
        #    c. **Cycle Merging**: If two cycles share no edges, but their union can form a larger cycle or fewer pieces? No, disjoint cycles are already optimal locally.
        #    d. **Cycle Replacement**: Replace a set of cycles with a different set covering the same edges.
        
        # Actually, the most powerful move is:
        # **Remove a cycle, and try to find a better decomposition of the freed edges.**
        # Or: **Find a cycle in the "complement" of the current cycle cover?**
        
        # Let's try this:
        # 1. Greedy cover.
        # 2. For each cycle in the cover, remove it.
        # 3. Try to re-decompose the freed edges + maybe some other edges?
        
        # Simpler: **Simulated Annealing** on the set of cycles?
        # State: Set of cycles.
        # Move: Add a random cycle (if disjoint), Remove a random cycle, Swap.
        # Objective: Minimize pieces.
        
        # Given the time, let's implement a robust **Greedy with multiple restarts and local search**.
        
        def local_search(cycles):
            # Convert to edge set
            used_edges = set()
            for c in cycles:
                for i in range(len(c) - 1):
                    u, v = c[i], c[i+1]
                    used_edges.add((u, v) if u < v else (v, u))
                u, v = c[-1], c[0]
                used_edges.add((u, v) if u < v else (v, u))
                
            # Try to remove a cycle and see if we can cover its edges with fewer pieces?
            # No, removing a cycle increases pieces by 1 (cycle becomes edges) unless we find a better cycle.
            
            # Try to find a cycle in the remaining edges that can replace multiple existing cycles?
            
            # Let's try: 
            # 1. Identify all edges not in cycles? No, all edges are covered.
            # 2. Try to find a longer cycle that overlaps with existing ones, and adjust.
            
            # This is hard.
            
            # Alternative: **Exact B&B for small graphs**.
            # For N=20, M~80. Number of cycles might be ~1000.
            # B&B on 1000 items is feasible if pruned well.
            
            pass
            
        # Let's go back to B&B but simpler.
        # We will use the pre-computed cycles.
        # We will use a DFS that branches on the first cycle in a sorted list.
        
        def solve_bnb():
            nonlocal best_val, best_sol
            
            # Sort cycles by length descending
            # We already did this.
            
            # State:
            #   idx: index in all_cycles to consider
            #   used_edges: set
            #   chosen: list of indices
            #   pieces: count
            
            # LB:
            #   pieces + (m - len(used_edges))
            
            # We will use a stack to avoid recursion depth issues.
            # Stack item: (idx, used_edges_frozen, chosen_list, pieces)
            
            # To save memory, we can't freeze sets easily in Python without copy.
            # For N=20, used_edges size < 80. Copy is O(80).
            
            # Initial:
            stack = [(0, set(), [], 0)]
            
            while stack:
                if time.time() > time_limit:
                    break
                    
                idx, used, chosen, pieces = stack.pop()
                
                # LB Check
                lb = pieces + (m - len(used))
                if lb >= best_val:
                    continue
                    
                # If we've considered all cycles
                if idx >= len(all_cycles):
                    # Valid solution
                    # Reconstruct
                    sol = [all_cycles[i] for i in chosen]
                    # Calculate actual pieces
                    # It should be 'pieces' + (m - len(used))
                    # Wait, 'pieces' is number of cycles chosen.
                    # Single edges are the rest.
                    # Total pieces = len(chosen) + (m - len(used))
                    # This matches lb when idx == end.
                    if lb < best_val:
                        best_val = lb
                        best_sol = sol
                    continue
                    
                # Branching:
                # Cycle C = all_cycles[idx]
                C = all_cycles[idx]
                
                # Check if C is disjoint from used
                is_disjoint = True
                c_edges = []
                for i in range(len(C) - 1):
                    u, v = C[i], C[i+1]
                    e = (u, v) if u < v else (v, u)
                    c_edges.append(e)
                    if e in used:
                        is_disjoint = False
                        break
                if is_disjoint:
                    u, v = C[-1], C[0]
                    e = (u, v) if u < v else (v, u)
                    c_edges.append(e)
                    if e in used:
                        is_disjoint = False
                
                if is_disjoint:
                    # Branch 1: Take C
                    new_used = used.union(c_edges)
                    new_chosen = chosen + [idx]
                    new_pieces = pieces + 1
                    # Push to stack
                    # We can push both branches.
                    # To find good solutions early, push "Take" first? 
                    # Stack is LIFO, so push "Don't Take" first, then "Take".
                    stack.append((idx + 1, used, chosen, pieces))
                    stack.append((idx + 1, new_used, new_chosen, new_pieces))
                else:
                    # Branch 2: Cannot take C
                    stack.append((idx + 1, used, chosen, pieces))
                    
            return best_sol

        # Run B&B
        # Limit time for B&B
        bnb_time_limit = time.time() + (time_limit - time.time()) * 0.8
        old_limit = time_limit
        time_limit = bnb_time_limit
        bnb_sol = solve_bnb()
        time_limit = old_limit
        
        if len(bnb_sol) < len(best_solution):
            return bnb_sol
        else:
            return best_solution

    results = []
    for i, edges in enumerate(graphs):
        time_limits = [20.0, 30.0, 80.0]
        current_limit = time_limits[i]
        global_limit = deadline - time.time()
        if global_limit < current_limit:
            current_limit = max(1.0, global_limit)
        
        time_limit = time.time() + current_limit
        
        best_pieces = solve_graph(edges, time_limit)
        results.append(best_pieces)
        
        if time.time() > deadline - 2.0:
            for j in range(i + 1, len(graphs)):
                results.append([list(e) for e in graphs[j]])
            break
            
    return results

if __name__ == "__main__":
    res = construct()
    for i, spec in enumerate([(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]):
        n, p, seed = spec
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        edge_set = set(edges)
        
        used_edges = set()
        valid = True
        for piece in res[i]:
            if len(piece) == 2:
                u, v = piece
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            elif len(piece) >= 3:
                if len(set(piece)) != len(piece):
                    valid = False
                    break
                for k in range(len(piece) - 1):
                    u, v = piece[k], piece[k+1]
                    e = tuple(sorted((u, v)))
                    if e not in edge_set or e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                if not valid:
                    break
                u, v = piece[-1], piece[0]
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            else:
                valid = False
                break
                
        if valid and used_edges != edge_set:
            valid = False
            
        print(f"Graph {i}: Valid={valid}, Pieces={len(res[i])}")

