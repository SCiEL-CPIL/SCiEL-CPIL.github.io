import time
import random
import numpy as np

def construct(k, g):
    # 1. Fallback: Lazebnik-Ustimenko-Woldar graph
    # This is the seed program's logic, optimized slightly for speed.
    def get_luw_graph(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    n_luw, edges_luw = get_luw_graph(k, g)
    best_n = n_luw
    best_edges = edges_luw

    # 2. Search for smaller graphs using Simulated Annealing with 2-switches
    
    deadline = time.time() + 110  # 10s safety margin

    def calculate_girth(n, adj):
        # Non-backtracking BFS from every node to find shortest cycle
        min_girth = float('inf')
        max_depth = g // 2
        for start in range(n):
            # BFS state: (node, prev, depth)
            queue = [(start, -1, 0)]
            visited = set()
            while queue:
                node, prev, depth = queue.pop(0)
                key = (node, prev)
                if key in visited:
                    continue
                visited.add(key)
                
                if depth == max_depth:
                    continue

                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if nb == start:
                        cycle_len = depth + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                        if min_girth < g:
                            return min_girth
                    else:
                        if depth + 1 < max_depth:
                            queue.append((nb, node, depth + 1))
        return min_girth

    def is_valid_graph(n, edges, k, g):
        # Check degree
        if len(edges) != n * k // 2:
            return False
        adj = [set() for _ in range(n)]
        for u, v in edges:
            if u == v: return False
            if v in adj[u]: return False # Duplicate
            adj[u].add(v)
            adj[v].add(u)
        
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Check girth
        girth = calculate_girth(n, adj)
        return girth >= g

    def generate_random_k_regular(n, k):
        # Random k-regular graph generation using Configuration Model
        # We need to handle the simple graph constraint (no loops, no multi-edges)
        # We use a "repair" approach or rejection sampling.
        # Rejection is slow. Let's try a smarter construction.
        
        # 1. Create stubs
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        
        # 2. Shuffle
        random.shuffle(stubs)
        
        # 3. Pair up stubs
        # To avoid loops and multi-edges, we can use a simple check and retry if invalid.
        # But we want a fast generator.
        # Let's try a few attempts.
        for _ in range(50):
            random.shuffle(stubs)
            edges = set()
            valid = True
            adj = [set() for _ in range(n)]
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
            
            if valid:
                return list(edges), adj
        return None, None

    # Simulated Annealing
    # Objective: Maximize Girth.
    # Move: 2-Switch.
    # Pick two non-adjacent edges (u,v) and (w,x) such that (u,w) and (v,x) are not edges and u!=w, v!=x, etc.
    # Replace with (u,w) and (v,x).
    
    # For n=320, k=4.
    # Number of edges = 640.
    # Picking random edges is O(1).
    # Checking validity is O(1) with adjacency sets.
    # Updating girth is expensive. We need a fast girth estimator or incremental update.
    # Incremental girth update is hard.
    # Alternative: Count number of short cycles.
    # Let's define the objective as -1 * (number of cycles of length < g).
    # If we reach 0, we are valid.
    
    def count_short_cycles(n, adj, max_len):
        # Count cycles of length 3 to max_len
        # This is still expensive if max_len is large.
        # For g=10, we care about 3,4,5,6,7,8,9.
        # Most random graphs have many 3,4,5 cycles.
        # We can just check for the existence of ANY cycle < g.
        # But for SA, we need a score.
        # Let's use the girth itself as the score.
        # But calculating girth is O(n * (k-1)^(g/2)).
        # For n=320, k=4, g=10: 320 * 3^5 = 320 * 243 = 77760 ops.
        # This is fast enough for a few iterations per second?
        # 77k ops is very fast in Python if optimized? No, Python is slow.
        # 77k Python steps is ~0.01s.
        # So we can do ~100 SA steps per second.
        # 110s -> 11000 steps.
        # Is that enough? Probably not for n=320.
        
        # We need a faster objective.
        # How about counting triangles?
        # Or just checking if girth >= g?
        # If we use a binary objective (valid/invalid), SA gets stuck.
        
        # Let's try to optimize calculate_girth.
        # We can stop early if we find a cycle of length 3.
        
        pass

    # Let's stick to the current design but improve the search.
    # The directive says: "Replace the deterministic algebraic construction with a stochastic local search (Simulated Annealing) that initializes a random k-regular graph and iteratively applies 2-switches to eliminate short cycles, using the girth as the objective function to be maximized."
    
    # So I must implement SA.
    
    # To make it fast:
    # 1. Fast girth calculation.
    # 2. Efficient 2-switches.
    
    # Let's try to implement a faster girth check.
    # We can precompute neighbors.
    
    def fast_girth_check(n, adj, limit):
        # Returns girth if < limit, else limit
        # Use BFS from each node, but prune early.
        min_girth = limit
        for start in range(n):
            # BFS
            queue = [(start, -1, 0)]
            visited = set()
            found = False
            while queue:
                node, prev, depth = queue.pop(0)
                key = (node, prev)
                if key in visited:
                    continue
                visited.add(key)
                
                if depth + 1 >= limit:
                    continue # We don't care about cycles >= limit
                
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if nb == start:
                        cycle_len = depth + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                        if min_girth < limit:
                            return min_girth
                    else:
                        if depth + 1 < limit:
                            queue.append((nb, node, depth + 1))
            if min_girth < limit:
                return min_girth
        return min_girth

    # SA Loop
    # Target n: We want to beat n_luw.
    # Let's try n = n_luw - 10, n_luw - 20, etc?
    # Or just try to find ANY valid graph with n < n_luw.
    # The Moore bound is 243.
    # Let's try n = 320 (if n_luw > 320) or n_luw - 1.
    
    # Actually, the seed program returns n_luw which is 2048 for k=4, g=10?
    # Let's check LUW size.
    # q=4, d=5 (g=10 -> d=5? g-5=5, odd -> d=5).
    # Vertices: 2 * q * q^(d-1)? No.
    # The LUW graph size is 2 * q * q^d?
    # For k=4, g=10, n_luw is likely large.
    # We want to get close to 320.
    
    # Let's try n = 320.
    # If n_luw <= 320, we are already good? No, we want to MINIMIZE n.
    # If n_luw is 320, we can't beat it easily?
    # But the task says "smallest known such graph has 320 vertices".
    # So if we return 320, we are at the best known.
    # If we return 319, we are better.
    
    # Let's try to find a graph with n < n_luw.
    # We will try a few candidate n values.
    
    candidate_ns = []
    if n_luw > 320:
        # Try 320, 325, 330...
        candidate_ns = [320, 325, 330, 335, 340]
    else:
        # Try to beat n_luw
        candidate_ns = [n_luw - 5, n_luw - 10, n_luw - 20]
    
    # Filter valid n (must be even? No, n*k must be even. k=4 is even, so n can be odd? No, sum of degrees is 2E. n*4 is always even. So n can be any integer.)
    # But simple graph constraints.
    
    for n in sorted(candidate_ns):
        if time.time() > deadline:
            break
            
        # Initialize
        edges, adj = generate_random_k_regular(n, k)
        if edges is None:
            continue
        
        # Calculate initial girth
        current_girth = fast_girth_check(n, adj, g)
        
        # SA
        T = 1.0
        cooling = 0.999
        steps = 5000 # Adjust based on time
        best_girth = current_girth
        best_edges = edges[:]
        
        # Precompute list of edges for faster access?
        # We need to pick two random edges.
        # edges is a list of tuples.
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            # Pick two random edges
            # To avoid bias, we can pick random indices.
            e1_idx = random.randint(0, len(edges)-1)
            e2_idx = random.randint(0, len(edges)-1)
            if e1_idx == e2_idx:
                continue
            
            u1, v1 = edges[e1_idx]
            u2, v2 = edges[e2_idx]
            
            # Check if they are disjoint
            if len({u1, v1, u2, v2}) < 4:
                continue
            
            # Try two possible swaps:
            # 1. (u1, u2) and (v1, v2)
            # 2. (u1, v2) and (v1, u2)
            
            # We must ensure new edges are not already present and not loops.
            
            def try_swap(new_e1, new_e2):
                a1, b1 = new_e1
                a2, b2 = new_e2
                if a1 == b1 or a2 == b2:
                    return False
                if b1 in adj[a1] or b2 in adj[a2]:
                    return False
                # Apply
                adj[a1].remove(v1)
                adj[v1].remove(a1)
                adj[a2].remove(v2)
                adj[v2].remove(a2)
                
                adj[a1].add(b1)
                adj[b1].add(a1)
                adj[a2].add(b2)
                adj[b2].add(a2)
                
                # Update edges list?
                # We can just keep a set of edges for validity, but we need the list for picking.
                # It's easier to maintain the list.
                # But finding and replacing in a list is O(N).
                # Let's just keep a set of edges and a list of edges.
                # Or better: Just use the adjacency matrix/set for the graph structure, and generate the edge list at the end.
                # For picking random edges, we can pick a random node, then a random neighbor?
                # That picks a random edge incident to a random node.
                # This is a valid way to pick a random edge (uniform over edges if we pick node uniformly? No, degree is constant k, so picking node uniformly and neighbor uniformly is uniform over edges).
                
                return True
            
            # We need a faster way to pick edges.
            # Let's switch to a representation where we can pick edges easily.
            # Since k is small, we can just pick a random node u, and a random neighbor v.
            # This gives a random edge (u,v).
            
            # Let's restart the SA loop with this insight.
            break # Break out of step loop to restructure? No, just return.
            
        # The implementation above is getting messy. Let's rewrite the SA part cleanly.
        pass

    # Clean implementation of SA
    
    def run_sa(n, k, g, time_limit):
        # Generate random k-regular graph
        # Using the method above
        for _ in range(100):
            stubs = []
            for i in range(n):
                stubs.extend([i] * k)
            random.shuffle(stubs)
            adj = [set() for _ in range(n)]
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v or v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            if valid:
                break
        
        if not valid:
            return None, None
        
        # Convert to list of edges for output
        def get_edges(adj):
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return edges
        
        current_girth = fast_girth_check(n, adj, g)
        if current_girth >= g:
            return n, get_edges(adj)
        
        T = 1.0
        cooling = 0.9995
        deadline_sa = time.time() + time_limit
        
        while time.time() < deadline_sa:
            # Pick two random edges
            # Pick random node u, random neighbor v -> edge (u,v)
            u1 = random.randint(0, n-1)
            v1 = random.choice(list(adj[u1]))
            
            u2 = random.randint(0, n-1)
            v2 = random.choice(list(adj[u2]))
            
            # Check disjoint
            if len({u1, v1, u2, v2}) < 4:
                continue
            
            # Try swap 1: (u1, u2) and (v1, v2)
            # Check if (u1, u2) and (v1, v2) are valid new edges
            if u2 not in adj[u1] and v2 not in adj[v1]:
                # Calculate new girth?
                # This is the expensive part.
                # We can approximate or just calculate.
                
                # Apply
                adj[u1].remove(v1)
                adj[v1].remove(u1)
                adj[u2].remove(v2)
                adj[v2].remove(u2)
                
                adj[u1].add(u2)
                adj[u2].add(u1)
                adj[v1].add(v2)
                adj[v2].add(v1)
                
                new_girth = fast_girth_check(n, adj, g)
                
                # Metropolis criterion
                # We want to maximize girth.
                # delta = new_girth - current_girth
                delta = new_girth - current_girth
                if delta > 0 or random.random() < np.exp(delta / T):
                    current_girth = new_girth
                else:
                    # Revert
                    adj[u1].remove(u2)
                    adj[u2].remove(u1)
                    adj[v1].remove(v2)
                    adj[v2].remove(v1)
                    
                    adj[u1].add(v1)
                    adj[v1].add(u1)
                    adj[u2].add(v2)
                    adj[v2].add(u2)
                
                T *= cooling
                
                if current_girth >= g:
                    return n, get_edges(adj)
            
            # Try swap 2: (u1, v2) and (v1, u2)
            if v2 not in adj[u1] and u2 not in adj[v1]:
                adj[u1].remove(v1)
                adj[v1].remove(u1)
                adj[u2].remove(v2)
                adj[v2].remove(u2)
                
                adj[u1].add(v2)
                adj[v2].add(u1)
                adj[v1].add(u2)
                adj[u2].add(v1)
                
                new_girth = fast_girth_check(n, adj, g)
                
                delta = new_girth - current_girth
                if delta > 0 or random.random() < np.exp(delta / T):
                    current_girth = new_girth
                else:
                    adj[u1].remove(v2)
                    adj[v2].remove(u1)
                    adj[v1].remove(u2)
                    adj[u2].remove(v1)
                    
                    adj[u1].add(v1)
                    adj[v1].add(u1)
                    adj[u2].add(v2)
                    adj[v2].add(u2)
                
                T *= cooling
                
                if current_girth >= g:
                    return n, get_edges(adj)

        # Return best found (which might not be valid)
        # We should only return if valid.
        if current_girth >= g:
            return n, get_edges(adj)
        return None, None

    # Execute SA for candidate sizes
    # We want to minimize n.
    # So we try smaller n first?
    # But smaller n is harder.
    # If we try n=320 and fail, we try n=330?
    # If we find a valid graph at n=330, it's better than n_luw (if n_luw > 330).
    # But n=320 is better.
    
    # Strategy:
    # 1. Try n=320 (if n_luw > 320) with 50% of time.
    # 2. If fail, try n=330, 340... with remaining time.
    # 3. Return best valid graph found.
    
    # But the current code structure returns immediately if found.
    # We should collect the best result.
    
    best_found_n = n_luw
    best_found_edges = edges_luw
    
    # Let's just try a few sizes.
    # Given the complexity, let's try n=320 first.
    
    if n_luw > 320:
        time_per_trial = 20 # 20 seconds per n
        for n in [320, 325, 330]:
            if time.time() > deadline:
                break
            res_n, res_edges = run_sa(n, k, g, time_per_trial)
            if res_n is not None:
                best_found_n = res_n
                best_found_edges = res_edges
                break # Found a valid one, stop? Or try smaller?
                # We tried 320 first. If it works, it's the best.
                # If 320 fails, we try 325. 325 is worse than 320 but better than n_luw.
    
    # If we found a valid graph, return it.
    # Otherwise return LUW.
    
    if best_found_n < n_luw:
        return best_found_n, best_found_edges
    else:
        return n_luw, edges_luw

