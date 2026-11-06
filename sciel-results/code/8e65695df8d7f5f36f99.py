import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 115  # seconds; leave safety margin before 120s limit

    def check_triangle_free(n, adj):
        """Check if graph is triangle-free using adjacency sets."""
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    for k in adj[j]:
                        if k > j and k in adj[i]:
                            return False
        return True

    def count_triangles(n, adj):
        """Count triangles in O(n^3) or faster using sets."""
        cnt = 0
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    for k in adj[j]:
                        if k > j and k in adj[i]:
                            cnt += 1
        return cnt

    def max_independent_set_size(n, adj):
        """Exact branch and bound for maximum independent set size."""
        best = [0]
        
        # Precompute degrees for ordering
        # Use a simple greedy coloring lower bound for alpha? No, exact is needed for validity.
        # But for SA energy, we might use a fast heuristic.
        # For final validation, we use exact.
        
        def branch(remaining, size):
            if time.time() - start_time > DEADLINE - 5:
                return
            if not remaining:
                if size > best[0]:
                    best[0] = size
                return
            if size + len(remaining) <= best[0]:
                return
            
            # Pick vertex with max degree in remaining subgraph
            # Heuristic: pick vertex with max degree in the induced subgraph of 'remaining'
            r_list = list(remaining)
            # To speed up, just pick first or random? Max degree is better for pruning.
            # But calculating degree in remaining is O(|r|^2).
            # For small n (40), it's fine.
            v = max(r_list, key=lambda x: sum(1 for y in r_list if y != x and y in adj[x]))
            
            # Branch 1: include v
            new_remaining = set(r_list) - {v} - adj[v]
            branch(new_remaining, size + 1)
            
            # Branch 2: exclude v
            new_remaining2 = set(r_list) - {v}
            branch(new_remaining2, size)

        branch(set(range(n)), 0)
        return best[0]

    def is_valid(n, edges):
        """Check both conditions: no triangle and no independent set of 10."""
        if time.time() - start_time > DEADLINE - 5:
            return False
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        if not check_triangle_free(n, adj):
            return False
        alpha = max_independent_set_size(n, adj)
        return alpha < 10

    def edges_to_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def adj_to_edges(adj):
        edges = []
        n = len(adj)
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    edges.append([i, j])
        return edges

    def generate_random_triangle_free(n, seed):
        """Generate a random triangle-free graph."""
        random.seed(seed)
        adj = [set() for _ in range(n)]
        # Random permutation of vertices
        order = list(range(n))
        random.shuffle(order)
        # Add edges greedily if they don't form a triangle
        # This creates a random maximal triangle-free graph
        for i in order:
            for j in order:
                if i < j:
                    # Check if adding (i,j) creates a triangle
                    # Triangle if exists k such that (i,k) and (j,k) are edges
                    # We can check common neighbors
                    common = adj[i] & adj[j]
                    if not common:
                        adj[i].add(j)
                        adj[j].add(i)
        return adj

    def simulated_annealing(n, initial_adj, time_budget):
        """
        Optimize the graph to minimize triangles and independence number.
        Target: 0 triangles and alpha < 10.
        """
        end_time = time.time() + time_budget
        adj = [set(s) for s in initial_adj]
        
        # Calculate initial energy
        # Energy = triangles + penalty for alpha
        # We want to drive triangles to 0 and alpha to < 10.
        
        def get_energy(adj):
            tris = count_triangles(n, adj)
            # Approximate alpha for speed? No, exact is safer for guidance if fast enough.
            # For n=40, exact alpha is ~1ms.
            alpha = max_independent_set_size(n, adj)
            # Penalty: if alpha >= 10, penalize heavily.
            # If triangles > 0, penalize heavily.
            pen_alpha = 0 if alpha < 10 else (alpha - 9) ** 2 * 10
            pen_tris = tris * 100
            return pen_tris + pen_alpha, tris, alpha

        current_energy, cur_tris, cur_alpha = get_energy(adj)
        
        T = 10.0
        T_min = 0.01
        cooling = 0.995
        
        best_adj = [set(s) for s in adj]
        best_energy = current_energy
        
        while time.time() < end_time and T > T_min:
            # Pick a random edge to flip
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            adding = (j not in adj[i])
            
            # Calculate delta energy
            # This is expensive if we recompute full alpha.
            # Optimization:
            # 1. Check triangle impact locally.
            # 2. Only recompute alpha if triangle count changes significantly or randomly.
            
            common = adj[i] & adj[j]
            
            if adding:
                delta_tris = len(common)
                # If we add an edge, triangles increase by delta_tris.
                # Alpha might decrease or stay same.
                # We accept if delta_energy < 0 or with prob exp(-delta/T)
                # Approximate delta_alpha: hard.
                # Let's just compute full energy for simplicity, but it might be slow.
                # n=40, 50k steps * 1ms = 50s. Might be too slow.
                
                # Faster check:
                # If delta_tris > 0, energy increases by 100*delta_tris.
                # We should only add edges if it helps alpha enough.
                # Let's try to add edge and check alpha only if it's promising.
                
                # Heuristic: Only add edge if it doesn't create triangles OR if it reduces alpha.
                if delta_tris == 0:
                    adj[i].add(j)
                    adj[j].add(i)
                    # Check if this improves alpha
                    new_alpha = max_independent_set_size(n, adj)
                    if new_alpha < cur_alpha:
                        cur_alpha = new_alpha
                        cur_tris = 0 # still 0
                        current_energy = get_energy(adj)[0]
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_adj = [set(s) for s in adj]
                    else:
                        # Revert if no improvement in alpha? Or keep?
                        # Keeping might help future moves.
                        pass
                else:
                    # Creates triangles. Probably bad.
                    # Accept with low prob?
                    pass
            else:
                # Removing edge (i,j)
                # Triangles decrease by len(common).
                # Alpha might increase.
                # We should remove edges to reduce triangles.
                adj[i].discard(j)
                adj[j].discard(i)
                new_tris = count_triangles(n, adj)
                new_alpha = max_independent_set_size(n, adj)
                # If alpha becomes >= 10, it's bad.
                # If triangles become 0, it's good.
                
                new_energy, _, _ = get_energy(adj)
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_adj = [set(s) for s in adj]
                
                cur_tris = new_tris
                cur_alpha = new_alpha
                current_energy = new_energy

            T *= cooling

        # Return best found
        return best_adj, best_energy

    # --- Main Search ---
    
    # Try to find a valid graph for n=39
    best_n = 0
    best_edges = []
    
    # Fallback: Andrasfai graph on 26 vertices
    n1 = 26
    edges1 = [[i, j] for i in range(n1) for j in range(i + 1, n1) if (j - i) % n1 % 3 == 1]
    if is_valid(n1, edges1):
        best_n = n1
        best_edges = edges1
        print(f"Found valid graph with n={n1}")

    # Try Simulated Annealing for n=39
    target_n = 39
    if time.time() - start_time < DEADLINE - 60:
        print(f"Attempting SA for n={target_n}")
        # Generate a few initial graphs
        for attempt in range(5):
            if time.time() - start_time > DEADLINE - 10:
                break
            init_adj = generate_random_triangle_free(target_n, seed=100 + attempt)
            # Run SA with limited time
            time_left = DEADLINE - 10 - (time.time() - start_time)
            if time_left < 5:
                break
            sa_time = min(20, time_left / 2) # Spend up to 20s per attempt
            best_adj, best_energy = simulated_annealing(target_n, init_adj, sa_time)
            
            # Check if best_adj is valid
            edges_candidate = adj_to_edges(best_adj)
            if is_valid(target_n, edges_candidate):
                best_n = target_n
                best_edges = edges_candidate
                print(f"SA found valid graph with n={target_n}")
                break

    # If we found n=39, try to extend to 40?
    # Or just return 39 as it's a huge jump from 26.
    # R(3,10) <= 41 means max valid n is 40.
    # If we have time, try n=40.
    if best_n == 39 and time.time() - start_time < DEADLINE - 30:
        target_n = 40
        print(f"Attempting SA for n={target_n}")
        for attempt in range(3):
            if time.time() - start_time > DEADLINE - 10:
                break
            init_adj = generate_random_triangle_free(target_n, seed=200 + attempt)
            time_left = DEADLINE - 10 - (time.time() - start_time)
            if time_left < 5:
                break
            sa_time = min(10, time_left / 2)
            best_adj, best_energy = simulated_annealing(target_n, init_adj, sa_time)
            
            edges_candidate = adj_to_edges(best_adj)
            if is_valid(target_n, edges_candidate):
                best_n = target_n
                best_edges = edges_candidate
                print(f"SA found valid graph with n={target_n}")
                break

    if best_n == 0:
        # Fallback
        best_n = n1
        best_edges = edges1

    return {'n': best_n, 'edges': best_edges}
