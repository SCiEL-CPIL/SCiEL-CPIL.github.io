import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj, n

    def popcount(x):
        return bin(x).count('1')

    def find_maximal_cliques(adj, n):
        """
        Bron-Kerbosch algorithm with pivot to find all maximal cliques.
        Returns a list of frozensets of vertices.
        """
        cliques = []
        
        def bron_kerbosch(R, P, X):
            if time.time() > deadline:
                return
            
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Pivot selection: choose u in P union X with max degree in P
            # To speed up, we can just pick any, or the one with max neighbors in P
            # Standard optimization: pick u in P U X such that |N(u) intersect P| is max.
            # If P is empty, we are done.
            
            if not P:
                return

            # Choose pivot u
            # We want to minimize the number of branches.
            # Branch on vertices in P \ N(u).
            
            # Simple pivot: first vertex in P
            # Better: vertex in P U X with max degree in P
            max_deg = -1
            pivot = -1
            # Iterate P U X
            # Since n is small, we can just scan
            candidates = list(P.union(X)) if X else list(P)
            if not candidates:
                return
            
            for u in candidates:
                deg = len(P.intersection(adj[u]))
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
            
            # Branch on P \ N(pivot)
            # P_minus_N = P - N(pivot)
            # Note: N(pivot) in the context of the graph induced by P?
            # Bron-Kerbosch: Branch on v in P \ N(u).
            # Here N(u) means neighbors of u in the *original* graph, intersected with P?
            # Yes, the recursion removes neighbors from P.
            
            # We need to iterate over v in P such that v is NOT a neighbor of pivot.
            # v is in P \ N(pivot).
            
            P_copy = set(P)
            for v in list(P_copy):
                if (adj[pivot] >> v) & 1:
                    P_copy.discard(v)
            
            # Now iterate over v in P_copy
            for v in list(P_copy):
                # Recurse
                bron_kerbosch(R.union({v}), P.intersection(adj[v]), X.intersection(adj[v]))
                # Update P and X
                P.discard(v)
                X.add(v)
                # Check time
                if time.time() > deadline:
                    return

        # Initial call: R=empty, P=all vertices, X=empty
        bron_kerbosch(set(), set(range(n)), set())
        return cliques

    def solve_hitting_set(cliques, n, time_limit):
        """
        Finds a minimum hitting set for the given list of cliques (frozensets).
        Uses Greedy Max Coverage + Iterative Removal.
        """
        if not cliques:
            return []
        
        # Map clique index to frozenset
        # Greedy:
        # 1. Count occurrences of each vertex in all cliques.
        # 2. Pick vertex with max count.
        # 3. Remove all cliques containing that vertex.
        # 4. Update counts.
        # 5. Repeat.
        
        # To handle ties and get better solutions, we can use a slight randomization or just deterministic.
        # Deterministic is safer for reproducibility, but random restarts might find better.
        # Given time constraints, let's try a few random restarts if time allows, 
        # but start with a strong deterministic greedy.
        
        best_solution = None
        best_size = n + 1
        
        # We will run a few variations of greedy
        # Variation 1: Standard Max Count
        # Variation 2: Max Count / Size of Clique (Weighted) - not standard for hitting set, usually just count.
        # Let's stick to Max Count, but break ties randomly in a few runs.
        
        num_restarts = 10
        # Allocate time per restart
        time_per_restart = time_limit / num_restarts
        
        for r in range(num_restarts):
            if time.time() > start_time + time_limit:
                break
            
            # State:
            # uncovered_cliques: list of indices of cliques not yet hit
            # current_solution: set of vertices selected
            # vertex_counts: dict or list of counts of how many times vertex appears in uncovered_cliques
            
            uncovered = list(range(len(cliques)))
            solution = set()
            
            # Pre-compute vertex -> list of clique indices containing it?
            # Or just compute counts on the fly?
            # For n=40, len(cliques) might be a few hundred/thousand.
            # Computing counts from scratch each step is O(V * C).
            # V=40, C=1000 -> 40k ops per step. Steps <= 40. Total 1.6M ops. Fast.
            
            while uncovered:
                # Calculate counts
                counts = [0] * n
                for idx in uncovered:
                    c = cliques[idx]
                    for v in c:
                        counts[v] += 1
                
                # Find max count
                max_count = max(counts)
                if max_count == 0:
                    # Should not happen if uncovered is not empty
                    break
                
                # Find all vertices with max_count
                candidates = [v for v in range(n) if counts[v] == max_count]
                
                # Pick one
                # Randomly pick to allow exploration in restarts
                v = random.choice(candidates)
                
                solution.add(v)
                
                # Remove all cliques containing v
                new_uncovered = []
                for idx in uncovered:
                    if v not in cliques[idx]:
                        new_uncovered.append(idx)
                uncovered = new_uncovered
            
            # Now we have a valid hitting set 'solution'
            # Perform Removal (Local Search)
            # Try to remove each vertex if possible
            changed = True
            while changed:
                changed = False
                # Iterate over a copy of solution
                for v in list(solution):
                    # Check if removing v is safe
                    # v is safe if every clique containing v is also contained in solution - {v}
                    # i.e., for every clique C containing v, there exists w in C \ {v} such that w in solution.
                    
                    # Optimization: Pre-compute which cliques are "only" covered by v?
                    # If a clique is covered ONLY by v, we cannot remove v.
                    
                    # Let's check all cliques.
                    can_remove = True
                    for c in cliques:
                        if v in c:
                            # Check if any other vertex in c is in solution
                            others_in_sol = False
                            for u in c:
                                if u != v and u in solution:
                                    others_in_sol = True
                                    break
                            if not others_in_sol:
                                can_remove = False
                                break
                    
                    if can_remove:
                        solution.discard(v)
                        changed = True
            
            if len(solution) < best_size:
                best_size = len(solution)
                best_solution = set(solution)
                
                # If we found a very small solution, we can stop early?
                # Maybe not, but it's a good heuristic.
            
            # Optional: Swap move?
            # Try to swap one vertex for another?
            # Since we are minimizing size, swaps don't change size.
            # Only removals reduce size.
            # So if we can't remove, we are stuck in a local minimum of size.
            # To escape, we might need to add a vertex to allow removing two?
            # That's complex. Greedy + Removal is usually sufficient for these sizes.
            
        if best_solution is None:
            # Fallback: Return all vertices
            return list(range(n))
            
        return list(best_solution)

    out = []
    # Process each instance
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    # Allocate time budget
    # 110s total.
    # Graph generation is fast.
    # Clique enumeration is the bottleneck for dense graphs, but p is low (0.2-0.3).
    # Hitting set solving is fast.
    
    for i, (n, p, seed) in enumerate(instances):
        if time.time() > deadline:
            # Fallback for remaining instances
            # Just return a greedy vertex cover (which is valid but suboptimal)
            adj, n_val = generate_graph(n, p, seed)
            # Greedy VC
            U = (1 << n_val) - 1
            cover = set()
            while U:
                best_v = -1
                max_deg = -1
                temp_U = U
                while temp_U:
                    lsb = temp_U & -temp_U
                    v = (lsb.bit_length() - 1)
                    deg = popcount(adj[v] & U)
                    if deg > max_deg:
                        max_deg = deg
                        best_v = v
                    temp_U -= lsb
                if best_v == -1: break
                cover.add(best_v)
                U &= ~(adj[best_v] | (1 << best_v))
            out.append(sorted(cover))
            continue
            
        adj, n_val = generate_graph(n, p, seed)
        
        # Find maximal cliques
        # Use a time limit for clique finding
        clique_time_limit = 20 # 20 seconds per instance for clique finding
        cliques = find_maximal_cliques(adj, n_val)
        
        # Solve hitting set
        # Remaining time
        remaining = deadline - time.time()
        if remaining < 5:
             # Not enough time, use greedy VC fallback
             U = (1 << n_val) - 1
             cover = set()
             while U:
                best_v = -1
                max_deg = -1
                temp_U = U
                while temp_U:
                    lsb = temp_U & -temp_U
                    v = (lsb.bit_length() - 1)
                    deg = popcount(adj[v] & U)
                    if deg > max_deg:
                        max_deg = deg
                        best_v = v
                    temp_U -= lsb
                if best_v == -1: break
                cover.add(best_v)
                U &= ~(adj[best_v] | (1 << best_v))
             out.append(sorted(cover))
             continue
             
        hs_solution = solve_hitting_set(cliques, n_val, remaining)
        out.append(sorted(hs_solution))

    return out
