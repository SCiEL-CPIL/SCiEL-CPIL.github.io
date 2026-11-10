# EVOLVE-BLOCK-START
import random
import sys

def construct():
    def build_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch_maximal_cliques(adj):
        n = len(adj)
        maximal_cliques = []
        
        def expand(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            pivot_pool = P | X
            if not pivot_pool:
                return
            pivot = max(pivot_pool, key=lambda v: len(P & adj[v]))
            candidates = P - adj[pivot]
            
            for v in list(candidates):
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                expand(new_R, new_P, new_X)
                P.remove(v)
                X.add(v)
        
        expand(set(), set(range(n)), set())
        return maximal_cliques

    def greedy_cover(cliques, universe):
        if not cliques:
            return set()
        uncovered = list(cliques)
        selected = set()
        
        # Calculate frequency
        freq = {v: 0 for v in universe}
        for c in cliques:
            for v in c:
                freq[v] += 1
                
        while uncovered:
            # Pick vertex covering most uncovered cliques
            best_v = None
            best_count = -1
            for v in universe:
                if v in selected:
                    continue
                count = sum(1 for c in uncovered if v in c)
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v is None or best_count == 0:
                break
            selected.add(best_v)
            uncovered = [c for c in uncovered if best_v not in c]
        return selected

    def solve_instance(n, p, seed):
        adj = build_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(adj)
        
        if not cliques:
            return []
            
        universe = set()
        for c in cliques:
            universe.update(c)
            
        # Initial upper bound from greedy
        initial_sol = greedy_cover(cliques, universe)
        best_size = len(initial_sol)
        best_solution = set(initial_sol)
        
        # Pre-compute vertex to clique index mapping for quick filtering? 
        # In B&B we pass lists, so we just filter.
        
        def branching_bound(remaining_cliques, selected_count):
            nonlocal best_size, best_solution
            
            if selected_count >= best_size:
                return
                
            if not remaining_cliques:
                if selected_count < best_size:
                    best_size = selected_count
                    # We need to reconstruct the solution. 
                    # This is tricky in standard B&B without storing the path.
                    # We will store the path in the recursion state or backtrack.
                    # For simplicity and speed, let's modify the state to include the current selection set.
                    pass 
                return

        # Standard B&B without path reconstruction is hard to get the actual set.
        # Let's use a different strategy: 
        # 1. Get a good upper bound (greedy).
        # 2. Use B&B to find if a smaller solution exists.
        # 3. If yes, we need the actual set.
        
        # Let's implement B&B that tracks the selection.
        
        def bb_search(remaining_cliques, current_selection):
            nonlocal best_size, best_solution
            
            if len(current_selection) >= best_size:
                return
                
            if not remaining_cliques:
                if len(current_selection) < best_size:
                    best_size = len(current_selection)
                    best_solution = set(current_selection)
                return
                
            # Lower Bound: Disjoint cliques
            # Find max number of disjoint cliques in remaining_cliques
            # This is NP-hard too, but for small sets, a greedy packing is a good LB.
            used_vertices = set()
            lb = 0
            # Sort cliques by size descending to pack larger ones first? 
            # Actually, just greedy pick any, remove intersecting.
            remaining_copy = list(remaining_cliques)
            while remaining_copy:
                # Pick a clique with min size to ensure we can pack more? 
                # Or just pick first.
                c = remaining_copy.pop(0)
                # Check if it intersects with used? 
                # We want disjoint cliques.
                # If c intersects used, we can't use it for this specific packing step?
                # No, we are just finding a lower bound.
                # A valid packing is a set of pairwise disjoint cliques.
                # Greedy: pick c, add to packing, remove all cliques intersecting c from remaining_copy.
                lb += 1
                new_remaining = []
                for rc in remaining_copy:
                    if not (rc & c):
                        new_remaining.append(rc)
                remaining_copy = new_remaining
                
            if len(current_selection) + lb >= best_size:
                return

            # Essential Vertices
            # If a vertex appears in exactly one remaining clique, it must be selected.
            # We can detect this by counting occurrences.
            # However, doing this every step is expensive.
            # Let's just branch on a clique.
            
            # Select a clique to branch on. 
            # Heuristic: Pick the clique with the smallest number of vertices? 
            # Or the one with the most "critical" vertices?
            # Smallest clique reduces branching factor.
            target_clique = min(remaining_cliques, key=len)
            
            for v in target_clique:
                # Branch: include v
                new_remaining = [c for c in remaining_cliques if v not in c]
                # Optimization: If new_remaining is same as remaining_cliques, v was not in any? Impossible, v is in target_clique which is in remaining.
                bb_search(new_remaining, current_selection | {v})
                
            # Note: We don't branch on "exclude v" explicitly because we MUST cover target_clique.
            # So we MUST pick at least one v from target_clique.
            # This is a valid reduction for Hitting Set.

        # Run B&B
        # To avoid recursion depth issues, n is small (40), depth is <= 40.
        bb_search(cliques, set())
        
        if best_solution:
            return sorted(best_solution)
        else:
            return sorted(initial_sol)

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        solution = solve_instance(n, p, seed)
        out.append(solution)
        
    return out
# EVOLVE-BLOCK-END
