"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for overhead/serialization
    
    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # If we run out of time, return empty lists for remaining to avoid exception
            # But we should try to return valid results if possible.
            # Given the small size, this is unlikely.
            results.append([])
            continue
            
        # 1. Build Graph
        adj = [set() for _ in range(n)]
        rng = random.Random(seed)
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        
        # 2. Find Maximal Cliques
        maximal_cliques = []
        def bron_kerbosch(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            
            # Choose pivot
            if P:
                # Pick pivot with max degree in P
                u = max(P, key=lambda x: len(adj[x] & P))
                candidates = P - adj[u]
            else:
                candidates = P
                
            for v in candidates:
                bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
                
        bron_kerbosch(set(), set(range(n)), set())
        
        if not maximal_cliques:
            results.append([])
            continue
            
        # 3. Solve Hitting Set
        # Map cliques to indices for bitmasking if count is small
        num_cliques = len(maximal_cliques)
        
        # If number of cliques is very large, greedy might be needed, 
        # but for n<=40, p<=0.3, num_cliques is usually small.
        
        # Pre-compute coverage: for each vertex, which cliques does it cover?
        vertex_coverage = [0] * n
        for i, clique in enumerate(maximal_cliques):
            mask = 1 << i
            for v in clique:
                vertex_coverage[v] |= mask
                
        # Greedy Solution for upper bound
        uncovered = (1 << num_cliques) - 1
        greedy_sol = []
        temp_uncovered = uncovered
        while temp_uncovered:
            best_v = -1
            best_count = -1
            for v in range(n):
                if not (vertex_coverage[v] & temp_uncovered):
                    continue
                count = bin(vertex_coverage[v] & temp_uncovered).count('1')
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v == -1:
                break
            greedy_sol.append(best_v)
            temp_uncovered &= ~vertex_coverage[best_v]
            
        best_solution_size = len(greedy_sol)
        best_solution = set(greedy_sol)
        
        # Branch and Bound with proper solution reconstruction
        # State: uncovered_mask
        # We want to minimize number of vertices.
        memo = {}
        
        def solve(uncovered_mask, current_size, current_set):
            if uncovered_mask == 0:
                return 0, set()
            if current_size >= best_solution_size:
                return float('inf'), set()
                
            if uncovered_mask in memo:
                return memo[uncovered_mask]
                
            # Choose a clique to branch on (e.g., one with fewest available vertices)
            # Find any set bit
            clique_idx = (uncovered_mask & -uncovered_mask).bit_length() - 1
            clique = maximal_cliques[clique_idx]
            
            min_cost = float('inf')
            min_set = set()
            
            # Try each vertex in this clique
            for v in clique:
                # If v is already in current solution? 
                # We are building a set, so we don't track "current solution" explicitly in mask, 
                # just the size. 
                # However, if we pick v, we cover all cliques v covers.
                new_uncovered = uncovered_mask & ~vertex_coverage[v]
                cost, sub_set = solve(new_uncovered, current_size + 1, current_set)
                if cost < min_cost:
                    min_cost = cost
                    min_set = {v} | sub_set
                    if min_cost == 0: # Should not happen if uncovered != 0
                        break
                        
            memo[uncovered_mask] = (min_cost, min_set)
            return min_cost, min_set

        # Run B&B
        # To avoid stack overflow or excessive recursion depth, 
        # and to respect time limit, we can limit recursion or time.
        # Given small N, recursion depth is bounded by best_solution_size.
        
        try:
            bb_cost, bb_set = solve(uncovered, 0, set())
            if bb_cost < best_solution_size:
                greedy_sol = list(bb_set)
                best_solution_size = bb_cost
        except RecursionError:
            pass
        except Exception:
            pass

        # Local Search: Try to swap or remove vertices.
        # 1. Removal: Can we remove any vertex from greedy_sol?
        changed = True
        while changed:
            changed = False
            for i in range(len(greedy_sol)):
                v = greedy_sol[i]
                new_sol = greedy_sol[:i] + greedy_sol[i+1:]
                # Check if new_sol covers all cliques
                covered = 0
                for u in new_sol:
                    covered |= vertex_coverage[u]
                if covered == uncovered:
                    greedy_sol = new_sol
                    changed = True
                    break
        
        # 2. Swap: Try to swap one vertex with another not in sol?
        # This is more complex. 
        # Given the time limit and small problem size, the greedy + removal 
        # is likely sufficient to get the optimal or near-optimal answer.
        # The "score" is the total size.
        
        results.append(sorted(greedy_sol))
        
    return results

# EVOLVE-BLOCK-END
