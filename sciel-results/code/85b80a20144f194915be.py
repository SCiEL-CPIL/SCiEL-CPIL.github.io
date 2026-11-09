"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def generate_edges(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch_bitmask(adj_masks, n):
        """Bron-Kerbosch with pivot using bitmasks for speed."""
        cliques = []
        
        def bk(R, P, X):
            if time.time() > DEADLINE:
                return
            if P == 0 and X == 0:
                if R.bit_count() >= 2:
                    cliques.append(R)
                return
            
            # Choose pivot u from P|X with max |P & N(u)|
            PX = P | X
            if PX == 0:
                return
            
            best_u = -1
            best_count = -1
            tmp = PX
            while tmp:
                u = (tmp & (-tmp)).bit_length() - 1
                tmp &= tmp - 1
                count = (P & adj_masks[u]).bit_count()
                if count > best_count:
                    best_count = count
                    best_u = u
            
            # Iterate over P \ N(u)
            candidates = P & ~adj_masks[best_u]
            while candidates:
                v = (candidates & (-candidates)).bit_length() - 1
                candidates &= candidates - 1
                bk(R | (1 << v), P & adj_masks[v], X & adj_masks[v])
                P &= ~(1 << v)
                X |= (1 << v)
        
        bk(0, (1 << n) - 1, 0)
        return cliques

    def enumerate_maximal_cliques(n, edges):
        """Enumerate all maximal cliques of size >= 2 using Bron-Kerbosch with pivot."""
        adj_masks = [0] * n
        for u, v in edges:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        cliques_bitmasks = bron_kerbosch_bitmask(adj_masks, n)
        # Convert to lists of vertices
        cliques = []
        for bm in cliques_bitmasks:
            clique = []
            tmp = bm
            while tmp:
                v = (tmp & (-tmp)).bit_length() - 1
                tmp &= tmp - 1
                clique.append(v)
            cliques.append(clique)
        return cliques

    def greedy_hitting_set(cliques, n):
        """Fast greedy hitting set with bitmask coverage."""
        if not cliques:
            return []
        
        m = len(cliques)
        # For each vertex, bitmask of cliques it covers
        vertex_cover = [0] * n
        for i, clique in enumerate(cliques):
            bit = 1 << i
            for v in clique:
                vertex_cover[v] |= bit
        
        uncovered = (1 << m) - 1
        result = []
        
        while uncovered:
            best_v = -1
            best_count = 0
            for v in range(n):
                if uncovered == 0:
                    break
                count = (vertex_cover[v] & uncovered).bit_count()
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v == -1 or best_count == 0:
                break
            
            result.append(best_v)
            uncovered &= ~vertex_cover[best_v]
        
        return result

    def local_search_improve(solution, cliques, n, max_iterations=200):
        """Improve solution via removal operations."""
        if not solution:
            return solution
        
        sol_set = set(solution)
        
        def is_valid(s):
            for clique in cliques:
                found = False
                for v in clique:
                    if v in s:
                        found = True
                        break
                if not found:
                    return False
            return True
        
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try removing each vertex
            for v in list(sol_set):
                candidate = sol_set - {v}
                if is_valid(candidate):
                    sol_set = candidate
                    improved = True
                    break
        
        return sorted(sol_set)

    def branch_and_bound_hitting_set(cliques, n, time_limit=10.0):
        """Branch and bound for minimum hitting set with bitmask and stronger bounds."""
        start = time.time()
        best_solution = None
        best_size = float('inf')
        
        m = len(cliques)
        if m == 0:
            return []
        
        # Sort cliques by size (smallest first) for better pruning
        indexed_cliques = [(i, c) for i, c in enumerate(cliques)]
        indexed_cliques.sort(key=lambda x: len(x[1]))
        sorted_cliques = [c for _, c in indexed_cliques]
        
        # Precompute: for each vertex, bitmask of which sorted cliques it covers
        vertex_cover = [0] * n
        for i, clique in enumerate(sorted_cliques):
            bit = 1 << i
            for v in clique:
                vertex_cover[v] |= bit
        
        # Precompute vertex frequencies
        vertex_freq = [0] * n
        for clique in sorted_cliques:
            for v in clique:
                vertex_freq[v] += 1
        
        full_mask = (1 << m) - 1
        
        def lower_bound(remaining_mask, current_size):
            """Compute a lower bound using packing argument."""
            if remaining_mask == 0:
                return current_size
            
            # Find max coverage of any single vertex
            max_cov = 0
            for v in range(n):
                cov = (vertex_cover[v] & remaining_mask).bit_count()
                if cov > max_cov:
                    max_cov = cov
            
            if max_cov == 0:
                return current_size + remaining_mask.bit_count()
            
            return current_size + (remaining_mask.bit_count() + max_cov - 1) // max_cov
        
        def bnb(idx, current_mask, remaining_mask, current_size):
            nonlocal best_solution, best_size
            
            if time.time() - start > time_limit:
                return
            
            if remaining_mask == 0:
                if current_size < best_size:
                    best_size = current_size
                    best_solution = current_mask
                return
            
            # Lower bound check
            lb = lower_bound(remaining_mask, current_size)
            if lb >= best_size:
                return
            
            # Pick the next clique (smallest in sorted order)
            clique = sorted_cliques[idx]
            
            # Try vertices in this clique, ordered by frequency (most frequent first)
            vertices_in_clique = sorted(clique, key=lambda v: -vertex_freq[v])
            
            for v in vertices_in_clique:
                if time.time() - start > time_limit:
                    return
                
                new_mask = current_mask | (1 << v)
                new_remaining = remaining_mask & ~vertex_cover[v]
                # Skip if this vertex doesn't cover the current clique
                if (vertex_cover[v] & (1 << idx)) == 0:
                    continue
                bnb(idx + 1, new_mask, new_remaining, current_size + 1)
        
        bnb(0, 0, full_mask, 0)
        
        if best_solution is not None:
            result = []
            tmp = best_solution
            while tmp:
                v = (tmp & (-tmp)).bit_length() - 1
                tmp &= tmp - 1
                result.append(v)
            return result
        return greedy_hitting_set(cliques, n)

    def randomized_greedy(cliques, n, seed):
        """Randomized greedy with different orderings."""
        rng = random.Random(seed)
        if not cliques:
            return []
        
        m = len(cliques)
        vertex_cover = [0] * n
        for i, clique in enumerate(cliques):
            bit = 1 << i
            for v in clique:
                vertex_cover[v] |= bit
        
        uncovered = (1 << m) - 1
        result = []
        
        while uncovered:
            best_v = -1
            best_score = -1.0
            for v in range(n):
                count = (vertex_cover[v] & uncovered).bit_count()
                if count == 0:
                    continue
                score = count + rng.random() * 0.05
                if score > best_score:
                    best_score = score
                    best_v = v
            
            if best_v == -1:
                break
            
            result.append(best_v)
            uncovered &= ~vertex_cover[best_v]
        
        return result

    def find_min_hitting_set(cliques, n, time_budget=30.0):
        """Try multiple methods and return the best hitting set."""
        if not cliques:
            return []
        
        start = time.time()
        solutions = {}
        
        # Method 1: Greedy
        sol_greedy = greedy_hitting_set(cliques, n)
        solutions['greedy'] = sol_greedy
        
        # Method 2: Branch and bound (primary optimizer)
        elapsed = time.time() - start
        bnb_time = min(15.0, time_budget - elapsed - 5.0)
        if bnb_time > 2.0:
            sol_bnb = branch_and_bound_hitting_set(cliques, n, time_limit=bnb_time)
            solutions['bnb'] = sol_bnb
        
        # Method 3: Randomized greedy restarts
        elapsed = time.time() - start
        for restart_seed in range(10):
            if time.time() - start > time_budget - 3.0:
                break
            sol_rand = randomized_greedy(cliques, n, restart_seed)
            solutions[f'rand_{restart_seed}'] = sol_rand
        
        # Pick the best initial solution
        best = min(solutions.values(), key=len)
        
        # Local search improvement
        if time.time() - start < time_budget - 1.0:
            improved = local_search_improve(best, cliques, n, max_iterations=100)
            if len(improved) < len(best):
                best = improved
        
        return best

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > DEADLINE:
            # Fallback: return all vertices
            results.append(list(range(n)))
            continue
        
        edges = generate_edges(n, p, seed)
        cliques = enumerate_maximal_cliques(n, edges)
        
        if not cliques:
            results.append([])
            continue
        
        # Allocate time budget based on remaining time
        remaining_time = DEADLINE - time.time()
        time_budget = min(40.0, remaining_time * 0.9)
        
        hitting_set = find_min_hitting_set(cliques, n, time_budget=time_budget)
        results.append(hitting_set)
    
    return results
# EVOLVE-BLOCK-END
