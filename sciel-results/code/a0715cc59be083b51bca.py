import time
import random
import itertools
from collections import deque

# EVOLVE-BLOCK-START

def _build_graph(n, p, seed):
    """Build adjacency sets from the same PRNG stream as the seed program."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _bron_kerbosch_max_cliques(adj, min_size=2):
    """Enumerate all maximal cliques of size >= min_size using Bron-Kerbosch with pivot."""
    n = len(adj)
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= min_size:
                cliques.append(frozenset(R))
            return
        # Choose pivot u from P union X to reduce recursion
        U = P | X
        if not U:
            return
        # Pick pivot with max degree in P (heuristic)
        pivot = max(U, key=lambda u: len(P & adj[u]))
        candidates = P - adj[pivot]
        for v in list(candidates):
            R.add(v)
            P_new = P & adj[v]
            X_new = X & adj[v]
            bk(R, P_new, X_new)
            R.remove(v)
            P.discard(v)
            X.add(v)
    
    bk(set(), set(range(n)), set())
    return cliques

def _solve_instance(adj, time_limit=40.0):
    """Solve the minimum transversal problem for maximal cliques using Branch and Bound with improvements."""
    n = len(adj)
    cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
    
    if not cliques:
        return []
    
    num_cliques = len(cliques)
    
    # Precompute vertex -> clique indices
    vertex_cliques = [[] for _ in range(n)]
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_cliques[v].append(i)
    
    # Bitmask representation
    # clique_mask[v] has bit i set if v is in clique i
    clique_mask = [0] * n
    for v in range(n):
        mask = 0
        for i in vertex_cliques[v]:
            mask |= (1 << i)
        clique_mask[v] = mask
        
    # Sort vertices by coverage (number of cliques) descending to help B&B ordering
    vertex_freq = [len(vertex_cliques[v]) for v in range(n)]
    
    # Precompute for each clique: list of vertices in it, sorted by frequency descending
    clique_vertices = []
    for i, clique in enumerate(cliques):
        verts = sorted(clique, key=lambda v: vertex_freq[v], reverse=True)
        clique_vertices.append(verts)
        
    # Precompute clique sizes
    clique_sizes = [len(c) for c in cliques]
    
    best_solution = None
    best_size = num_cliques + 1 # Upper bound
    
    # Multiple greedy strategies for better initial upper bound
    def greedy_solve(seed=0):
        rng = random.Random(seed)
        S = set()
        uncovered = (1 << num_cliques) - 1
        while uncovered:
            best_v = -1
            best_count = -1
            # Add some randomness to break ties
            for v in range(n):
                if v in S:
                    continue
                count = bin(clique_mask[v] & uncovered).count('1')
                # Add small random perturbation to break ties
                perturbed = count + rng.random() * 0.1
                if perturbed > best_count:
                    best_count = perturbed
                    best_v = v
            if best_v == -1 or best_count == 0:
                break
            S.add(best_v)
            uncovered &= ~clique_mask[best_v]
        return S if not uncovered else None
    
    # Try multiple greedy restarts
    for attempt in range(10):
        S_greedy = greedy_solve(seed=attempt)
        if S_greedy is not None and len(S_greedy) < best_size:
            best_size = len(S_greedy)
            best_solution = set(S_greedy)
    
    # Also try a different greedy: prioritize vertices that cover rare cliques
    def greedy_rare_cliques():
        S = set()
        uncovered = (1 << num_cliques) - 1
        # Count how many vertices can cover each clique
        clique_coverage_count = [0] * num_cliques
        for v in range(n):
            for i in range(num_cliques):
                if clique_mask[v] & (1 << i):
                    clique_coverage_count[i] += 1
        
        while uncovered:
            best_v = -1
            best_score = -1
            for v in range(n):
                if v in S:
                    continue
                # Score: sum of 1/coverage_count for uncovered cliques covered by v
                score = 0.0
                for i in range(num_cliques):
                    if (clique_mask[v] & (1 << i)) and (uncovered & (1 << i)):
                        score += 1.0 / clique_coverage_count[i]
                if score > best_score:
                    best_score = score
                    best_v = v
            if best_v == -1 or best_score == 0:
                break
            S.add(best_v)
            uncovered &= ~clique_mask[best_v]
        return S if not uncovered else None
    
    S_rare = greedy_rare_cliques()
    if S_rare is not None and len(S_rare) < best_size:
        best_size = len(S_rare)
        best_solution = set(S_rare)
    
    # B&B Search with improved clique selection
    start_time = time.time()
    
    # Precompute for each vertex, the list of cliques it covers (for faster lower bound)
    vertex_covers = [clique_mask[v] for v in range(n)]
    
    def fast_lower_bound(uncovered_mask):
        """Fast lower bound: max number of uncovered cliques any single vertex can cover."""
        if not uncovered_mask:
            return 0
        max_cover = 0
        for v in range(n):
            c = bin(clique_mask[v] & uncovered_mask).count('1')
            if c > max_cover:
                max_cover = c
        return max_cover
    
    def packing_lower_bound(uncovered_mask):
        """Tighter lower bound using packing argument."""
        if not uncovered_mask:
            return 0
        remaining = uncovered_mask
        packing = 0
        while remaining:
            # Pick the smallest uncovered clique
            min_clique = None
            min_size = 1000
            um = remaining
            while um:
                bit = um & -um
                idx = bit.bit_length() - 1
                if clique_sizes[idx] < min_size:
                    min_size = clique_sizes[idx]
                    min_clique = idx
                um ^= bit
            if min_clique is None:
                break
            packing += 1
            # Remove all cliques that share a vertex with min_clique
            clique_set = cliques[min_clique]
            new_remaining = 0
            for i in range(num_cliques):
                if remaining & (1 << i):
                    shares = False
                    for v in clique_set:
                        if clique_mask[v] & (1 << i):
                            shares = True
                            break
                    if not shares:
                        new_remaining |= (1 << i)
            remaining = new_remaining
        return packing
    
    def dfs(current_size, uncovered_mask, path):
        nonlocal best_size, best_solution
        
        if time.time() - start_time > time_limit:
            return False # Timeout
        
        if uncovered_mask == 0:
            if current_size < best_size:
                best_size = current_size
                best_solution = set(path)
            return True # Success
            
        # Pruning with lower bound - use fast bound first, then packing if needed
        lb_fast = fast_lower_bound(uncovered_mask)
        if current_size + lb_fast >= best_size:
            return False
        
        if current_size >= best_size:
            return False
        
        # Use packing bound for tighter pruning
        lb_pack = packing_lower_bound(uncovered_mask)
        lb = max(lb_fast, lb_pack)
        if current_size + lb >= best_size:
            return False
            
        # Pick the most constrained uncovered clique (smallest size)
        target_clique_idx = -1
        min_cand = 1000
        um = uncovered_mask
        while um:
            bit = um & -um
            idx = bit.bit_length() - 1
            num_cand = clique_sizes[idx]
            if num_cand < min_cand:
                min_cand = num_cand
                target_clique_idx = idx
            um ^= bit
        
        if target_clique_idx == -1:
            target_clique_idx = (uncovered_mask & -uncovered_mask).bit_length() - 1
        
        # Get vertices in this clique, sorted by how many uncovered cliques they cover
        verts = clique_vertices[target_clique_idx]
        # Sort by coverage of uncovered cliques, descending
        verts_sorted = sorted(verts, key=lambda v: bin(clique_mask[v] & uncovered_mask).count('1'), reverse=True)
        
        # Try each vertex in the clique
        for v in verts_sorted:
            new_uncovered = uncovered_mask & ~clique_mask[v]
            
            path.append(v)
            dfs(current_size + 1, new_uncovered, path)
            path.pop()
            
            # If we found a solution with size <= current_size, we can prune
            if best_size <= current_size:
                return False
                
    dfs(0, (1 << num_cliques) - 1, [])
    
    # Local search polish: try to remove vertices and swap vertices
    if best_solution is not None and time.time() - start_time < time_limit - 2:
        polished = set(best_solution)
        improved = True
        while improved and time.time() - start_time < time_limit - 1:
            improved = False
            # Try removing vertices
            for v in list(polished):
                candidate = polished - {v}
                # Check if candidate still covers all cliques
                ok = True
                for i, clique in enumerate(cliques):
                    if not (clique & candidate):
                        ok = False
                        break
                if ok:
                    polished = candidate
                    improved = True
                    best_size = len(polished)
                    best_solution = set(polished)
                    break  # Restart the loop with the new solution
            
            # Try swap moves: remove one, add another
            if not improved:
                for v_remove in list(polished):
                    for v_add in range(n):
                        if v_add in polished:
                            continue
                        candidate = (polished - {v_remove}) | {v_add}
                        if len(candidate) >= len(polished):
                            continue
                        # Check if candidate still covers all cliques
                        ok = True
                        for i, clique in enumerate(cliques):
                            if not (clique & candidate):
                                ok = False
                                break
                        if ok:
                            polished = candidate
                            improved = True
                            best_size = len(polished)
                            best_solution = set(polished)
                            break
                    if improved:
                        break
    
    if best_solution is not None:
        return sorted(best_solution)
    else:
        # Fallback to greedy if B&B timed out without finding anything better
        for attempt in range(5):
            S_greedy = greedy_solve(seed=100 + attempt)
            if S_greedy is not None:
                return sorted(S_greedy)
        # Last resort: return all vertices
        return list(range(n))

def construct():
    deadline = time.time() + 145  # 5s safety margin
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    # Allocate more time to harder instances (larger n tends to be harder)
    time_weights = [0.20, 0.35, 0.45]
    
    for idx, (n, p, seed) in enumerate(instances):
        remaining_time = deadline - time.time() - 3  # 3s safety margin
        if remaining_time < 5:
            # Fallback: simple greedy
            adj = _build_graph(n, p, seed)
            cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
            if not cliques:
                results.append([])
                continue
            # Simple greedy
            S = set()
            is_covered = [False] * len(cliques)
            vertex_cliques = [[] for _ in range(n)]
            for i, c in enumerate(cliques):
                for v in c:
                    vertex_cliques[v].append(i)
                    
            remaining = list(range(len(cliques)))
            while remaining:
                best_v = -1
                best_count = -1
                for v in range(n):
                    if v in S: continue
                    cnt = sum(1 for ci in vertex_cliques[v] if not is_covered[ci])
                    if cnt > best_count:
                        best_count = cnt
                        best_v = v
                if best_v == -1: break
                S.add(best_v)
                for ci in vertex_cliques[best_v]:
                    if not is_covered[ci]:
                        is_covered[ci] = True
                        remaining.remove(ci)
            results.append(sorted(S))
            continue
            
        time_budget = remaining_time * time_weights[idx]
        adj = _build_graph(n, p, seed)
        sol = _solve_instance(adj, time_limit=time_budget)
        results.append(sol)
        
    return results

# EVOLVE-BLOCK-END
