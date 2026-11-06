"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def construct():
    deadline = time.time() + 85
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute LCM for all pairs
    lcm = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            lcm[i][j] = i * j // gcd(i, j)
    
    # Precompute prime factorizations
    def prime_factors(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors
    
    factorizations = [None] * (N + 1)
    for i in range(1, N + 1):
        factorizations[i] = prime_factors(i)
    
    # Build conflict structure: for each element x, store which pairs (y,z) form a bad triple
    conflicts = defaultdict(set)
    
    # Iterate over all triples a<b<c and check if they're bad
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            L_ab = lcm[a][b]
            for c in range(b+1, N+1):
                if L_ab % c != 0:
                    continue
                if a * c // gcd(a, c) == L_ab and b * c // gcd(b, c) == L_ab:
                    conflicts[a].add(frozenset([b, c]))
                    conflicts[b].add(frozenset([a, c]))
                    conflicts[c].add(frozenset([a, b]))
    
    # Compute conflict degree
    degree = [0] * (N+1)
    for x in range(1, N+1):
        degree[x] = len(conflicts.get(x, set()))
    
    # Build adjacency: for each x, set of y that appear in any conflict pair with x
    adj = defaultdict(set)
    for x in range(1, N+1):
        for pair in conflicts.get(x, set()):
            for y in pair:
                adj[x].add(y)
                adj[y].add(x)
    
    # Find connected components of the conflict graph
    visited = set()
    components = []
    for x in range(1, N+1):
        if x not in visited:
            # BFS to find component
            comp = set()
            queue = [x]
            visited.add(x)
            while queue:
                u = queue.pop()
                comp.add(u)
                for v in adj.get(u, set()):
                    if v not in visited:
                        visited.add(v)
                        queue.append(v)
            components.append(comp)
    
    # Sort components by size (largest first)
    components.sort(key=len, reverse=True)
    
    best_set = set()
    best_size = 0
    
    # Function to check if x can be added to set S
    def can_add(x, S):
        for pair in conflicts.get(x, set()):
            if pair.issubset(S):
                return False
        return True
    
    # Branch-and-bound for a single component
    def solve_component(comp, time_budget):
        """Solve MIS on a single component using branch-and-bound with matching bound."""
        comp_list = sorted(comp, key=lambda x: -degree[x])
        comp_set = set(comp)
        
        # Precompute: for each x in comp, conflicts restricted to comp
        local_conflicts = {}
        for x in comp:
            local_conflicts[x] = set()
            for pair in conflicts.get(x, set()):
                if pair.issubset(comp_set):
                    local_conflicts[x].add(pair)
        
        # Local degree within component
        local_degree = {x: len(local_conflicts[x]) for x in comp}
        
        # Reorder by local degree
        comp_list.sort(key=lambda x: -local_degree[x])
        
        # For matching-based upper bound: compute max matching in the "conflict graph"
        # where two vertices are connected if they appear together in some conflict triple
        local_adj = {x: set() for x in comp}
        for x in comp:
            for pair in local_conflicts[x]:
                for y in pair:
                    local_adj[x].add(y)
        
        def max_matching_bound(vertices):
            """Compute maximum matching size in the induced subgraph on vertices."""
            if not vertices:
                return 0
            # Use simple greedy matching (not optimal but fast upper bound)
            matched = set()
            matching_size = 0
            v_list = list(vertices)
            random.shuffle(v_list)
            for u in v_list:
                if u in matched:
                    continue
                for v in local_adj.get(u, set()):
                    if v in vertices and v not in matched and v > u:
                        matched.add(u)
                        matched.add(v)
                        matching_size += 1
                        break
            # Upper bound: vertices not in matching can all be taken, matched vertices at most 1 per edge
            # Actually, the bound is: |V| - max_matching (each matched edge forces at least one exclusion)
            return len(vertices) - matching_size
        
        best_comp = set()
        best_comp_size = 0
        nodes = 0
        start_time = time.time()
        
        # Use iterative deepening or simple B&B
        def bbb(idx, S, remaining_set):
            nonlocal best_comp, best_comp_size, nodes
            
            if time.time() - start_time > time_budget:
                return
            
            nodes += 1
            
            if not remaining_set:
                if len(S) > best_comp_size:
                    best_comp_size = len(S)
                    best_comp = set(S)
                return
            
            # Upper bound using matching
            ub = len(S) + max_matching_bound(remaining_set)
            if ub <= best_comp_size:
                return
            
            # Pick vertex with highest degree in remaining
            x = max(remaining_set, key=lambda v: len([p for p in local_conflicts.get(v, set()) if p.issubset(remaining_set)]) if local_conflicts.get(v) else 0)
            
            new_remaining = remaining_set - {x}
            
            # Option 1: Include x
            can_include = True
            for pair in local_conflicts.get(x, set()):
                if pair.issubset(S):
                    can_include = False
                    break
            
            if can_include:
                S.add(x)
                bbb(idx + 1, S, new_remaining)
                S.remove(x)
            
            # Option 2: Exclude x
            bbb(idx + 1, S, new_remaining)
        
        remaining = set(comp)
        bbb(0, set(), remaining)
        
        return best_comp, best_comp_size
    
    # Solve each component independently
    for i, comp in enumerate(components):
        if time.time() > deadline - 15:
            break
        
        # Allocate time budget based on component size
        time_budget = min(3.0, (deadline - time.time() - 15) / max(1, len(components) - i))
        
        comp_set, comp_size = solve_component(comp, time_budget)
        best_set.update(comp_set)
    
    # If we didn't get a good solution from decomposition, try global greedy + local search
    if best_size < 50 or time.time() < deadline - 20:
        # Global greedy with multiple restarts
        for r in range(200):
            if time.time() > deadline - 10:
                break
            
            S = set()
            order = elements[:]
            random.shuffle(order)
            
            for x in order:
                if x not in S and can_add(x, S):
                    S.add(x)
            
            # Local search
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                S_list = list(S)
                random.shuffle(S_list)
                
                for u in S_list:
                    if time.time() > deadline - 5:
                        break
                    
                    S.remove(u)
                    best_completion = set(S)
                    best_completion_size = len(S)
                    
                    for trial in range(3):
                        if time.time() > deadline - 5:
                            break
                        S_trial = set(S)
                        candidates = [x for x in range(1, N+1) if x not in S_trial]
                        random.shuffle(candidates)
                        
                        for x in candidates:
                            if can_add(x, S_trial):
                                S_trial.add(x)
                        
                        if len(S_trial) > best_completion_size:
                            best_completion_size = len(S_trial)
                            best_completion = set(S_trial)
                    
                    S = set(best_completion)
                    
                    if len(S) > len(best_set):
                        best_set = set(S)
                        improved = True
                        break
        
        # Final polish
        if time.time() < deadline - 3:
            for x in elements:
                if x not in best_set and can_add(x, best_set):
                    best_set.add(x)
    
    if not best_set:
        best_set = {1}
        
    return list(best_set)
# EVOLVE-BLOCK-END
