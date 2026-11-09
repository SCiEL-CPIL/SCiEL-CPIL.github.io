"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s kill
    
    N = 256
    
    # Precompute prime factors for 1..N
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        if not prime_factors[i]:
            for j in range(i, N + 1, i):
                prime_factors[j].add(i)
    # 1 has no prime factors, prime_factors[1] remains empty
    
    # Precompute compatibility matrix
    compat = [[True] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            c = not (prime_factors[a] & prime_factors[b])
            compat[a][b] = c
            compat[b][a] = c
    
    def is_compatible(a, b):
        return compat[a][b]
    
    # Primary approach: Maximum Weight Independent Set via NetworkX
    try:
        import networkx as nx
        
        G = nx.Graph()
        G.add_nodes_from(range(1, N + 1))
        
        for a in range(1, N + 1):
            for b in range(a + 1, N + 1):
                if is_compatible(a, b):
                    G.add_edge(a, b)
        
        for node in G.nodes:
            G.nodes[node]['weight'] = node
        
        clique = nx.max_weight_clique(G, weight='weight')
        
        if clique:
            return [int(x) for x in clique]
    except Exception:
        pass
    
    # Fallback: greedy + local search with multiple strategies
    def greedy_solution():
        sol = [1]
        for a in range(N, 1, -1):
            if all(is_compatible(a, b) for b in sol):
                sol.append(a)
        return sol
    
    def local_search(sol, time_limit):
        solution = list(sol)
        current_sum = sum(solution)
        solution_set = set(solution)
        
        def can_replace(a, b):
            for c in solution_set:
                if c == b:
                    continue
                if not is_compatible(a, c):
                    return False
            return True
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Strategy 1: Single swap (replace one element with a larger one)
            for i in range(len(solution)):
                a = solution[i]
                for b in range(N, a, -1):
                    if b in solution_set:
                        continue
                    if can_replace(b, a):
                        solution_set.remove(a)
                        solution_set.add(b)
                        solution[i] = b
                        current_sum += b - a
                        improved = True
                        break
            
            # Strategy 2: Two-for-one removal (remove two elements, add one larger)
            if time.time() < time_limit:
                for i in range(len(solution)):
                    for j in range(i + 1, len(solution)):
                        a1 = solution[i]
                        a2 = solution[j]
                        for b in range(N, max(a1, a2), -1):
                            if b in solution_set:
                                continue
                            # Check if b is compatible with all remaining elements
                            compatible = True
                            for c in solution_set:
                                if c == a1 or c == a2:
                                    continue
                                if not is_compatible(b, c):
                                    compatible = False
                                    break
                            if compatible:
                                gain = b - a1 - a2
                                if gain > 0:
                                    solution_set.remove(a1)
                                    solution_set.remove(a2)
                                    solution_set.add(b)
                                    solution.remove(a1)
                                    solution.remove(a2)
                                    solution.append(b)
                                    current_sum += gain
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
        
        return solution, current_sum
    
    # Try multiple random restarts
    best_solution = None
    best_sum = 0
    
    # Initial greedy solution
    sol0 = greedy_solution()
    sol0, sum0 = local_search(sol0, deadline - 5)
    best_solution = sol0
    best_sum = sum0
    
    # Randomized restarts
    num_restarts = 0
    while time.time() < deadline - 10:
        # Create a randomized initial solution
        sol = [1]
        candidates = list(range(2, N + 1))
        random.shuffle(candidates)
        
        for a in candidates:
            if all(is_compatible(a, b) for b in sol):
                sol.append(a)
        
        sol, s = local_search(sol, deadline - 10)
        
        if s > best_sum:
            best_sum = s
            best_solution = sol
        
        num_restarts += 1
        # Limit restarts to avoid excessive time
        if num_restarts >= 20:
            break
    
    return [int(x) for x in best_solution]
# EVOLVE-BLOCK-END
