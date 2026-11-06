import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Target n. We aim for 35. If we can't find it fast, we might try 34, 33...
    # But the task says "Maximize n". The seed returns 17 as fallback.
    # The record is 35.
    # We will try to solve for n=35.
    
    n = 35
    num_edges = n * (n - 1) // 2
    
    # Edge indexing: e_{ij} for i < j
    # Map (i, j) to edge index
    edge_id = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_id[(i, j)] = idx
            idx += 1
            
    def get_edge_id(i, j):
        if i > j: i, j = j, i
        if i == j: return -1
        return edge_id.get((i, j), -1)

    # SAT Variables: 0..num_edges-1 represent x_{ij} = 1 (edge exists)
    
    # Clauses:
    # 1. K4-free: For every set of 4 vertices {a,b,c,d}, NOT (x_ab AND x_ac AND x_ad AND x_bc AND x_bd AND x_cd)
    #    Clause: (NOT x_ab OR NOT x_ac OR NOT x_ad OR NOT x_bc OR NOT x_bd OR NOT x_cd)
    #    In CNF, this is a single clause with 6 negative literals.
    
    # 2. No independent set of size 6 (alpha <= 5):
    #    For every set of 6 vertices {a,b,c,d,e,f}, NOT (NOT x_ab AND NOT x_ac AND ... AND NOT x_ef)
    #    This is equivalent to: For every 6-set, at least one edge must exist.
    #    Clause: (x_ab OR x_ac OR ... OR x_ef)
    #    This is a single clause with 15 positive literals.
    
    # Building all clauses explicitly might be memory heavy if we store them all,
    # but n=35 is small enough.
    # Number of 4-sets: C(35, 4) = 52365
    # Number of 6-sets: C(35, 6) = 1,623,160
    # Total clauses: ~1.7 million.
    # Storing 1.7 million lists of 6 or 15 ints is feasible in Python (maybe ~100MB?).
    # Let's try to generate them on the fly or store efficiently.
    
    # Efficient DPLL implementation
    # We will use a recursive DPLL with unit propagation and pure literal elimination.
    
    # To handle the large number of I6 clauses (15 literals each), we need to be careful with propagation.
    # However, we can pre-compute the clauses.
    
    # Optimization:
    # Instead of storing all clauses, we can generate K4 clauses (short) and handle I6 clauses
    # by checking if the current assignment forces an independent set.
    # But standard DPLL works on clauses.
    
    # Let's try to build the clause database.
    # Clause format: list of (var, sign). sign=1 for positive, -1 for negative.
    # var is 0..num_edges-1.
    
    clauses = []
    
    # 1. K4 Clauses
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        # Edges: ab, ac, ad, bc, bd, cd
        e1 = get_edge_id(a, b)
        e2 = get_edge_id(a, c)
        e3 = get_edge_id(a, d)
        e4 = get_edge_id(b, c)
        e5 = get_edge_id(b, d)
        e6 = get_edge_id(c, d)
        # Clause: !e1 or !e2 or !e3 or !e4 or !e5 or !e6
        clauses.append([(-e1, -1), (-e2, -1), (-e3, -1), (-e4, -1), (-e5, -1), (-e6, -1)])
        
    # 2. I6 Clauses
    # For every 6-set, at least one edge must be present.
    for sext in combinations(range(n), 6):
        # 15 edges in a K6
        edges_in_sext = []
        verts = list(sext)
        for i in range(6):
            for j in range(i+1, 6):
                e = get_edge_id(verts[i], verts[j])
                edges_in_sext.append(e)
        # Clause: e1 or e2 or ... or e15
        clause = [(e, 1) for e in edges_in_sext]
        clauses.append(clause)
        
    # Total clauses ~ 1.7M. This might be slow to initialize in Python.
    # C(35,6) is 1.6M. Creating 1.6M lists of 15 tuples...
    # Time estimate: 1.6M * 15 * ~100ns = 2.4s? Plus list overhead. Maybe 5-10s.
    # Acceptable within 150s.
    
    # DPLL Solver
    # State: assignment dict {var: value}, value is 1 (True) or 0 (False) or None (Unassigned)
    # We will use a simple recursive DPLL with backtracking.
    # To make it faster, we will implement:
    # 1. Unit Propagation
    # 2. Pure Literal Elimination (optional, might be slow with 1.7M clauses)
    # 3. Variable Ordering: Choose variable involved in most clauses (or just sequential with randomization)
    
    # Given the density, a smarter approach might be better.
    # But let's try standard DPLL first.
    
    # Actually, storing 1.7M clauses and iterating through them for unit propagation
    # at every node of the search tree will be extremely slow in Python.
    # Each propagation step is O(N_clauses * Clause_Length).
    # If we do this thousands of times, it's too slow.
    
    # Alternative Strategy:
    # Use the SAT solver concept but with a more efficient representation or a different search.
    # What if we use a **Backtracking Search with Constraint Propagation** but optimized for this specific structure?
    
    # Let's rethink. The "SAT" idea is good, but naive DPLL in Python on 1.7M clauses is likely too slow.
    # We need a more efficient solver or a different encoding.
    
    # Efficient Encoding for I6:
    # Instead of 1.6M clauses, can we reduce them?
    # No, they are all distinct constraints.
    
    # Alternative: **Local Search with SAT-based Repair**
    # 1. Start with a random graph.
    # 2. Find violations.
    # 3. Instead of random flips, use a SAT solver to find a minimal set of edge flips to resolve the current violations?
    #    This is still hard.
    
    # Let's go back to **Simulated Annealing** but with a **much better cost function** and **neighborhood**.
    # The previous program failed likely due to poor cost function or getting stuck.
    # Cost function: Number of K4s + Number of I6s.
    # We can count these incrementally?
    # Counting K4s is O(n^4) or O(n^3) with bitmasks.
    # Counting I6s is hard.
    
    # However, we can use a **Penalty Method** with a fast approximation.
    # Or, we can use **Min-Conflict Algorithm** style:
    # 1. Identify all violated constraints (K4s and I6s).
    # 2. For each violated constraint, pick a random edge to flip to satisfy it.
    # 3. If a flip satisfies one but breaks another, that's fine, we just minimize total conflicts.
    
    # This is essentially what the seed program tried ("Greedy Repair"), but it failed.
    # Why? Maybe because flipping one edge in a K4 (removing it) can create new I6s, and flipping one edge in an I6 (adding it) can create new K4s.
    # The "conflict graph" is dense.
    
    # New Strategy: **Tabu Search**
    # 1. Maintain a tabu list of recently flipped edges.
    # 2. Move to the best neighbor (minimizing total violations) that is not tabu.
    # 3. To find the best neighbor, we evaluate all possible single-edge flips.
    #    There are ~600 edges.
    #    Evaluating a flip requires updating the violation count.
    #    If we can update violation counts incrementally, this is fast.
    
    # Let's try to implement **Incremental Violation Counting**.
    
    # State:
    # adj: n x n matrix
    # k4_count: int
    # i6_count: int (This is the hard one)
    
    # Counting I6s incrementally is very hard.
    # Counting K4s is easier.
    
    # What if we only track K4s and use a separate mechanism for I6s?
    # Or, we accept that we can't count I6s fast, so we use a **Sampling** approach for I6s in the cost function?
    # No, we need exact validity.
    
    # Let's try a different structural change: **Genetic Algorithm**
    # 1. Population of graphs.
    # 2. Fitness: 0 if valid, else - (num_K4s + num_I6s_sampled).
    # 3. Crossover: Uniform crossover of edge bits.
    # 4. Mutation: Random edge flips.
    # 5. Selection: Tournament.
    
    # This is a different design family (Evolutionary vs Local Search vs SAT).
    # It explores the space more broadly.
    
    # Let's implement a simple Genetic Algorithm.
    # To make it fast:
    # - Use numpy arrays for genomes.
    # - Vectorized fitness evaluation?
    #   Checking validity is the bottleneck.
    #   We can use a fast bitset-based validity check.
    
    def check_validity_fast(n, adj):
        # adj is numpy array 0/1
        # Check K4
        masks = [0] * n
        for i in range(n):
            m = 0
            row = adj[i]
            for j in range(n):
                if row[j]:
                    m |= (1 << j)
            masks[i] = m
            
        # K4 Check
        for i in range(n):
            mi = masks[i]
            for j in range(i + 1, n):
                if mi & (1 << j):
                    mj = masks[j]
                    common = mi & mj
                    common &= ~(1 << i)
                    common &= ~(1 << j)
                    
                    c_mask = common
                    while c_mask:
                        lsb = c_mask & -c_mask
                        k = lsb.bit_length() - 1
                        c_mask ^= lsb
                        rest = common & ~(1 << k)
                        if masks[k] & rest:
                            return False, 'K4'
                            
        # I6 Check
        comp_masks = [0] * n
        for i in range(n):
            m = 0
            for j in range(n):
                if i != j and not adj[i][j]:
                    m |= (1 << j)
            comp_masks[i] = m
            
        found = [False]
        
        def dfs(current_set, candidates, depth):
            if found[0]:
                return
            if depth == 6:
                found[0] = True
                return
            if bin(candidates).count('1') < (6 - depth):
                return
                
            c_mask = candidates
            while c_mask:
                if found[0]:
                    return
                lsb = c_mask & -c_mask
                v = lsb.bit_length() - 1
                c_mask ^= lsb
                new_candidates = c_mask & comp_masks[v]
                dfs(current_set | lsb, new_candidates, depth + 1)
                
        dfs(0, (1 << n) - 1, 0)
        if found[0]:
            return False, 'I6'
            
        return True, None

    # GA Parameters
    pop_size = 100
    generations = 0
    best_fitness = -1
    best_adj = None
    
    # Initialize Population
    population = []
    for _ in range(pop_size):
        # Random graph with p=0.5
        adj = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i][j] = 1
                    adj[j][i] = 1
        population.append(adj)
        
    # To speed up, we can try a few random restarts of a local search within the GA?
    # Or just pure GA.
    
    # Fitness Function
    def fitness(adj):
        valid, _ = check_validity_fast(n, adj)
        if valid:
            return 1.0
        # Penalty: count violations roughly
        # We can return 0 for invalid to keep it simple, but then GA has no gradient.
        # Let's try to count K4s and I6s (sampled?)
        # For now, just 0/1. GA might struggle without gradient.
        # Let's add a small penalty for "density deviation" from optimal?
        # Optimal density for R(4,6) is around 0.5?
        # Let's just use 0 for invalid.
        return 0.0

    # Since 0/1 fitness is binary, GA will only select valid individuals.
    # If no valid individuals exist, it will stall.
    # We need a gradient.
    
    # Let's compute a "score" based on the number of violations found by a quick check.
    # But counting I6s is hard.
    
    # Alternative: **Memetic Algorithm** (GA + Local Search)
    # 1. GA to explore.
    # 2. Local search to exploit.
    
    # Given the time limit, let's stick to a robust **Simulated Annealing** with **Restart**
    # but with a **better initial state** and **smarter moves**.
    
    # Why did the seed fail?
    # It used "Greedy Repair" which is deterministic and gets stuck.
    # SA with random restarts is better.
    
    # Let's implement a high-performance SA.
    # Key: Fast validity check.
    # We can cache the masks?
    
    # Let's try to optimize the validity check further.
    # Pre-compute masks from adj.
    # The check is O(n^3) roughly.
    # For n=35, n^3 = 42875. Very fast.
    # The I6 DFS might be slow if it explores many branches.
    # But if the graph is "random", I6s are likely, so DFS finds one fast.
    # If the graph is "close to valid", I6s are rare, DFS explores more.
    # This is the "needle in a haystack" problem.
    
    # To help, we can use **Constraint Propagation** in the I6 check?
    # Or, we can use **Random Walk** on the space of valid graphs?
    
    # Let's try a different approach: **Constructive Heuristic**
    # Start with n=1. Add vertices one by one.
    # When adding vertex v, choose its edges to existing vertices such that:
    # 1. No K4 is formed.
    # 2. No I6 is formed.
    # This is a constraint satisfaction problem for each new vertex.
    # We can use a SAT solver for the sub-problem of adding one vertex?
    # Or just greedy backtracking.
    
    # Let's try **Greedy Construction with Backtracking**.
    # 1. Start with empty graph.
    # 2. Add vertex 0. No edges.
    # 3. Add vertex 1. Choose edge (0,1) randomly.
    # 4. Add vertex 2. Choose edges (0,2), (1,2) such that no K4 (impossible yet) and no I6 (impossible yet).
    # ...
    # 5. Add vertex k.
    #    For each existing vertex i, decide if (i,k) is an edge.
    #    Constraint 1: If i, j, l form a triangle in G[0..k-1], then k cannot be connected to all of them (K4).
    #    Constraint 2: If i, j, l, m, p form an independent set of size 5 in G[0..k-1], then k must be connected to at least one of them (I6).
    
    # This is a CSP for each new vertex.
    # We can solve it with a simple backtracking search or SAT.
    # Since the sub-problem is small (n-1 variables), it's fast.
    
    # Let's implement this!
    
    def construct_vertex(k, adj_prev, n_prev):
        """
        Determines edges from new vertex k to vertices 0..k-1.
        adj_prev is k x k matrix.
        Returns a list of edges (i, k) to add, or None if stuck.
        """
        # Constraints:
        # 1. For every triangle (i, j, l) in G[0..k-1], NOT (x_ik AND x_jk AND x_lk)
        # 2. For every independent set of size 5 (i, j, l, m, p) in G[0..k-1], (x_ik OR x_jk OR x_lk OR x_mk OR x_pk)
        
        # Find all triangles in G[0..k-1]
        triangles = []
        masks = [0] * n_prev
        for i in range(n_prev):
            m = 0
            for j in range(n_prev):
                if adj_prev[i][j]:
                    m |= (1 << j)
            masks[i] = m
            
        for i in range(n_prev):
            mi = masks[i]
            for j in range(i + 1, n_prev):
                if mi & (1 << j):
                    mj = masks[j]
                    common = mi & mj
                    common &= ~(1 << i)
                    common &= ~(1 << j)
                    c_mask = common
                    while c_mask:
                        lsb = c_mask & -c_mask
                        l = lsb.bit_length() - 1
                        c_mask ^= lsb
                        triangles.append((i, j, l))
                        
        # Find all independent sets of size 5 in G[0..k-1]
        # Independent set in G is Clique in Complement.
        comp_masks = [0] * n_prev
        for i in range(n_prev):
            m = 0
            for j in range(n_prev):
                if i != j and not adj_prev[i][j]:
                    m |= (1 << j)
            comp_masks[i] = m
            
        indep_5s = []
        
        def dfs_clique(current_set, candidates, depth, path):
            if depth == 5:
                indep_5s.append(path[:])
                return
            if bin(candidates).count('1') < (5 - depth):
                return
            c_mask = candidates
            while c_mask:
                lsb = c_mask & -c_mask
                v = lsb.bit_length() - 1
                c_mask ^= lsb
                path.append(v)
                new_candidates = c_mask & comp_masks[v]
                dfs_clique(current_set | lsb, new_candidates, depth + 1, path)
                path.pop()
                
        dfs_clique(0, (1 << n_prev) - 1, 0, [])
        
        # Now solve CSP for x_0, ..., x_{k-1}
        # Variables: 0..k-1
        # Clauses:
        # From triangles: (NOT x_i OR NOT x_j OR NOT x_l)
        # From indep_5s: (x_i OR x_j OR x_l OR x_m OR x_p)
        
        # Use DPLL for this small CSP
        # Number of vars: k (up to 34)
        # Number of clauses: Triangles + Indep5s
        
        # To speed up, we can use a simple backtracking with forward checking.
        
        # Let's use a randomized greedy approach first, and backtrack if stuck.
        
        def solve_csp():
            # Randomize variable order
            vars_to_assign = list(range(n_prev))
            random.shuffle(vars_to_assign)
            
            assignment = {}
            
            def backtrack(idx):
                if idx == len(vars_to_assign):
                    return assignment.copy()
                
                v = vars_to_assign[idx]
                
                # Try 1 then 0 (or random)
                # Check if 1 is consistent
                # Check if 0 is consistent
                
                # To be efficient, we can just try both and see if it leads to a dead end.
                # But checking consistency requires propagating.
                
                # Simple approach: Assign, check all clauses involving v?
                # This is O(N_clauses).
                
                # Let's just try random assignment and validate at the end?
                # No, that's too slow.
                
                # Let's use a simple recursive search with pruning.
                
                # For each value in [1, 0]:
                for val in [random.choice([0, 1]), 1 - random.choice([0, 1])]:
                    assignment[v] = val
                    
                    # Check if any clause is violated by this assignment (if all literals are assigned)
                    violated = False
                    for clause in csp_clauses:
                        # clause is list of (var, sign)
                        # If all vars in clause are assigned, check if satisfied
                        all_assigned = True
                        satisfied = False
                        for var, sign in clause:
                            if var not in assignment:
                                all_assigned = False
                                break
                            if (assignment[var] == 1 and sign == 1) or (assignment[var] == 0 and sign == -1):
                                satisfied = True
                                break
                        if all_assigned and not satisfied:
                            violated = True
                            break
                            
                    if not violated:
                        res = backtrack(idx + 1)
                        if res:
                            return res
                            
                    del assignment[v]
                return None
                
            return backtrack(0)
            
        # Build CSP Clauses
        csp_clauses = []
        for tri in triangles:
            i, j, l = tri
            csp_clauses.append([(i, -1), (j, -1), (l, -1)])
        for ind in indep_5s:
            csp_clauses.append([(v, 1) for v in ind])
            
        # Optimization: If there are too many clauses, it might be slow.
        # But for n=35, the number of triangles and indep_5s is manageable.
        
        sol = solve_csp()
        if sol is None:
            return None
            
        # Construct edges
        edges = []
        for v, val in sol.items():
            if val == 1:
                edges.append((v, k))
        return edges

    # Main Construction Loop
    adj = np.zeros((n, n), dtype=int)
    current_n = 0
    
    for k in range(n):
        if time.time() > deadline - 10:
            break
            
        if k == 0:
            current_n = 1
            continue
            
        # Get subgraph 0..k-1
        adj_prev = adj[:k, :k]
        
        # Try to add vertex k
        # Randomize the order of vertices in the subgraph? No, the structure is fixed.
        # But we can randomize the CSP solver's variable order.
        
        edges = construct_vertex(k, adj_prev, k)
        
        if edges is None:
            # Stuck!
            # We can try to restart the whole construction with a different random seed?
            # Or, we can try to "repair" the previous step?
            # For now, let's just break and return what we have.
            # But we want to maximize n.
            # If we get stuck at k=10, we return n=10.
            # This is bad.
            
            # Alternative: Use Backtracking in the Construction.
            # If we get stuck at k, we go back to k-1 and try a different edge configuration.
            
            # This is complex. Let's just do a few restarts of the whole construction.
            break
            
        # Add edges
        for u, v in edges:
            adj[u][v] = 1
            adj[v][u] = 1
            
        current_n = k + 1
        
    # Check if we reached n
    if current_n == n:
        # Verify
        valid, _ = check_validity_fast(n, adj)
        if valid:
            edges_out = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i][j]:
                        edges_out.append([i, j])
            return {'n': n, 'edges': edges_out}
            
    # If we failed, try a few more restarts with different random seeds?
    # Or fall back to a known valid graph.
    
    # Let's try 5 restarts of the constructive heuristic.
    
    best_n = 0
    best_adj = None
    
    for attempt in range(5):
        if time.time() > deadline - 5:
            break
            
        adj = np.zeros((n, n), dtype=int)
        current_n = 0
        success = True
        
        for k in range(n):
            if time.time() > deadline - 5:
                success = False
                break
                
            if k == 0:
                current_n = 1
                continue
                
            adj_prev = adj[:k, :k]
            edges = construct_vertex(k, adj_prev, k)
            
            if edges is None:
                success = False
                break
                
            for u, v in edges:
                adj[u][v] = 1
                adj[v][u] = 1
            current_n = k + 1
            
        if success and current_n > best_n:
            valid, _ = check_validity_fast(current_n, adj[:current_n, :current_n])
            if valid:
                best_n = current_n
                best_adj = adj[:current_n, :current_n].copy()
                
    if best_n > 0:
        edges_out = []
        for i in range(best_n):
            for j in range(i + 1, best_n):
                if best_adj[i][j]:
                    edges_out.append([i, j])
        return {'n': best_n, 'edges': edges_out}
        
    # Fallback: Paley 17
    n_p = 17
    squares = {x * x % n_p for x in range(1, n_p)}
    p_edges = [[i, j] for i in range(n_p) for j in range(i + 1, n_p) if (j - i) % n_p in squares]
    return {'n': n_p, 'edges': p_edges}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
