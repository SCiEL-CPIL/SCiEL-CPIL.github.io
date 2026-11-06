import time
import math
import random
from math import gcd
from collections import deque

def construct():
    start_time = time.time()
    N = 512
    TIME_LIMIT = 85  # seconds, leave 5s margin

    def check_deadline():
        return (time.time() - start_time) > TIME_LIMIT

    # Precompute LCM compatibility matrix
    compat = [[False] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            l = i // g * j
            if l <= N:
                compat[i][j] = True
                compat[j][i] = True
            else:
                compat[i][j] = False
                compat[j][i] = False

    # Precompute divisors for each number
    divisors = [set() for _ in range(N + 1)]
    for d in range(1, N + 1):
        for m in range(d, N + 1, d):
            divisors[m].add(d)

    def max_closure(weights, edges):
        """
        Maximum Weight Closure in a DAG via min-cut.
        weights: dict node -> weight
        edges: list of (u, v) meaning if u is selected, v must be selected
        Returns: set of nodes in the maximum weight closed set
        """
        nodes = list(weights.keys())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        n_nodes = len(nodes)
        
        # Build flow network
        # Source -> node with capacity = weight if weight > 0
        # Node -> sink with capacity = -weight if weight < 0
        # Edge u->v with infinite capacity if u selected implies v selected
        
        INF = float('inf')
        total_pos = sum(w for w in weights.values() if w > 0)
        
        # Adjacency list for residual graph
        # Each edge: [to, cap, rev_index]
        graph = [[] for _ in range(n_nodes + 2)]
        source = n_nodes
        sink = n_nodes + 1
        
        def add_edge(u, v, cap):
            graph[u].append([v, cap, len(graph[v])])
            graph[v].append([u, 0, len(graph[u]) - 1])
        
        # Add edges from source to nodes with positive weight
        for node in nodes:
            idx = node_to_idx[node]
            w = weights[node]
            if w > 0:
                add_edge(source, idx, w)
            elif w < 0:
                add_edge(idx, sink, -w)
        
        # Add closure constraint edges (infinite capacity)
        for u, v in edges:
            if u in node_to_idx and v in node_to_idx:
                add_edge(node_to_idx[u], node_to_idx[v], INF)
        
        # Max flow using Dinic's algorithm
        def max_flow():
            flow = 0
            while True:
                # BFS to find augmenting paths
                level = [-1] * (n_nodes + 2)
                level[source] = 0
                queue = deque([source])
                while queue:
                    u = queue.popleft()
                    for edge in graph[u]:
                        v, cap, rev = edge
                        if cap > 0 and level[v] < 0:
                            level[v] = level[u] + 1
                            queue.append(v)
                
                if level[sink] < 0:
                    break
                
                # DFS to find blocking flow
                def dfs(u, f):
                    if u == sink:
                        return f
                    for i in range(len(graph[u])):
                        v, cap, rev = graph[u][i]
                        if cap > 0 and level[v] == level[u] + 1:
                            pushed = dfs(v, min(f, cap))
                            if pushed > 0:
                                graph[u][i][1] -= pushed
                                graph[v][rev][1] += pushed
                                return pushed
                    return 0
                
                while True:
                    pushed = dfs(source, INF)
                    if pushed == 0:
                        break
                    flow += pushed
            
            return flow
        
        flow_value = max_flow()
        
        # Find min cut: nodes reachable from source in residual graph
        visited = [False] * (n_nodes + 2)
        visited[source] = True
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for edge in graph[u]:
                v, cap, rev = edge
                if cap > 0 and not visited[v]:
                    visited[v] = True
                    queue.append(v)
        
        # Nodes on source side of min cut are in the closure
        result = set()
        for node in nodes:
            idx = node_to_idx[node]
            if visited[idx]:
                result.add(node)
        
        return result

    # Strategy: Model as Maximum Weight Closure
    # Key insight: If we select a number x, we must also select all its divisors
    # (because lcm(d, x) = x <= 512 for any divisor d of x)
    # But we don't HAVE to select divisors of unselected numbers
    
    # The constraint is: if x and y are both selected, lcm(x,y) <= 512
    # This is equivalent to: for each prime power p^a in x and p^b in y, p^(max(a,b)) <= 512
    
    # Let's try a different formulation:
    # We want to find the largest set S such that for all x,y in S, lcm(x,y) <= 512
    
    # Observation: If S is valid, then for each x in S, all divisors of x are also compatible with all of S
    # (since if d|x, then lcm(d,y) divides lcm(x,y) <= 512)
    # So we can think of building S by deciding which "maximal" elements to include
    
    # Let's use the closure approach:
    # For each number x, define its "closure requirement" as all numbers y such that 
    # if x is in S, then y must also be in S for S to be valid
    # But this isn't quite right - the constraint is pairwise
    
    # Alternative: Let's just do local search but with better initialization
    # and use the closure idea to identify good seeds
    
    # First, let's find the optimal set using the divisor lattice structure
    # The optimal set should be closed under taking divisors
    
    def is_divisor_closed(S):
        """Check if S is closed under taking divisors"""
        for x in S:
            for d in divisors[x]:
                if d not in S:
                    return False
        return True

    def make_divisor_closed(S):
        """Add all divisors of elements in S"""
        result = set(S)
        changed = True
        while changed:
            changed = False
            for x in list(result):
                for d in divisors[x]:
                    if d not in result:
                        result.add(d)
                        changed = True
        return result

    # Precompute which sets of "top elements" give valid divisor-closed sets
    # A set S is valid iff for all x,y in S, lcm(x,y) <= 512
    
    # Let's try: pick a set of "maximal" elements M, then S = divisors(M)
    # S is valid iff for all x,y in M, lcm(x,y) <= 512 (and this implies all divisors work)
    
    # So we want to find the largest M such that all elements in M are pairwise compatible
    # and then S = divisors(M)
    
    # This is a maximum clique problem on the compatibility graph!
    # But the graph has 512 nodes, so we need heuristics
    
    # Let's use a greedy + local search approach for the clique, then take divisors
    
    def greedy_clique(compat, N):
        """Greedy maximum clique"""
        clique = set()
        candidates = list(range(1, N + 1))
        
        # Sort by degree in descending order
        candidates.sort(key=lambda x: sum(1 for y in candidates if compat[x][y]), reverse=True)
        
        for x in candidates:
            if x in clique:
                continue
            # Check if x is compatible with all in clique
            compatible = True
            for y in clique:
                if not compat[x][y]:
                    compatible = False
                    break
            if compatible:
                clique.add(x)
        
        return clique

    def local_search_clique(clique, compat, N):
        """Local search to improve clique size"""
        clique = set(clique)
        improved = True
        while improved and not check_deadline():
            improved = False
            # Try adding any element
            for x in range(1, N + 1):
                if x not in clique:
                    compatible = all(compat[x][y] for y in clique)
                    if compatible:
                        clique.add(x)
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try swaps: remove one, add one or more
            elems = list(clique)
            for u in elems:
                if check_deadline():
                    break
                trial = set(clique)
                trial.discard(u)
                # Try to add as many as possible
                for x in range(1, N + 1):
                    if x not in trial:
                        if all(compat[x][y] for y in trial):
                            trial.add(x)
                
                if len(trial) > len(clique):
                    clique = trial
                    improved = True
                    break
        
        return clique

    best_set = set()
    best_size = 0

    # Strategy 1: Greedy clique + local search
    if not check_deadline():
        clique = greedy_clique(compat, N)
        clique = local_search_clique(clique, compat, N)
        # Take divisor closure
        S = make_divisor_closed(clique)
        # Verify
        valid = True
        S_list = sorted(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                if not compat[S_list[i]][S_list[j]]:
                    valid = False
                    break
            if not valid:
                break
        if valid and len(S) > best_size:
            best_set = S
            best_size = len(S)

    # Strategy 2: Try different greedy orderings
    if not check_deadline():
        random.seed(42)
        for _ in range(20):
            if check_deadline():
                break
            # Random perturbation of greedy
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            # Sort by degree but with random tie-breaking
            candidates.sort(key=lambda x: (sum(1 for y in candidates if compat[x][y]), random.random()), reverse=True)
            
            clique = set()
            for x in candidates:
                if x in clique:
                    continue
                compatible = all(compat[x][y] for y in clique)
                if compatible:
                    clique.add(x)
            
            clique = local_search_clique(clique, compat, N)
            S = make_divisor_closed(clique)
            valid = True
            S_list = sorted(S)
            for i in range(len(S_list)):
                for j in range(i + 1, len(S_list)):
                    if not compat[S_list[i]][S_list[j]]:
                        valid = False
                        break
                if not valid:
                    break
            if valid and len(S) > best_size:
                best_set = S
                best_size = len(S)

    # Strategy 3: Start from highly composite numbers
    if not check_deadline():
        # Numbers with many divisors
        nums_by_divs = sorted(range(1, N + 1), key=lambda x: -len(divisors[x]))
        for m in nums_by_divs[:15]:
            if check_deadline():
                break
            # Start with divisors of m
            clique = set(divisors[m])
            clique = local_search_clique(clique, compat, N)
            S = make_divisor_closed(clique)
            valid = True
            S_list = sorted(S)
            for i in range(len(S_list)):
                for j in range(i + 1, len(S_list)):
                    if not compat[S_list[i]][S_list[j]]:
                        valid = False
                        break
                if not valid:
                    break
            if valid and len(S) > best_size:
                best_set = S
                best_size = len(S)

    # Final local search on best set
    if not check_deadline() and best_set:
        # Try to add more elements to best_set
        improved = True
        while improved and not check_deadline():
            improved = False
            for x in range(1, N + 1):
                if x not in best_set:
                    if all(compat[x][y] for y in best_set):
                        best_set.add(x)
                        improved = True
                        break

    # Verification
    final_list = sorted(best_set)
    valid = len(final_list) > 0
    for i in range(len(final_list)):
        for j in range(i + 1, len(final_list)):
            if not compat[final_list[i]][final_list[j]]:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback
        return list(range(1, int(math.isqrt(N)) + 1))

    return final_list
