import time
import random

def _build_adj(n, edges):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _edges_from_adj(adj, n):
    edges = []
    for u in range(n):
        m = adj[u]
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            if u < v:
                edges.append([u, v])
    return edges

def _is_valid_graph(n, edges):
    """
    Exact check for no K4 and no independent set of 6.
    Uses branch and bound with degree pruning for speed.
    """
    if not edges:
        return n <= 5 and n >= 1 # Trivial check for empty graph (though n>=1)
    
    adj = _build_adj(n, edges)
    
    # Check K4
    # A K4 exists if there is a triangle (u,v,w) such that u,v,w share a common neighbor x.
    # Equivalently, for any edge (u,v), if the common neighbors of u and v contain an edge (w,x), 
    # then {u,v,w,x} is a K4.
    # Faster: Iterate over triangles.
    # For each vertex u, look at its neighbors. If any two neighbors v,w are connected, 
    # and they share a common neighbor x (which must be in N(u) as well? No, x just needs to be connected to v and w).
    # Actually, a K4 is a set of 4 vertices where every pair is connected.
    # Let's check: For every pair (u, v) that is an edge, let C = N(u) & N(v).
    # If C contains an edge (w, x), then {u, v, w, x} is a K4.
    # This is O(E * degree^2) roughly.
    
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                common = adj[u] & adj[v]
                # Check if 'common' induces any edge
                # Iterate over vertices in common
                m = common
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    m ^= lsb
                    # Check if w is connected to any other vertex in common
                    # common & adj[w] will give neighbors of w that are also in common
                    if common & adj[w]:
                        return False
    
    # Check Independent Set of size 6
    # Use Bron-Kerbosch style or simple DFS with pruning.
    # Pruning: If the number of remaining vertices is less than needed, fail.
    # Also, if the maximum possible independent set size from remaining vertices is < needed, fail.
    # Upper bound on independent set size of a graph G: 
    # A simple bound is n, but we can use greedy coloring or just count.
    # For n<=40, a well-pruned DFS is usually fine.
    # We can use the bound: alpha(G) <= n - floor(sum(degrees)/2) is not tight enough.
    # Better: Greedy maximal independent set size is a lower bound on alpha.
    # Upper bound: Color the graph, alpha <= number of colors.
    # Let's use a simple recursive search with vertex ordering by degree.
    
    comp_adj = [0] * n
    full_mask = (1 << n) - 1
    for i in range(n):
        comp_adj[i] = full_mask ^ adj[i] ^ (1 << i)
        
    # Precompute degrees in complement for ordering?
    # Let's just do DFS with a simple bound: 
    # If we have 'depth' picked, and 'verts' available, we need 6-depth more.
    # If popcount(verts) < 6-depth, return False.
    # We can also use: if the induced subgraph on 'verts' has independence number < 6-depth, return False.
    # Computing independence number is hard, but for small 'verts' we can estimate.
    # For now, let's rely on the fact that n is small (<=40) and the graph is dense enough 
    # (since we are looking for Ramsey graphs, they are usually not too sparse).
    
    def dfs(verts, depth):
        if depth == 6:
            return True
        if not verts:
            return False
            
        # Pruning: if number of vertices < needed
        if verts.bit_count() < 6 - depth:
            return False
            
        # Order vertices by degree in complement (higher degree in complement = lower degree in original? 
        # Actually, in complement, we want to pick vertices that are non-adjacent in original.
        # So in complement, they are adjacent. We want to find a clique of size 6 in complement.
        # So we are looking for a K6 in comp_adj.
        
        # Heuristic: pick vertex with highest degree in complement first?
        # Or lowest? 
        # For finding a clique, picking high degree vertices first is usually good.
        
        # Let's just iterate.
        v = verts
        while v:
            lsb = v & -v
            u = lsb.bit_length() - 1
            v ^= lsb
            
            # New candidates must be neighbors of u in complement
            new_verts = v & comp_adj[u]
            
            if dfs(new_verts, depth + 1):
                return True
                
        return False
        
    return dfs(full_mask, 0)

def _paley_graph(p):
    """Generate Paley graph for prime p."""
    squares = {x * x % p for x in range(1, p)}
    edges = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                edges.append([i, j])
    return {'n': p, 'edges': edges}

def _clebsch_graph():
    """Generate the Clebsch graph: 16 vertices, K4-free, alpha=5."""
    n = 16
    vertices = []
    for i in range(16):
        bits = bin(i).count('1')
        if bits % 2 == 0:
            vertices.append(i)
    idx_map = {v: i for i, v in enumerate(vertices)}
    edges = []
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            diff = bin(vertices[i] ^ vertices[j]).count('1')
            if diff in (1, 3):
                edges.append([i, j])
    return {'n': n, 'edges': edges}

def _simulated_annealing(n, start_adj, time_limit, seed=None):
    """
    Simulated annealing to find a valid graph on n vertices.
    Start from a valid graph (e.g., Paley) and flip edges.
    """
    if seed is not None:
        random.seed(seed)
        
    start_time = time.time()
    deadline = start_time + time_limit
    
    adj = start_adj[:]
    
    # Ensure we start valid. If not, we might need to fix it or restart.
    # But we initialize from valid constructions.
    
    best_valid = None
    
    # Temperature schedule
    # Start high, decay.
    # Cost function: 
    # If valid: cost = 0 (or -1)
    # If invalid: cost = number of violations? 
    # Counting violations is expensive.
    # Alternative: 
    # Use a penalty function.
    # Let's try to minimize: (1 if has K4 else 0) + (1 if has Ind6 else 0).
    # This is binary. SA works better with gradients.
    
    # Better objective:
    # Minimize the size of the largest K4? No, we want 0.
    # Minimize the size of the largest independent set? No, we want < 6.
    
    # Let's use:
    # Score = 0 if valid.
    # Score = 1 + 0.1 * (number of K4s found in sample) + 0.1 * (number of Ind6s found in sample)
    # But counting is hard.
    
    # Alternative strategy:
    # Randomly flip an edge.
    # Check if new graph is valid.
    # If valid, accept.
    # If invalid, accept with probability exp(-delta / T).
    # What is delta?
    # If we go from Valid to Invalid, delta = 1.
    # If we go from Invalid to Valid, delta = -1 (always accept).
    # If we go from Invalid to Invalid, delta = 0? Or based on some metric?
    
    # Since most random flips will keep the graph invalid or make it more invalid,
    # we need a good initial state.
    # Paley(37) is valid? 
    # R(4,6) > 35. Paley(37) has n=37.
    # Is Paley(37) K4-free?
    # Paley graph P_q has clique number ~ sqrt(q) log q? 
    # For q=37, sqrt(37) ~ 6. 
    # Actually, Paley graphs are often used for Ramsey bounds.
    # Let's check if Paley(37) is valid.
    # If it is, we have n=37 immediately!
    
    # The current program checks Paley(37) first.
    # If Paley(37) is valid, the score is 37/17 > 2.
    # The current score is 1.29 (which is ~22 vertices).
    # Why didn't the current program find Paley(37)?
    # Maybe Paley(37) is NOT valid.
    # Let's assume Paley(37) is invalid (has K4 or Ind6).
    
    # So we start from a valid graph, say Paley(29) or Clebsch(16), and try to expand?
    # No, SA on fixed n.
    # We want to find a valid graph on n=35 or 36.
    # We can start from a random graph on n=35, but that's hard.
    # Better: Start from a valid graph on n=34 (if we can find one) or n=33.
    # Or start from Paley(37) and REMOVE edges to fix violations?
    
    # Let's try a different approach:
    # Start with a dense graph (e.g., complete graph) on n vertices.
    # It has many K4s and no Ind6 (alpha=1).
    # Flip edges to break K4s. This reduces density, potentially creating Ind6.
    # This is like the current local search but with SA.
    
    # Or start with a sparse graph (e.g., empty).
    # It has no K4s but large Ind6.
    # Add edges to break Ind6. This increases density, potentially creating K4s.
    
    # The current program starts random.
    
    # Let's stick to the "fix violations" strategy but with SA.
    # State: adj.
    # Move: flip a random edge.
    # Cost: 
    #   0 if valid.
    #   1 if invalid.
    #   (This is too coarse).
    
    # Let's use:
    #   Cost = (1 if has K4 else 0) * 10 + (1 if has Ind6 else 0) * 10
    #   This doesn't help with gradient.
    
    # How about we count the number of K4s and Ind6s?
    # Counting K4s is O(n^4) or faster with bitsets.
    # Counting Ind6 is harder.
    
    # Let's try a heuristic:
    # If we flip an edge (u,v):
    # - If we add an edge:
    #   - It might create K4s containing (u,v).
    #   - It might destroy Ind6s containing (u,v) as non-edge.
    # - If we remove an edge:
    #   - It might destroy K4s containing (u,v).
    #   - It might create Ind6s containing (u,v) as non-edge.
    
    # We can compute the "change" in violations locally.
    # But "has K4" is a global property.
    
    # Given the time limit, and that the current program already does local search,
    # the main improvement should be:
    # 1. Faster validity check (done above with pruning).
    # 2. Better initial guesses (try more known constructions).
    # 3. Try n=36 directly with SA starting from a valid n=35 graph if found, 
    #    or from a random graph with high probability of being close to valid.
    
    # Actually, the most impactful change is to **verify if Paley(37) is valid**.
    # If it is, we return it immediately.
    # If not, we try to repair it.
    
    # Let's implement a "repair" SA:
    # Start with Paley(37).
    # While invalid:
    #   Find a violation.
    #   Flip an edge involved in the violation with probability p.
    #   Or flip random edge.
    
    # This is similar to current program.
    
    # What if we use **Genetic Algorithm**?
    # Population of graphs.
    # Crossover: take edges from two parents.
    # Mutation: flip edges.
    # Selection: keep valid ones, or those with fewer violations.
    
    # Let's try GA.
    
    pop_size = 50
    population = []
    
    # Initialize population
    # 1. Paley(37)
    # 2. Random graphs
    # 3. Clebsch + random
    
    for i in range(pop_size):
        if i == 0:
            g = _paley_graph(n)
            adj = _build_adj(n, g['edges'])
        elif i == 1:
            g = _clebsch_graph()
            # Extend to n? If n > 16, add random vertices?
            # This is tricky.
            # Let's just use random for others.
            adj = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        else:
            adj = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        population.append(adj)
        
    # Evolution
    while time.time() < deadline:
        # Evaluate fitness
        # Fitness: 0 if valid, else 1 + small penalty
        # We need a fast "almost valid" check.
        # Let's just use _is_valid_graph for the top few candidates.
        
        # Select best
        # For speed, let's just pick random parents and mutate.
        # If a child is valid, return it.
        
        p1 = random.choice(population)
        p2 = random.choice(population)
        
        # Crossover
        child = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                # Randomly pick edge from p1 or p2
                if random.random() < 0.5:
                    if (p1[u] >> v) & 1:
                        child[u] |= (1 << v)
                        child[v] |= (1 << u)
                else:
                    if (p2[u] >> v) & 1:
                        child[u] |= (1 << v)
                        child[v] |= (1 << u)
                        
        # Mutation
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.05: # 5% mutation rate
                    if (child[u] >> v) & 1:
                        child[u] &= ~(1 << v)
                        child[v] &= ~(1 << u)
                    else:
                        child[u] |= (1 << v)
                        child[v] |= (1 << u)
                        
        if _is_valid_graph(n, _edges_from_adj(child, n)):
            return _edges_from_adj(child, n)
            
        # Replace worst in population?
        # Or just keep random.
        population[random.randint(0, pop_size-1)] = child
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Check known constructions
    candidates = [
        ('paley37', _paley_graph(37)),
        ('paley29', _paley_graph(29)),
        ('paley17', _paley_graph(17)),
        ('clebsch', _clebsch_graph()),
    ]
    
    best_graph = None
    best_n = 0
    
    for name, g in candidates:
        if time.time() > deadline:
            break
        if _is_valid_graph(g['n'], g['edges']):
            if g['n'] > best_n:
                best_graph = g
                best_n = g['n']
                
    # If we found n=37, we are done (it's the max possible for this seed? No, max is 40).
    # But R(4,6) is likely 36 or 37.
    # If Paley(37) is valid, return it.
    if best_n >= 36:
        return best_graph
        
    # 2. Try to find larger n via GA/SA
    # We want to maximize n.
    # Try n=36, 35, 34...
    
    for target_n in [36, 35, 34, 33]:
        if time.time() > deadline:
            break
        if target_n <= best_n:
            continue
            
        remaining = deadline - time.time()
        if remaining < 5:
            break
            
        # Use GA for this n
        edges = _simulated_annealing(target_n, _build_adj(target_n, _paley_graph(29)['edges'] if target_n >= 29 else _build_adj(target_n, [])), remaining - 2, seed=42)
        
        if edges is not None:
            g = {'n': target_n, 'edges': edges}
            if _is_valid_graph(target_n, edges):
                best_graph = g
                best_n = target_n
                break # Found the largest possible we tried
                
    if best_graph is None:
        best_graph = _clebsch_graph()
        
    return best_graph
