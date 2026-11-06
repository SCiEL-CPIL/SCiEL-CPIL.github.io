"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Known valid graph: 8 disjoint K4's -> n=32, alpha=8 < 9, no C5
    known_32 = {
        'n': 32,
        'edges': [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
    }
    
    def build_adj(n, edges):
        """Build adjacency bitmasks from edge list."""
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def has_c5(adj, n):
        """Check if graph contains a 5-cycle using bitsets. Returns True if found."""
        for u in range(n):
            nu = adj[u]
            v_mask = nu
            while v_mask:
                v_bit = v_mask & (-v_mask)
                v = v_bit.bit_length() - 1
                v_mask ^= v_bit
                
                a_mask = nu & ~(1 << v)
                while a_mask:
                    a_bit = a_mask & (-a_mask)
                    a = a_bit.bit_length() - 1
                    a_mask ^= a_bit
                    
                    b_mask = adj[a] & ~(1 << u) & ~(1 << v)
                    while b_mask:
                        b_bit = b_mask & (-b_mask)
                        b = b_bit.bit_length() - 1
                        b_mask ^= b_bit
                        
                        c_mask = adj[b] & ~(1 << u) & ~(1 << a) & ~(1 << v)
                        if c_mask & adj[v]:
                            return True
        return False
    
    def has_independent_set_9(adj, n):
        """Check if graph has an independent set of size 9 using bitset backtracking."""
        
        def can_find(available, count):
            if count == 0:
                return True
            if available == 0:
                return False
            if bin(available).count('1') < count:
                return False
            
            # Pick vertex with minimum degree in available (excludes fewest)
            best_v = -1
            best_excluded = 0
            avail = available
            while avail:
                v_bit = avail & (-avail)
                v = v_bit.bit_length() - 1
                avail ^= v_bit
                adj_in_avail = available & adj[v]
                excluded = bin(adj_in_avail).count('1')
                if best_v == -1 or excluded < best_excluded:
                    best_excluded = excluded
                    best_v = v
                if excluded == 0:
                    break
            
            if best_v == -1:
                return False
            
            v = best_v
            v_bit = 1 << v
            
            # Option 1: include v
            new_avail = available & ~adj[v] & ~v_bit
            if can_find(new_avail, count - 1):
                return True
            
            # Option 2: exclude v
            if can_find(available & ~v_bit, count):
                return True
            
            return False
        
        return can_find((1 << n) - 1, 9)
    
    def is_valid(n, adj):
        """Check if graph is C5-free and has no independent set of 9."""
        if has_c5(adj, n):
            return False
        if has_independent_set_9(adj, n):
            return False
        return True
    
    def adj_to_edges(adj, n):
        """Convert adjacency bitmasks to edge list."""
        edges = []
        for i in range(n):
            mask = adj[i]
            for j in range(i + 1, n):
                if mask & (1 << j):
                    edges.append([i, j])
        return edges
    
    def complement_adj(adj, n):
        """Compute complement adjacency bitmasks."""
        full = (1 << n) - 1
        comp = [0] * n
        for i in range(n):
            comp[i] = full ^ adj[i] ^ (1 << i)
        return comp
    
    def has_k9(comp_adj, n):
        """Check if complement graph has a K9 (i.e., original has independent set of 9)."""
        # K9 in complement = 9 vertices all connected in complement
        # Use backtracking
        def find_clique(available, count):
            if count == 0:
                return True
            if available == 0:
                return False
            if bin(available).count('1') < count:
                return False
            
            # Pick vertex with max degree in available
            best_v = -1
            best_deg = -1
            avail = available
            while avail:
                v_bit = avail & (-avail)
                v = v_bit.bit_length() - 1
                avail ^= v_bit
                deg = bin(available & comp_adj[v]).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                if deg == bin(available).count('1') - 1:
                    break
            
            if best_v == -1:
                return False
            
            v = best_v
            v_bit = 1 << v
            
            # Include v
            new_avail = available & comp_adj[v] & ~v_bit
            if find_clique(new_avail, count - 1):
                return True
            
            # Exclude v
            if find_clique(available & ~v_bit, count):
                return True
            
            return False
        
        return find_clique((1 << n) - 1, 9)
    
    def has_induced_c5(comp_adj, n):
        """Check if complement graph has an induced C5 (i.e., original has induced C5 in complement).
        Note: We need original G to have NO C5 as subgraph, not induced.
        So this check is for the complement formulation where we want G' to have no induced C5.
        But actually, we want G (original) to have no C5 subgraph.
        In complement: G has no C5 subgraph means... this is complex.
        Let's just use the original has_c5 for validation.
        """
        # For the complement approach, we'll validate using original constraints
        return False  # Placeholder - we validate with original has_c5
    
    def ga_complement_search(target_n, time_budget):
        """Genetic algorithm on complement graph to find G' with no K9 and no induced C5.
        G' valid means original G = complement(G') has no C5 and alpha(G) < 9.
        """
        # Initialize population
        pop_size = 50
        population = []
        
        for _ in range(pop_size):
            # Random complement graph
            comp = [0] * target_n
            p = random.uniform(0.1, 0.3)
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < p:
                        comp[i] |= (1 << j)
                        comp[j] |= (1 << i)
            population.append(comp)
        
        best_comp = None
        best_score = -1
        
        gen = 0
        while time.time() < time_budget:
            gen += 1
            
            # Evaluate fitness
            scored = []
            for comp in population:
                # Convert to original
                full = (1 << target_n) - 1
                orig = [0] * target_n
                for i in range(target_n):
                    orig[i] = full ^ comp[i] ^ (1 << i)
                
                # Check constraints
                c5_free = not has_c5(orig, target_n)
                no_ind9 = not has_independent_set_9(orig, target_n)
                
                # Score: prefer valid, then higher edge count in original
                if c5_free and no_ind9:
                    # Count edges in original
                    edge_count = sum(bin(orig[i]).count('1') for i in range(target_n)) // 2
                    score = 1000 + edge_count
                elif c5_free:
                    score = 500 + bin(full).count('1') - sum(bin(comp[i]).count('1') for i in range(target_n)) // 2
                else:
                    score = 0
                
                scored.append((score, comp))
            
            scored.sort(key=lambda x: x[0], reverse=True)
            
            # Update best
            if scored[0][0] > best_score:
                best_score = scored[0][0]
                best_comp = scored[0][1]
                if best_score >= 1000:
                    # Found valid solution
                    full = (1 << target_n) - 1
                    orig = [0] * target_n
                    for i in range(target_n):
                        orig[i] = full ^ best_comp[i] ^ (1 << i)
                    edges = adj_to_edges(orig, target_n)
                    return {'n': target_n, 'edges': edges}
            
            # Create next generation
            new_pop = []
            
            # Elitism: keep top 10%
            elite_count = max(1, pop_size // 10)
            for i in range(elite_count):
                new_pop.append(scored[i][1][:])
            
            # Crossover and mutation to fill rest
            while len(new_pop) < pop_size:
                if time.time() > time_budget:
                    break
                
                # Select parents (tournament)
                def tournament_select(k=3):
                    candidates = random.sample(scored, k)
                    return max(candidates, key=lambda x: x[0])[1]
                
                p1 = tournament_select()
                p2 = tournament_select()
                
                # Uniform crossover
                child = [0] * target_n
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if random.random() < 0.5:
                            if p1[i] & (1 << j):
                                child[i] |= (1 << j)
                                child[j] |= (1 << i)
                        else:
                            if p2[i] & (1 << j):
                                child[i] |= (1 << j)
                                child[j] |= (1 << i)
                
                # Mutation
                mut_rate = 0.05
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if random.random() < mut_rate:
                            if child[i] & (1 << j):
                                child[i] &= ~(1 << j)
                                child[j] &= ~(1 << i)
                            else:
                                child[i] |= (1 << j)
                                child[j] |= (1 << i)
                
                new_pop.append(child)
            
            population = new_pop[:pop_size]
        
        return None
    
    best_result = known_32
    
    # Strategy 0: Complement GA for n=33
    if time.time() < deadline:
        ga_budget = min(30.0, deadline - time.time())
        result = ga_complement_search(33, start_time + ga_budget)
        if result is not None:
            best_result = result
    
    # Strategy 1: Extend 8xK4 by adding vertices with careful connections
    if best_result['n'] < 33:
        for target_n in [33, 34, 35]:
            if time.time() > deadline:
                break
            
            extra = target_n - 32
            num_attempts = 2000
            for attempt in range(num_attempts):
                if time.time() > deadline:
                    break
                
                # Initialize: 8xK4 structure
                adj = [0] * target_n
                for b in range(8):
                    for i in range(4):
                        for j in range(i + 1, 4):
                            u = 4 * b + i
                            v = 4 * b + j
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                
                # Add edges from extra vertices to existing vertices
                for ex in range(32, target_n):
                    # Connect to vertices from 1-2 blocks to maintain C5-freeness
                    block = random.randint(0, 7)
                    base = 4 * block
                    num_conn = random.randint(2, 4)
                    targets = random.sample(range(4), num_conn)
                    for t in targets:
                        v = base + t
                        adj[ex] |= (1 << v)
                        adj[v] |= (1 << ex)
                    
                    # Optionally connect to a second block
                    if random.random() < 0.4:
                        block2 = random.randint(0, 7)
                        if block2 != block:
                            base2 = 4 * block2
                            num_conn2 = random.randint(1, 2)
                            targets2 = random.sample(range(4), num_conn2)
                            for t in targets2:
                                v = base2 + t
                                adj[ex] |= (1 << v)
                                adj[v] |= (1 << ex)
                
                # Check validity
                if has_c5(adj, target_n):
                    continue
                if has_independent_set_9(adj, target_n):
                    continue
                
                edges = adj_to_edges(adj, target_n)
                best_result = {'n': target_n, 'edges': edges}
                break
    
    # Strategy 2: Local search - start from valid n=32 graph and modify
    if best_result['n'] < 33 and time.time() < deadline:
        for target_n in [33, 34, 35]:
            if time.time() > deadline:
                break
            
            found = False
            for attempt in range(3000):
                if time.time() > deadline:
                    break
                
                # Start with 8xK4 + extra isolated vertices
                adj = [0] * target_n
                for b in range(8):
                    for i in range(4):
                        for j in range(i + 1, 4):
                            u = 4 * b + i
                            v = 4 * b + j
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                
                # Add random edges from extra vertices
                for ex in range(32, target_n):
                    # Connect to 3-5 random vertices
                    num_conn = random.randint(3, 5)
                    targets = random.sample(range(32), num_conn)
                    for t in targets:
                        adj[ex] |= (1 << t)
                        adj[t] |= (1 << ex)
                
                # Try to fix C5s by removing edges
                if has_c5(adj, target_n):
                    adj_copy = adj[:]
                    for _ in range(20):
                        if time.time() > deadline:
                            break
                        if not has_c5(adj_copy, target_n):
                            break
                        all_edges = []
                        for i in range(target_n):
                            for j in range(i + 1, target_n):
                                if adj_copy[i] & (1 << j):
                                    all_edges.append((i, j))
                        if not all_edges:
                            break
                        u, v = random.choice(all_edges)
                        adj_copy[u] &= ~(1 << v)
                        adj_copy[v] &= ~(1 << u)
                    adj = adj_copy
                
                if has_c5(adj, target_n):
                    continue
                if has_independent_set_9(adj, target_n):
                    continue
                
                edges = adj_to_edges(adj, target_n)
                best_result = {'n': target_n, 'edges': edges}
                found = True
                break
            
            if found:
                break
    
    # Strategy 3: Pure random search with low density
    if best_result['n'] < 33 and time.time() < deadline:
        for target_n in [33, 34, 35]:
            if time.time() > deadline:
                break
            
            found = False
            for attempt in range(5000):
                if time.time() > deadline:
                    break
                
                p = random.uniform(0.08, 0.20)
                
                adj = [0] * target_n
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if random.random() < p:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                
                if has_c5(adj, target_n):
                    continue
                if has_independent_set_9(adj, target_n):
                    continue
                
                edges = adj_to_edges(adj, target_n)
                best_result = {'n': target_n, 'edges': edges}
                found = True
                break
            
            if found:
                break
    
    return best_result

# EVOLVE-BLOCK-END
