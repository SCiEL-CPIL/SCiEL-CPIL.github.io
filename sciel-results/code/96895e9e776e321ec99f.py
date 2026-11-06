import itertools
import time
import random

def construct():
        start_time = time.time()
        timeout = 285
        
        # Fallback: Paley 37
        def get_paley(q):
            residues = {(x * x) % q for x in range(1, q)}
            edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
            return q, edges
        
        n_seed, edges_seed = get_paley(37)
        
        # Try to build n=42
        n_target = 42
        
        # Precompute all 5-sets
        all_5sets = list(itertools.combinations(range(n_target), 5))
        
        # Helper to check if a 5-set is a clique or independent given adj
        def is_bad_5set(combo, adj):
            # Check Clique
            ok_clique = True
            # Check Independent
            ok_indep = True
            for i in range(5):
                u = combo[i]
                for j in range(i+1, 5):
                    v = combo[j]
                    if adj[u] & (1 << v):
                        ok_indep = False
                    else:
                        ok_clique = False
                    if not ok_clique and not ok_indep:
                        return False
            return ok_clique or ok_indep
        
        # Local Search
        best_result = (n_seed, edges_seed)
        
        # Precompute adjacency bitmasks for fast checking
        def count_bad_5sets(adj):
            """Count number of bad 5-sets (cliques or independent sets)"""
            count = 0
            for c in all_5sets:
                if is_bad_5set(c, adj):
                    count += 1
            return count
        
        # Precompute for each pair (u,v), the list of 5-sets containing both u and v
        # This allows efficient violation updates
        pair_to_5sets = {}
        for c in all_5sets:
            for i in range(5):
                for j in range(i+1, 5):
                    u, v = c[i], c[j]
                    if u > v:
                        u, v = v, u
                    key = (u, v)
                    if key not in pair_to_5sets:
                        pair_to_5sets[key] = []
                    pair_to_5sets[key].append(c)
        
        for restart in range(50): # 50 restarts
            if time.time() - start_time > timeout:
                break
            
            # Initialize: start from Paley-like structure extended to 42 vertices
            # Use probabilistic initialization with p tuned for 5-5-free graphs
            random.seed(42 + restart)
            adj = [0] * n_target
            edges = set()
            
            # For R(5,5), optimal edge density is around 0.5
            # Use p=0.5 but with some structure
            for u in range(n_target):
                for v in range(u+1, n_target):
                    if random.random() < 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edges.add((u, v))
            
            # Initial violation set
            violations = set()
            for c in all_5sets:
                if is_bad_5set(c, adj):
                    violations.add(c)
            
            # If too many violations, try different p
            if len(violations) > 60000:
                continue
            
            # Local Search Loop with simulated annealing
            max_steps = 12000
            current_temp = 1.0
            cooling = 0.9995
            
            for step in range(max_steps):
                if time.time() - start_time > timeout:
                    break
                
                if not violations:
                    # Success!
                    edge_list = list(edges)
                    return n_target, edge_list
                
                # Pick a random violation
                v_set = random.choice(list(violations))
                
                # Determine if it's a clique or independent set
                is_clique = True
                is_indep = True
                for i in range(5):
                    u = v_set[i]
                    for j in range(i+1, 5):
                        v = v_set[j]
                        if adj[u] & (1 << v):
                            is_indep = False
                        else:
                            is_clique = False
                        if not is_clique and not is_indep:
                            break
                    if not is_clique and not is_indep:
                        break
                
                # Generate pairs in v_set
                pairs = list(itertools.combinations(v_set, 2))
                u, v = random.choice(pairs)
                if u > v: u, v = v, u
                
                # Flip edge (u, v)
                if is_clique and not is_indep:
                    # Remove edge
                    if (u, v) in edges:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        edges.remove((u, v))
                    else:
                        continue 
                elif is_indep and not is_clique:
                    # Add edge
                    if (u, v) not in edges:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edges.add((u, v))
                    else:
                        continue
                else:
                    # Mixed case - try both
                    if random.random() < 0.5:
                        if (u, v) in edges:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            edges.remove((u, v))
                        else:
                            continue
                    else:
                        if (u, v) not in edges:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            edges.add((u, v))
                        else:
                            continue
                        
                # Update violations efficiently using precomputed mapping
                key = (u, v)
                if key in pair_to_5sets:
                    for full_combo in pair_to_5sets[key]:
                        if is_bad_5set(full_combo, adj):
                            violations.add(full_combo)
                        else:
                            violations.discard(full_combo)
                
                # Simulated annealing: occasionally accept worse solutions
                if step % 100 == 0:
                    current_temp *= cooling
        
        return best_result
