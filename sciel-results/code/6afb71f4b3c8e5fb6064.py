"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    START_TIME = time.time()
    DEADLINE = START_TIME + 115.0

    def instance(splits, seed, palette=5):
        from random import Random
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_instance(n, edges, lists, rng_solver, time_limit):
        # Build adjacency as bitmasks for fast neighbor checks
        adj_mask = [0] * n
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        # Convert lists to bitmask domains
        init_domains = [0] * n
        for v in range(n):
            mask = 0
            for c in lists[v]:
                mask |= (1 << c)
            init_domains[v] = mask
        
        # Check for empty initial domains
        for v in range(n):
            if init_domains[v] == 0:
                return None
        
        # We'll use a randomized backtracking with MRV and forward checking.
        # State: 
        #   domains: list of int bitmasks
        #   assigned: list of int, -1 if unassigned
        #   unassigned: list of vertex indices
        
        # To support multiple attempts/restarts with different random orders, 
        # we can just shuffle the initial unassigned list and re-run the backtracker.
        # Since the problem is small, a single run should suffice, but restarts add robustness.
        
        # Helper to get a random element from a bitmask
        def random_from_mask(mask, rng):
            # Get all set bits
            bits = []
            m = mask
            while m:
                lb = m & (-m)
                idx = lb.bit_length() - 1
                bits.append(idx)
                m ^= lb
            return rng.choice(bits)
        
        # Helper to get count of bits
        def popcount(x):
            return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')
        
        def backtrack(domains, assigned, unassigned, nodes_limit):
            if not unassigned:
                return True
            
            # Check time
            if time.time() > time_limit:
                return False
            
            # MRV: find vertex with minimum remaining values
            min_count = 5
            best_idx = -1
            candidates = []
            for i, v in enumerate(unassigned):
                c = popcount(domains[v])
                if c < min_count:
                    min_count = c
                    candidates = [i]
                    if min_count == 0:
                        break
                elif c == min_count:
                    candidates.append(i)
                
            if min_count == 0:
                return False
            
            best_idx = rng_solver.choice(candidates)
            v = unassigned[best_idx]
            unassigned.pop(best_idx)
            
            # Get values to try
            mask = domains[v]
            # Extract values
            vals = []
            m = mask
            while m:
                lb = m & (-m)
                idx = lb.bit_length() - 1
                vals.append(idx)
                m ^= lb
            rng_solver.shuffle(vals)
            
            for val in vals:
                val_bit = 1 << val
                
                # Check consistency with assigned neighbors
                ok = True
                # Iterate over set bits in adj_mask[v]
                nb_mask = adj_mask[v]
                m = nb_mask
                while m:
                    lb = m & (-m)
                    u = lb.bit_length() - 1
                    if assigned[u] == val:
                        ok = False
                        break
                    m ^= lb
                if not ok:
                    continue
                
                assigned[v] = val
                
                # Forward checking
                saved = []
                ok_fc = True
                nb_mask = adj_mask[v]
                m = nb_mask
                while m:
                    lb = m & (-m)
                    u = lb.bit_length() - 1
                    if assigned[u] == -1:
                        if domains[u] & val_bit:
                            saved.append((u, val_bit))
                            domains[u] &= ~val_bit
                            if domains[u] == 0:
                                ok_fc = False
                                break
                    m ^= lb
                
                if ok_fc:
                    if backtrack(domains, assigned, unassigned, nodes_limit):
                        return True
                
                # Restore forward checking changes
                for u, bit in saved:
                    domains[u] |= bit
                
                assigned[v] = -1
            
            unassigned.insert(best_idx, v)
            return False

        # Main solve loop with potential restarts
        max_restarts = 10
        for r in range(max_restarts):
            if time.time() > time_limit:
                break
            
            # Initialize state
            domains = init_domains[:]
            assigned = [-1] * n
            unassigned = list(range(n))
            rng_solver.shuffle(unassigned)
            
            # Limit nodes per restart to allow quick restarts if stuck
            # A large limit since backtracking is exponential
            nodes_limit = 500000 
            
            # The backtrack function above doesn't actually use nodes_limit for counting,
            # it relies on time_limit. We can add a node counter if needed, 
            # but time_limit is usually sufficient.
            # To prevent a single deep search from eating all time, 
            # we can rely on the global time_limit.
            
            if backtrack(domains, assigned, unassigned, nodes_limit):
                return [assigned[i] for i in range(n)]
        
        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > DEADLINE:
            # If we run out of time, we must still return a valid structure.
            # However, the contract says any broken rule makes the result invalid.
            # Returning a list of 4 lists of correct length is required for validity.
            # If we can't solve it, we can't guarantee correctness.
            # But we assume the solver is fast enough.
            pass

        n, edges, lists = instance(splits, seed)
        
        # Solver RNG
        rng_solver = random.Random(seed + 42)
        
        # Allocate time
        time_per_instance = 25.0
        t_start = time.time()
        
        coloring = solve_instance(n, edges, lists, rng_solver, t_start + time_per_instance)
        
        if coloring is None:
            # Fallback: Greedy with degree ordering
            # This will likely fail for some instances, resulting in score 0 for that instance,
            # but it prevents a crash/exception which would make the whole result invalid.
            
            # Build adjacency list for greedy
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            order = sorted(range(n), key=lambda x: -len(adj[x]))
            col = [0] * n
            for v in order:
                taken = set()
                for u in adj[v]:
                    taken.add(col[u])
                free = [c for c in lists[v] if c not in taken]
                if free:
                    col[v] = free[0]
                else:
                    # Constraint violation, pick first available from list
                    col[v] = lists[v][0]
            coloring = col
            
        # Ensure types are Python int
        # The solver returns Python ints. The greedy fallback also returns Python ints.
        # We must ensure the list elements are int, not bool or numpy int.
        # Since we use standard python types, they are int.
        
        out.append(coloring)
        
    return out
# EVOLVE-BLOCK-END
