    def find_clique(allowed_primes):
        # Generate all smooth numbers up to 2*MAX_VAL
        smooth_nums = set()
        # Use BFS/DFS to generate smooth numbers
        # Or just iterate?
        # Iterating 1..20000 and checking is_smooth is 20000 checks. Fast.
        for i in range(2, 2*MAX_VAL + 1):
            if is_smooth(i, allowed_primes):
                smooth_nums.add(i)
                
        # Build adjacency list
        # adj[x] is a set of y such that x+y in smooth_nums
        adj = [set() for _ in range(MAX_VAL + 1)]
        
        for x in range(1, MAX_VAL + 1):
            for s in smooth_nums:
                y = s - x
                if 1 <= y <= MAX_VAL and y != x:
                    adj[x].add(y)
                    
        # Now search for clique of size 12
        # Random restarts
        for _ in range(1000):
            # Pick a random start
            start = random.randint(1, MAX_VAL)
            current = [start]
            # Candidates are neighbors of start
            candidates = set(adj[start])
            
            while len(current) < N:
                if not candidates:
                    break # Failed this restart
                    
                # Pick a candidate
                # Heuristic: Pick one with highest degree in the induced subgraph?
                # Or just random?
                # Random is simpler.
                # But to avoid getting stuck in low-degree parts, 
                # maybe pick the one with max len(adj[node] & candidates)?
                # That's expensive.
                # Let's just pick random.
                
                next_node = random.choice(list(candidates))
                current.append(next_node)
                
                # Update candidates: intersection of current candidates and adj[next_node]
                candidates = candidates.intersection(adj[next_node])
                
                if time.time() > deadline:
                    return None
            
            if len(current) == N:
                return current
                
        return None
