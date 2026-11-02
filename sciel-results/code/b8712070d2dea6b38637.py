def construct(k, g):
    n = 320
    half = 160
    deadline = time.time() + 110
    random.seed(42)
    
    def init_perms():
        return [list(range(half)) for _ in range(4)] # Not shuffled yet
        # Actually, shuffle each
        perms = []
        for _ in range(4):
            p = list(range(half))
            random.shuffle(p)
            perms.append(p)
        return perms

    def calc_bad_for_node(perms, i):
        """Calculate contribution of node i to total 4-cycles with all j > i? No, just pairs."""
        # We need to count pairs (i, j) for all j != i.
        # Let's define bad(i, j) = C(|N(i) \cap N(j)|, 2)
        # Total bad = sum_{i < j} bad(i, j)
        
        # To update, we recalculate bad(i, j) for all j != i.
        cost = 0
        ni = [perms[a][i] for a in range(4)]
        for j in range(half):
            if i == j: continue
            nj = [perms[a][j] for a in range(4)]
            common = 0
            for r in ni:
                if r in nj:
                    common += 1
            if common >= 2:
                cost += common * (common - 1) // 2
        return cost

    def check_girth(perms):
        # Same as before
        adj = [[] for _ in range(2*half)]
        for i in range(half):
            for a in range(4):
                j = perms[a][i]
                adj[i].append(half + j)
                adj[half + j].append(i)
        
        for start in range(2*half):
            dist = [-1] * (2*half)
            dist[start] = 0
            queue = [start]
            for d in range(1, g//2):
                new_queue = []
                for node in queue:
                    for neighbor in adj[node]:
                        if dist[neighbor] == -1:
                            dist[neighbor] = d
                            new_queue.append(neighbor)
                        else:
                            cycle_len = d + dist[neighbor] + 1
                            if cycle_len < g:
                                return False
                queue = new_queue
        return True

    best_perms = None
    best_bad = float('inf')
    
    while time.time() < deadline:
        perms = init_perms()
        
        # Initial bad
        bad = 0
        for i in range(half):
            bad += calc_bad_for_node(perms, i)
        # Wait, this counts each pair twice (i, j) and (j, i).
        # So bad is 2 * actual 4-cycles.
        # We can just minimize this.
        
        for step in range(10000):
            if time.time() > deadline:
                break
            
            if bad == 0:
                if check_girth(perms):
                    edges = []
                    for i in range(half):
                        for a in range(4):
                            edges.append((i, half + perms[a][i]))
                    return 2*half, edges
            
            a, b = random.sample(range(4), 2)
            i, j = random.sample(range(half), 2)
            
            old_ai = perms[a][i]
            old_bj = perms[b][j]
            
            # Swap
            perms[a][i] = old_bj
            perms[b][j] = old_ai
            
            # Recalculate bad for i and j
            new_bad_i = calc_bad_for_node(perms, i)
            new_bad_j = calc_bad_for_node(perms, j)
            
            # Old bad_i and bad_j were included in 'bad'
            # But we didn't store them.
            # Let's store bad_i and bad_j in arrays?
            # Or just recalculate total bad? Too slow.
            
            # Approximation: if new_bad_i + new_bad_j < old_bad_i + old_bad_j, accept.
            # We need old_bad_i and old_bad_j.
            
            # Let's maintain an array bad_contrib[i]
            # This requires careful update.
            
            # For simplicity, let's just do full recalc every 100 steps?
            # No, incremental is better.
            
            # Let's just use the full count if it's fast enough with NumPy?
            # No, let's stick to Python but optimize.
            
            # Revert for now if we can't easily track
            perms[a][i] = old_ai
            perms[b][j] = old_bj
            
            # Try a simpler move: swap two values in the SAME permutation?
            # No, that doesn't change 4-cycles.
            
            # Swap values in two different permutations at the SAME index?
            # a, i and b, i.
            # This changes N(i) but not N(j) for j!=i.
            # So only bad(i, j) changes for all j.
            
            a, b = random.sample(range(4), 2)
            i = random.randint(0, half - 1)
            
            old_ai = perms[a][i]
            old_bi = perms[b][i]
            
            # They are distinct because permutations are distinct?
            # Not necessarily.
            if old_ai == old_bi: continue
            
            perms[a][i] = old_bi
            perms[b][i] = old_ai
            
            new_bad_i = calc_bad_for_node(perms, i)
            # We need old_bad_i.
            
            # This is getting complicated.
            # Let's just run the full count but with a smaller search space or faster check.
            
            # Actually, 10000 steps * 0.1s = 1000s. Too slow.
            # We must use incremental.
            
            break # Break inner loop to implement incremental properly in next thought
            
        # If we didn't find, update best
        # ...
        
    return 2*half, []
