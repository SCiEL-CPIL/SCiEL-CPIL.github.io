import time
import random

def construct():
    deadline = time.time() + 110
    random.seed(42)
    
    def count_k4s(adj, n):
        """Count number of K4s using bitsets."""
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    common = adj[i] & adj[j]
                    common &= ~(1 << i)
                    common &= ~(1 << j)
                    c = common
                    while c:
                        lsb = c & -c
                        k = lsb.bit_length() - 1
                        c ^= lsb
                        # Count edges in common neighbors of k
                        m = adj[k] & common
                        count += bin(m).count('1')
        return count // 4  # Each K4 counted 4 times (once per edge in the K4? No, once per pair (i,j) and then (k,m) in common. Actually each K4 {i,j,k,m} is found when we look at pair (i,j) and find edge (k,m) in common. Also when looking at (i,k) and finding (j,m), etc. So 6 pairs? No.
        # Let's re-verify.
        # For a K4 {a,b,c,d}, the pairs are (a,b), (a,c), (a,d), (b,c), (b,d), (c,d).
        # When we check pair (a,b), common neighbors are {c,d}. We find edge (c,d). Count +1.
        # When we check pair (a,c), common neighbors are {b,d}. We find edge (b,d). Count +1.
        # ...
        # So each K4 contributes 6 to the raw count.
        # So we should divide by 6.
        # Let's fix the count logic to be clearer.
        # Actually, the loop above:
        # for i, j: if edge(i,j)
        #   common = N(i) & N(j)
        #   for k in common:
        #     count += popcount(N(k) & common)
        # For K4 {0,1,2,3}:
        # Pair (0,1): common={2,3}. k=2: N(2)&{2,3} = {3}. count+=1. k=3: N(3)&{2,3}={2}. count+=1. Total for (0,1) is 2.
        # Pair (0,2): common={1,3}. k=1: N(1)&{1,3}={3}. count+=1. k=3: N(3)&{1,3}={1}. count+=1. Total for (0,2) is 2.
        # ...
        # So each K4 contributes 6 * 2? No.
        # Let's trace carefully.
        # Pair (0,1): common = {2,3}.
        #   k=2: adj[2] & common = {0,1,3} & {2,3} = {3}. popcount=1.
        #   k=3: adj[3] & common = {0,1,2} & {2,3} = {2}. popcount=1.
        #   Sum for pair (0,1) is 2.
        # There are 6 pairs in a K4. So total sum is 12.
        # So count // 12 gives the number of K4s.
        # Let's just return the raw count and divide by 12 outside or adjust.
        # For the purpose of "score", a smaller number is better.
        # I will just return the raw count / 12.0 to be precise.
        pass # Re-implementing below

    def count_k4s_v2(adj, n):
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    common = adj[i] & adj[j]
                    common &= ~(1 << i)
                    common &= ~(1 << j)
                    c = common
                    while c:
                        lsb = c & -c
                        k = lsb.bit_length() - 1
                        c ^= lsb
                        m = adj[k] & common
                        count += bin(m).count('1')
        return count / 12.0

    def find_indep6(adj, n):
        """Find one independent 6-set using bitsets and backtracking."""
        non_adj = [0] * n
        full_mask = (1 << n) - 1
        for v in range(n):
            non_adj[v] = (~adj[v]) & full_mask & ~(1 << v)
            
        def backtrack(remaining_mask, chosen):
            if len(chosen) == 6:
                return tuple(chosen)
            if bin(remaining_mask).count('1') < 6 - len(chosen):
                return None
            
            r = remaining_mask
            while r:
                lsb = r & -r
                v = lsb.bit_length() - 1
                r ^= lsb
                
                new_chosen = chosen + [v]
                new_remaining = remaining_mask & non_adj[v]
                new_remaining &= ~(1 << v)
                
                result = backtrack(new_remaining, new_chosen)
                if result is not None:
                    return result
            return None
        
        return backtrack((1 << n) - 1, [])

    def max_indep_size(adj, n):
        """Find the size of the maximum independent set using branch and bound."""
        non_adj = [0] * n
        full_mask = (1 << n) - 1
        for v in range(n):
            non_adj[v] = (~adj[v]) & full_mask & ~(1 << v)
        
        max_size = 0
        
        def backtrack(remaining_mask, current_size):
            nonlocal max_size
            if not remaining_mask:
                if current_size > max_size:
                    max_size = current_size
                return
            
            if current_size + bin(remaining_mask).count('1') <= max_size:
                return
            
            # Pick a vertex to branch on
            # Heuristic: pick vertex with highest degree in remaining induced subgraph?
            # Or just pick the first one.
            lsb = remaining_mask & -remaining_mask
            v = lsb.bit_length() - 1
            
            # Option 1: Include v
            new_remaining = remaining_mask & non_adj[v]
            new_remaining &= ~(1 << v)
            backtrack(new_remaining, current_size + 1)
            
            # Option 2: Exclude v
            remaining_mask &= ~(1 << v)
            backtrack(remaining_mask, current_size)
            
        backtrack(full_mask, 0)
        return max_size

    def is_valid(adj, n):
        # Check K4
        if count_k4s_v2(adj, n) > 0:
            return False
        # Check Indep 6
        if find_indep6(adj, n) is not None:
            return False
        return True

    def extract_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def random_graph_bitset(n, p):
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    best_result = None
    best_n = 0

    # 1. Try the product construction from the seed (fast)
    n35 = 35
    D5 = {1, 4}
    D7 = {1, 2, 4}
    adj35 = [0] * n35
    for i in range(n35):
        a1, b1 = divmod(i, 7)
        for j in range(i+1, n35):
            a2, b2 = divmod(j, 7)
            da = (a2 - a1) % 5
            db = (b2 - b1) % 7
            if (da in D5 and db in D7) or (da not in D5 and db not in D7):
                adj35[i] |= (1 << j)
                adj35[j] |= (1 << i)
                
    if is_valid(adj35, n35):
        best_result = {'n': n35, 'edges': extract_edges(adj35, n35)}
        return best_result

    # 2. Local Search with Score
    # Score = count_k4s + lambda * max(0, alpha - 5)
    # We want to minimize Score.
    # If Score == 0, it's valid.
    
    def local_search(adj, n, max_iters=2000):
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            k4_count = count_k4s_v2(adj, n)
            alpha = max_indep_size(adj, n)
            
            if k4_count == 0 and alpha <= 5:
                return adj
            
            # If K4s exist, remove an edge from a K4
            if k4_count > 0:
                # Find a specific K4 to guide removal
                # We can reuse find_k4 logic or just pick random edge if K4 count is high?
                # Better to find a specific K4.
                k4_found = False
                for i in range(n):
                    for j in range(i+1, n):
                        if (adj[i] >> j) & 1:
                            common = adj[i] & adj[j]
                            common &= ~(1 << i)
                            common &= ~(1 << j)
                            c = common
                            while c:
                                lsb = c & -c
                                k = lsb.bit_length() - 1
                                c ^= lsb
                                m_bit = adj[k] & common
                                if m_bit:
                                    m = (m_bit & -m_bit).bit_length() - 1
                                    # Found K4 (i,j,k,m)
                                    # Remove a random edge from it
                                    pair_options = [(i, j), (i, k), (i, m), (j, k), (j, m), (k, m)]
                                    u, v = random.choice(pair_options)
                                    adj[u] &= ~(1 << v)
                                    adj[v] &= ~(1 << u)
                                    k4_found = True
                                    break
                            if k4_found: break
                    if k4_found: break
                
                if k4_found:
                    continue
            
            # If Alpha > 5, add an edge to an independent set
            if alpha > 5:
                # Find an independent set of size 6
                indep6 = find_indep6(adj, n)
                if indep6:
                    u, v = random.sample(indep6, 2)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    continue
                # If find_indep6 fails (shouldn't if alpha>5), try random edge
                # Or just break to avoid infinite loop if logic error
                break
            
            # If neither condition triggers but score > 0? 
            # This implies a bug in counting or detection.
            # For safety, if we are stuck, make a random move?
            # Or just break.
            break
        
        return None

    # Try to repair the 35-vertex graph
    adj_repair = adj35[:]
    result = local_search(adj_repair, n35)
    if result:
        best_result = {'n': n35, 'edges': extract_edges(result, n35)}
        return best_result

    # 3. Random Restarts
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        
        # Vary p slightly
        p = 0.4 + 0.1 * random.random()
        adj_rand = random_graph_bitset(35, p)
        result = local_search(adj_rand, 35)
        if result:
            best_result = {'n': 35, 'edges': extract_edges(result, 35)}
            return best_result

    # 4. Fallback to smaller n if 35 fails
    for target_n in range(34, 15, -1):
        if time.time() > deadline - 5:
            break
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            adj_rand = random_graph_bitset(target_n, 0.45)
            result = local_search(adj_rand, target_n)
            if result:
                if best_result is None or target_n > best_result['n']:
                    best_result = {'n': target_n, 'edges': extract_edges(result, target_n)}
                break
        if best_result is not None and best_result['n'] == target_n:
            break

    if best_result is None:
        # Return a valid small graph, e.g., Paley 17
        squares_17 = {x*x % 17 for x in range(1, 17)}
        edges_17 = [[i, j] for i in range(17) for j in range(i+1, 17) if (j-i) % 17 in squares_17]
        best_result = {'n': 17, 'edges': edges_17}

    return best_result
