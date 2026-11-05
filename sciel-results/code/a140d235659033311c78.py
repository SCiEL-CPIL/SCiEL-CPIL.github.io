import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    def is_valid(orient, n):
        # orient is a list of bits for pairs (i,j) i<j
        # 0 means i->j, 1 means j->i
        # Build adjacency matrix for faster access
        # adj[i][j] = 1 if i->j else 0
        adj = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                bit = orient[idx]
                if bit == 0:
                    adj[i][j] = 1
                else:
                    adj[j][i] = 1
                idx += 1
        
        # Check all 4-subsets
        for comb in itertools.combinations(range(n), 4):
            a, b, c, d = comb
            # Check if any vertex has out-degree 3 within the subset
            # a beats b,c,d?
            if adj[a][b] and adj[a][c] and adj[a][d]:
                continue
            if adj[b][a] and adj[b][c] and adj[b][d]:
                continue
            if adj[c][a] and adj[c][b] and adj[c][d]:
                continue
            if adj[d][a] and adj[d][b] and adj[d][c]:
                continue
            return False
        return True

    def find_violation(orient, n, adj):
        # Pick a random 4-subset and check
        while True:
            comb = random.sample(range(n), 4)
            a, b, c, d = comb
            # Check if any vertex has out-degree 3
            if adj[a][b] and adj[a][c] and adj[a][d]:
                continue
            if adj[b][a] and adj[b][c] and adj[b][d]:
                continue
            if adj[c][a] and adj[c][b] and adj[c][d]:
                continue
            if adj[d][a] and adj[d][b] and adj[d][c]:
                continue
            return comb

    def local_search(n, time_limit):
        m = n * (n - 1) // 2
        orient = [random.randint(0, 1) for _ in range(m)]
        
        # Build adj
        adj = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                bit = orient[idx]
                if bit == 0:
                    adj[i][j] = 1
                else:
                    adj[j][i] = 1
                idx += 1

        if is_valid(orient, n):
            return orient

        while time.time() < time_limit:
            comb = find_violation(orient, n, adj)
            a, b, c, d = comb
            # Flip a random edge in the violation
            pair_idx = random.choice([(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)])
            u, v = pair_idx
            if u > v:
                u, v = v, u
            
            # Find index in orient
            # Index calculation:
            # Pairs are (0,1), (0,2), ..., (0,n-1), (1,2), ...
            # Offset for i: i*(2*n - i - 1) // 2
            # Offset for j: j - i - 1
            # Total index: i*(2*n - i - 1) // 2 + (j - i - 1)
            idx = u * (2 * n - u - 1) // 2 + (v - u - 1)
            
            # Flip
            orient[idx] ^= 1
            
            # Update adj
            if adj[u][v]:
                adj[u][v] = 0
                adj[v][u] = 1
            else:
                adj[u][v] = 1
                adj[v][u] = 0
            
            # Periodic full check
            # If we do this too often, it's slow.
            # Let's check every 100 steps? Or just rely on random sampling?
            # Random sampling might miss validity if the set of violations is small.
            # But if we keep fixing violations, we should converge.
            # To be safe, let's do a full check every 1000 steps.
            # We need a step counter.
            
            # This logic is inside the while loop, so we need a counter outside or passed in.
            # Let's restructure.
            pass

    # Restructured local search
    def local_search_v2(n, time_limit):
        m = n * (n - 1) // 2
        orient = [random.randint(0, 1) for _ in range(m)]
        
        adj = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                bit = orient[idx]
                if bit == 0:
                    adj[i][j] = 1
                else:
                    adj[j][i] = 1
                idx += 1

        if is_valid(orient, n):
            return orient

        steps = 0
        while time.time() < time_limit:
            # Find violation
            comb = find_violation(orient, n, adj)
            a, b, c, d = comb
            
            # Flip random edge
            pair_idx = random.choice([(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)])
            u, v = pair_idx
            if u > v:
                u, v = v, u
            
            idx = u * (2 * n - u - 1) // 2 + (v - u - 1)
            orient[idx] ^= 1
            
            if adj[u][v]:
                adj[u][v] = 0
                adj[v][u] = 1
            else:
                adj[u][v] = 1
                adj[v][u] = 0
                
            steps += 1
            if steps % 500 == 0:
                if is_valid(orient, n):
                    return orient

        return None

    # Try different n
    # We want to minimize n.
    # Let's try a range.
    # If we find a valid one for n=50, we return it.
    # We should try smaller n first?
    # But smaller n is harder to find?
    # Actually, random tournaments are more likely to have violations for small n?
    # For n=5, $\binom{5}{4}=5$ subsets.
    # For n=83, $\binom{83}{4} \approx 1.5 \times 10^6$.
    # The seed is valid.
    # Let's try n from 50 up to 82.
    # If we find one for 50, great.
    # If not, try 60, etc.
    
    # To maximize score, we want the smallest n.
    # Let's allocate time to different n.
    # Or just loop n from 31 to 82, giving each a small time slice?
    # 50 values. 100s / 50 = 2s each.
    # 2s is enough for local search to find a solution if one exists?
    # Probably.
    
    for n in range(31, 83):
        if time.time() > deadline - 5:
            break
        time_limit = time.time() + 2  # 2 seconds per n
        result = local_search_v2(n, time_limit)
        if result:
            return {'n': n, 'orientation': result}
            
    # If we didn't find anything smaller, return seed
    p = 83
    squares = {x * x % p for x in range(1, p)}
    return {'n': p, 'orientation': [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]}
