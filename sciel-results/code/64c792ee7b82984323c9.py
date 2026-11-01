def find_generators_320_fast():
    N = 320
    candidates = [x for x in range(1, 160, 2)]
    
    # Precompute pairwise sums for all pairs
    # We need 4 generators s1, s2, s3, s4 such that
    # s1+s2, s1+s3, s1+s4, s2+s3, s2+s4, s3+s4 are all distinct mod N.
    
    # Let's iterate over s1, s2, s3, s4.
    # To reduce complexity, we can fix s1=1?
    # By symmetry, we can assume s1=1?
    # Not exactly, because the group is not cyclic in a way that allows scaling?
    # Yes, we can scale by a unit. 1 is a unit.
    # So we can fix s1=1.
    
    s1 = 1
    found = None
    
    # Iterate s2, s3, s4
    # s2, s3, s4 must be odd, distinct from 1.
    others = [x for x in candidates if x != 1]
    
    # Check 4-cycle condition for s1, s2, s3, s4
    # Sums: 1+s2, 1+s3, 1+s4, s2+s3, s2+s4, s3+s4
    
    for s2 in others:
        for s3 in others:
            if s3 <= s2: continue
            sum_1_2 = (1 + s2) % N
            sum_1_3 = (1 + s3) % N
            sum_2_3 = (s2 + s3) % N
            if sum_1_2 == sum_1_3 or sum_1_2 == sum_2_3 or sum_1_3 == sum_2_3:
                continue
            
            for s4 in others:
                if s4 <= s3: continue
                sum_1_4 = (1 + s4) % N
                sum_2_4 = (s2 + s4) % N
                sum_3_4 = (s3 + s4) % N
                
                sums = [sum_1_2, sum_1_3, sum_2_3, sum_1_4, sum_2_4, sum_3_4]
                if len(set(sums)) < 6:
                    continue
                
                # Check 6-cycles and 8-cycles via BFS
                # This is expensive, but we only do it for candidates passing 4-cycle check.
                if _check_girth_10_cayley([1, s2, s3, s4]):
                    found = [1, s2, s3, s4]
                    break
            if found: break
        if found: break
        
    return found

def _check_girth_10_cayley(gens):
    N = 320
    adj = [[] for _ in range(N)]
    for g in gens:
        for i in range(N):
            adj[i].append((i + g) % N)
            adj[i].append((i - g) % N)
            
    # BFS
    for start in range(N):
        dist = [-1] * N
        parent = [-1] * N
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= 4:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    if parent[u] != v and parent[v] != u:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < 10:
                            return False
    return True
