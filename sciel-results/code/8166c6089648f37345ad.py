def construct():
    n = 256
    
    # Precompute pairs and triples that complete 3-APs and 4-APs when x is added
    # For each x, store:
    # pairs_3[x]: list of (y1, y2) such that {x, y1, y2} is a 3-AP
    # triples_4[x]: list of (y1, y2, y3) such that {x, y1, y2, y3} is a 4-AP
    
    pairs_3 = [[] for _ in range(n + 1)]
    triples_4 = [[] for _ in range(n + 1)]
    
    # Generate all 3-APs
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 2 + 1):
            b = a + d
            c = a + 2 * d
            # x=a, others b,c
            pairs_3[a].append((b, c))
            # x=b, others a,c
            pairs_3[b].append((a, c))
            # x=c, others a,b
            pairs_3[c].append((a, b))
            
    # Generate all 4-APs
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # x=a, others b,c,e
            triples_4[a].append((b, c, e))
            # x=b, others a,c,e
            triples_4[b].append((a, c, e))
            # x=c, others a,b,e
            triples_4[c].append((a, b, e))
            # x=e, others a,b,c
            triples_4[e].append((a, b, c))
            
    in_A = [False] * (n + 1)
    A_count = 0
    total_3aps = 0
    
    # Greedy loop
    for _ in range(n):
        best_x = -1
        best_gain = -1
        
        for x in range(1, n + 1):
            if in_A[x]:
                continue
            
            # Check 4-AP constraint
            valid = True
            for y1, y2, y3 in triples_4[x]:
                if in_A[y1] and in_A[y2] and in_A[y3]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Count 3-AP gain
            gain = 0
            for y1, y2 in pairs_3[x]:
                if in_A[y1] and in_A[y2]:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_x = x
        
        if best_x == -1:
            break
        
        in_A[best_x] = True
        A_count += 1
        total_3aps += best_gain
        
    return [i for i in range(1, n + 1) if in_A[i]]
