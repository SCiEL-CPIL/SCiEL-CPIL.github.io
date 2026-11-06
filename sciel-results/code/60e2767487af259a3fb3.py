import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    N = 180
    
    # Precompute conflicts
    # conflicts[a] contains list of (b, c) such that 1/a = 1/b + 1/c with a < b < c
    conflicts = {i: [] for i in range(1, N + 1)}
    
    for a in range(1, N + 1):
        a2 = a * a
        # Find factor pairs d, e of a^2 such that d < e
        # b = a + d, c = a + e
        # We need b, c <= N
        # d * e = a^2
        for d in range(1, int(a2**0.5) + 1):
            if a2 % d != 0:
                continue
            e = a2 // d
            if d >= e:
                continue # We want distinct b, c, so d != e. If d==e, b==c.
            b = a + d
            c = a + e
            if b <= N and c <= N:
                conflicts[a].append((b, c))
                
    # Greedy construction: add from N down to 1
    S = set()
    # Using a list for S membership check is O(1) if we use a boolean array or set
    # Set is fine for N=180
    
    for a in range(N, 0, -1):
        if time.time() > deadline:
            break
            
        # Check if adding a creates a conflict with existing S
        # A conflict exists if there is (b, c) in conflicts[a] such that b in S and c in S
        can_add = True
        for b, c in conflicts[a]:
            if b in S and c in S:
                can_add = False
                break
        
        if can_add:
            S.add(a)
            
    # If we have time, try to improve with local search?
    # Simple local search: try to add any excluded element by removing one conflict partner
    # Or just return the greedy result.
    # Given the small N, the greedy result is likely very close to optimal.
    
    return sorted(S)
