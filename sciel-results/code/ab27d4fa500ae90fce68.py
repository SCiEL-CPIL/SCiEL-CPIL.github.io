import time

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit (evaluator kills at 90s, but keep buffer)
    
    # Strategy: Start with K_7,8 (56 edges), then try to add edges within partitions
    # while maintaining the constraint that no 3-edge path has endpoints of equal degree.
    
    # Initial adjacency as bitmasks for speed
    adj = [0] * n
    
    # Build K_7,8: A = {0..6}, B = {7..14}
    for u in range(7):
        for v in range(7, 15):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    def compute_degrees(adj):
        return [bin(adj[i]).count('1') for i in range(n)]
    
    def is_valid(adj):
        """Check if no 3-edge path s-u-v-t has deg(s) == deg(t)."""
        degrees = compute_degrees(adj)
        for u in range(n):
            au = adj[u]
            for v in range(n):
                if not (au & (1 << v)):
                    continue
                if u > v:
                    continue
                # Edge (u, v) exists
                # s is a neighbor of u other than v
                s_mask = au & ~(1 << v)
                # t is a neighbor of v other than u
                t_mask = adj[v] & ~(1 << u)
                
                # For each s in s_mask, for each t in t_mask where s != t
                # check if degrees[s] == degrees[t]
                # Optimization: group t's by degree
                t_degrees = {}
                tm = t_mask
                while tm:
                    bit = tm & -tm
                    t = bit.bit_length() - 1
                    tm ^= bit
                    d = degrees[t]
                    if d not in t_degrees:
                        t_degrees[d] = 0
                    t_degrees[d] += 1
                
                sm = s_mask
                while sm:
                    bit = sm & -sm
                    s = bit.bit_length() - 1
                    sm ^= bit
                    ds = degrees[s]
                    # If there's any t with degree ds and t != s, then invalid
                    if ds in t_degrees:
                        # Check if all t's with degree ds are just s itself
                        # We need to check if s is the ONLY t with degree ds
                        # Let's count how many t's have degree ds
                        # Actually, we just need to know if there exists t != s with deg(t) == deg(s)
                        # Let's re-iterate t's of that specific degree to check
                        # Better approach: precompute set of t's with each degree
                        pass
        
        # Simpler but still fast check for n=15:
        # Just iterate all pairs
        degrees = compute_degrees(adj)
        for u in range(n):
            au = adj[u]
            for v in range(n):
                if not (au & (1 << v)):
                    continue
                if u > v:
                    continue
                s_mask = au & ~(1 << v)
                t_mask = adj[v] & ~(1 << u)
                
                sm = s_mask
                while sm:
                    s_bit = sm & -sm
                    s = s_bit.bit_length() - 1
                    sm ^= s_bit
                    
                    tm = t_mask & ~(1 << s)  # t != s
                    while tm:
                        t_bit = tm & -tm
                        t = t_bit.bit_length() - 1
                        tm ^= t_bit
                        
                        if degrees[s] == degrees[t]:
                            return False
        return True
    
    def count_edges(adj):
        total = 0
        for i in range(n):
            total += bin(adj[i]).count('1')
        return total // 2
    
    best_adj = adj[:]
    best_count = count_edges(best_adj)
    
    # Try to add edges within A (0..6) or within B (7..14)
    # These are the only missing edges in K_7,8
    candidate_edges = []
    for i in range(7):
        for j in range(i + 1, 7):
            candidate_edges.append((i, j))
    for i in range(7, 15):
        for j in range(i + 1, 15):
            candidate_edges.append((i, j))
    
    # Try adding each candidate edge if valid
    for u, v in candidate_edges:
        if time.time() > deadline:
            break
        new_adj = best_adj[:]
        new_adj[u] |= (1 << v)
        new_adj[v] |= (1 << u)
        if is_valid(new_adj):
            best_adj = new_adj
            best_count += 1
    
    # Local search: try to add more edges by checking all missing edges
    # (in case adding one enables another)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for u in range(n):
            if time.time() > deadline:
                break
            for v in range(u + 1, n):
                if time.time() > deadline:
                    break
                if not (best_adj[u] & (1 << v)):
                    new_adj = best_adj[:]
                    new_adj[u] |= (1 << v)
                    new_adj[v] |= (1 << u)
                    if is_valid(new_adj):
                        best_adj = new_adj
                        best_count += 1
                        improved = True
    
    # Build final edge list
    final_edges = []
    for i in range(n):
        ai = best_adj[i]
        for j in range(i + 1, n):
            if ai & (1 << j):
                final_edges.append([int(i), int(j)])
    
    return final_edges
