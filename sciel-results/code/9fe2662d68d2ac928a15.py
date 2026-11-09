import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for 150s limit
    
    # 1. Try to find a valid graph for n=40
    # 2. Try to find a valid graph for n=39
    # 3. Fallback to Andrasfai(9) which is n=27, alpha=9
    
    for n in [40, 39]:
        if time.time() > deadline:
            break
            
        # Allocate time for this n
        # We don't know how long it will take, so we just run until deadline or success
        # But we should interleave or just let it run.
        # Let's give each n a specific time budget or just run until deadline.
        # Since 40 is harder, if it fails quickly, we move to 39.
        
        result = _local_search(n, deadline)
        if result is not None:
            return result
            
    # Fallback: Andrasfai graph And(9)
    # n = 27, alpha = 9
    return _andrasfai(9)

def _local_search(n, deadline):
    """
    Attempts to find a triangle-free graph on n vertices with alpha < 10.
    Uses randomized local search (greedy edge addition to break independent sets).
    """
    num_restarts = 0
    max_restarts = 50  # Cap restarts to avoid infinite loop if time is low
    
    while time.time() < deadline and num_restarts < max_restarts:
        num_restarts += 1
        if time.time() > deadline:
            break
            
        # 1. Initialize random maximal triangle-free graph
        adj = [0] * n
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            if adj[u] & (1 << v) == 0 and adj[v] & (1 << u) == 0:
                # Check triangle-free condition
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
        # 2. Local Search
        # We want to reduce alpha.
        # We look for independent sets of size 10.
        # If found, we try to add edges between vertices in that set.
        
        # To speed up, we can just check if alpha < 10 directly.
        # But finding a specific independent set of size 10 is useful for guidance.
        
        # Let's use a loop that runs for a fixed number of steps or time slice
        steps = 0
        max_steps = 2000  # Adjust based on n
        
        while steps < max_steps and time.time() < deadline:
            steps += 1
            
            # Check if valid
            if not _has_independent_set(adj, n, 10):
                # Valid!
                edge_list = []
                for i in range(n):
                    mask = adj[i]
                    j = i + 1
                    while mask:
                        if mask & 1:
                            if j > i:
                                edge_list.append([i, j])
                        mask >>= 1
                        j += 1
                return {'n': n, 'edges': edge_list}
            
            # Find an independent set of size 10 to break
            # We can use a greedy approach to find a large independent set
            # Or just pick random pairs that can be connected
            
            # Greedy: Find a vertex with low degree in the "independent set potential"
            # Simpler: Pick a random pair (u, v) not connected, check if they can be connected.
            # If yes, and if they are "far apart" in some sense, add edge.
            
            # Better: Find a maximal independent set.
            # Greedy MIS:
            mis = []
            remaining = (1 << n) - 1
            while remaining:
                # Pick vertex with min degree in remaining?
                # Or random
                v_list = []
                r = remaining
                while r:
                    lsb = r & -r
                    v = lsb.bit_length() - 1
                    v_list.append(v)
                    r ^= lsb
                v = random.choice(v_list)
                mis.append(v)
                remaining &= ~adj[v]
                remaining &= ~(1 << v)
                
            if len(mis) < 10:
                # We didn't find a size 10 set with this greedy?
                # But _has_independent_set said yes.
                # Greedy MIS is not exact.
                # Let's just try to add edges from the greedy MIS
                pass
                
            # Try to add an edge between two vertices in the greedy MIS
            # that are not connected and have no common neighbors
            candidates = []
            for i in range(len(mis)):
                for j in range(i + 1, len(mis)):
                    u = mis[i]
                    v = mis[j]
                    if not (adj[u] & (1 << v)):
                        if (adj[u] & adj[v]) == 0:
                            candidates.append((u, v))
                            
            if candidates:
                u, v = random.choice(candidates)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            else:
                # Stuck. Try removing a random edge to escape local optimum?
                # Or just break and restart
                # Removing edges increases alpha, so it's bad.
                # But it might allow adding *different* edges later.
                # Let's just restart this iteration
                break

    return None

def _has_independent_set(adj, n, k):
    """
    Returns True if there is an independent set of size k in G.
    Equivalent to: Clique of size k in complement.
    """
    # Build complement masks
    comp = [0] * n
    full_mask = (1 << n) - 1
    for i in range(n):
        comp[i] = full_mask ^ adj[i] ^ (1 << i)
        
    # Branch and Bound for Max Clique in complement
    # We just need to know if size >= k exists
    
    # Order vertices by degree in complement (descending)
    degrees = [bin(comp[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda x: -degrees[x])
    
    # Remap
    new_comp = [0] * n
    remap = [0] * n
    for new_idx, old_idx in enumerate(order):
        remap[old_idx] = new_idx
        mask = 0
        c = comp[old_idx]
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            mask |= (1 << remap[v])
            c ^= lsb
        new_comp[new_idx] = mask
        
    # B&B
    # State: candidates mask, current_size
    # Pruning: current_size + popcount(candidates) < k
    
    def popcount(x):
        return bin(x).count('1')
        
    def bnb(candidates, current_size):
        if current_size >= k:
            return True
        if current_size + popcount(candidates) < k:
            return False
            
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            c ^= lsb
            
            # Prune: if v's neighbors in candidates + current_size < k
            # deg_in_cands = popcount(candidates & new_comp[v])
            # if current_size + 1 + deg_in_cands < k: continue
            
            new_candidates = candidates & new_comp[v]
            if bnb(new_candidates, current_size + 1):
                return True
        return False
        
    return bnb((1 << n) - 1, 0)

def _andrasfai(k):
    """
    Constructs Andrasfai graph And(k).
    n = 3k
    Triangle-free, alpha = k
    """
    n = 3 * k
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if 1 <= diff <= k - 1 or 2 * k + 1 <= diff <= 3 * k - 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    edge_list = []
    for i in range(n):
        mask = adj[i]
        j = i + 1
        while mask:
            if mask & 1:
                if j > i:
                    edge_list.append([i, j])
            mask >>= 1
            j += 1
            
    return {'n': n, 'edges': edge_list}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
