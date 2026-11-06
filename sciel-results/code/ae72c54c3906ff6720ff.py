import time
import random
import sys

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We stop at 115s.
    deadline = start_time + 115.0
    
    n = 20
    best_count = 0
    best_adj = [0] * n
    
    def is_valid_add(adj, u, v):
        # Check if u and v are already connected
        if (adj[u] >> v) & 1:
            return False
        
        # 1. Common neighbors of u and v must be <= 2
        common_uv = adj[u] & adj[v]
        if common_uv.bit_count() > 2:
            return False
            
        # 2. For any w in N(u), u becomes a common neighbor of v and w.
        #    So current common(v, w) must be <= 1.
        #    If common(v, w) >= 2, adding u makes it 3.
        w_mask = adj[u]
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            common_vw = adj[v] & adj[w]
            if common_vw.bit_count() > 1:
                return False
            w_mask ^= lsb
            
        # 3. For any w in N(v), v becomes a common neighbor of u and w.
        #    So current common(u, w) must be <= 1.
        w_mask = adj[v]
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            common_uw = adj[u] & adj[w]
            if common_uw.bit_count() > 1:
                return False
            w_mask ^= lsb
            
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def get_missing_edges(adj):
        missing = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] >> j) & 1:
                    missing.append((i, j))
        return missing

    def get_present_edges(adj):
        present = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    present.append((i, j))
                mask ^= lsb
        return present

    def greedy_fill(adj, count):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            missing = get_missing_edges(adj)
            random.shuffle(missing)
            for u, v in missing:
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    count += 1
                    improved = True
        return count

    def local_search(adj, count, max_iters=1000):
        # Try to add edges, if stuck, try swaps
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            missing = get_missing_edges(adj)
            if not missing:
                break
                
            added = False
            random.shuffle(missing)
            for u, v in missing:
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    count += 1
                    added = True
                    break
            
            if added:
                continue
                
            # If no edge could be added, try a swap
            # Pick a random present edge and a random missing edge
            present = get_present_edges(adj)
            if not present:
                break
                
            u1, v1 = random.choice(present)
            u2, v2 = random.choice(missing)
            
            # Try swapping (u1, v1) with (u2, v2)
            remove_edge(adj, u1, v1)
            count -= 1
            
            if is_valid_add(adj, u2, v2):
                add_edge(adj, u2, v2)
                count += 1
                # We might be able to add more now
                # Run a quick greedy pass
                count = greedy_fill(adj, count)
            else:
                # Revert
                add_edge(adj, u1, v1)
                count += 1
                
        return count

    def random_regular_graph(deg):
        """Generate a random regular graph of given degree using simple random restarts on configuration model."""
        # For n=20, deg=6, this is fast.
        while True:
            adj = [0] * n
            # Create a list of stubs
            stubs = []
            for i in range(n):
                for _ in range(deg):
                    stubs.append(i)
            random.shuffle(stubs)
            
            # Pair them up
            valid = True
            for k in range(0, len(stubs), 2):
                u = stubs[k]
                v = stubs[k+1]
                if u == v:
                    valid = False
                    break
                if (adj[u] >> v) & 1:
                    valid = False
                    break
                add_edge(adj, u, v)
            
            if valid:
                return adj

    # Run multiple restarts
    restarts = 0
    while time.time() < deadline - 10:
        # Strategy 1: Start with a 6-regular graph
        if restarts % 2 == 0:
            adj = random_regular_graph(6)
            count = n * 6 // 2
        else:
            # Strategy 2: Start with a random dense graph and prune? 
            # Or just random greedy.
            adj = [0] * n
            count = 0
            all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(all_edges)
            for u, v in all_edges:
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    count += 1
        
        # Local search to improve
        count = local_search(adj, count)
        
        if count > best_count:
            best_count = count
            best_adj = adj[:]
            
        restarts += 1
        
        if time.time() > deadline - 15:
            break

    # Construct the output list of edges
    edges = []
    for i in range(n):
        mask = best_adj[i]
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            mask ^= lsb
            
    return edges

if __name__ == "__main__":
    # Quick self-test
    start = time.time()
    res = construct()
    print(f"Found {len(res)} edges in {time.time() - start:.2f}s")
    # Verify validity
    n = 20
    adj = [0] * n
    for u, v in res:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    valid = True
    for u in range(n):
        for v in range(u+1, n):
            common = adj[u] & adj[v]
            if common.bit_count() > 2:
                valid = False
                print(f"Invalid: {u}, {v} have {common.bit_count()} common neighbors")
                break
        if not valid:
            break
    
    print(f"Valid: {valid}")
