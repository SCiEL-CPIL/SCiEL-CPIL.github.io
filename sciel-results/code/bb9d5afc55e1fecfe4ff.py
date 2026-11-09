import time
import numpy as np
from itertools import combinations

N = 20
DEADLINE_MARGIN = 5.0  # seconds before the 150s limit to stop

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN  # We have 150s, use 145s to be safe?
    # The prompt says "KILLS the program after 150s wall-clock".
    # Let's set deadline to start_time + 140 to be very safe.
    deadline = start_time + 140.0

    # Initialize with a random graph or empty?
    # Let's start with a random graph with p=0.5 to have a good starting point.
    # Or start with complete graph and remove edges?
    # Starting with complete graph (190 edges) and removing is likely faster to converge to high density.
    # But checking K2,3 violations on complete graph is heavy.
    # Let's try starting with a random graph with p=0.6.
    
    best_edges = None
    best_count = -1
    
    # Precompute all pairs
    all_pairs = list(combinations(range(N), 2))
    pair_index = {p: i for i, p in enumerate(all_pairs)}
    M = len(all_pairs)
    
    # Precompute adjacency matrix of pairs?
    # For fast violation counting, we need to check for any pair (u,v) if they have >= 3 common neighbors.
    # Common neighbors of u and v are vertices w such that (u,w) and (v,w) are edges.
    
    # Let's use a bitset representation for adjacency for speed.
    # adj[i] is an integer where bit j is set if (i,j) is an edge.
    
    def get_violations(adj):
        # Returns list of pairs (u,v) that have >= 3 common neighbors.
        # We want to minimize the number of such pairs, or rather, ensure none exist.
        # Actually, we just need a valid graph.
        # For local search, we can count the "badness".
        # Badness = sum over all pairs (u,v) of max(0, common_neighbors(u,v) - 2)
        # If badness == 0, it's valid.
        
        badness = 0
        bad_pairs = []
        for u in range(N):
            for v in range(u+1, N):
                # Common neighbors
                # adj[u] & adj[v] gives bitmask of common neighbors
                # We need to exclude u and v themselves (they aren't neighbors of themselves in simple graph, so bits u and v are 0 in adj[u] and adj[v] respectively, but adj[u] has bit v set if edge, adj[v] has bit u set).
                # Wait, adj[u] contains neighbors of u. So bit v is set if (u,v) is edge.
                # Common neighbors w must be w != u and w != v.
                # Since it's a simple graph, u is not in adj[u], v is not in adj[v].
                # But u is in adj[v] if (u,v) is edge.
                # So adj[u] & adj[v] will have bit u set if (v,u) is edge (which is same as (u,v)).
                # And bit v set if (u,v) is edge.
                # So we must mask out u and v.
                mask = ~(1 << u) & ~(1 << v)
                common = adj[u] & adj[v] & mask
                count = bin(common).count('1')
                if count > 2:
                    badness += count - 2
                    bad_pairs.append((u, v))
        return badness, bad_pairs

    def try_flip(adj, u, v, current_badness, current_bad_pairs):
        # Try adding or removing edge (u,v).
        # We need to update badness efficiently?
        # Given N=20, full recalculation is O(N^2 * N) = 8000 ops.
        # 10000 flips * 8000 = 80M ops. In Python this might be slow.
        # Let's try to optimize or use NumPy.
        
        pass

    # Given the small N, let's use a simpler, robust approach:
    # 1. Start with a random graph.
    # 2. While badness > 0:
    #    - Pick a random bad pair (u,v).
    #    - Pick a random common neighbor w of u,v (or u or v themselves if we want to remove edge).
    #    - Try removing one of the edges incident to u or v to reduce common neighbors.
    #      Specifically, remove (u,w) or (v,w).
    # 3. Once valid, try to add edges to maximize count.
    #    - Pick random non-edge (u,v).
    #    - If adding it creates no K2,3, add it.
    #    - If it creates K2,3, maybe try a swap?
    
    # Let's implement the "Remove to fix, then Add to maximize" strategy.
    
    def is_valid(adj):
        # Check if no K2,3
        for u in range(N):
            for v in range(u+1, N):
                mask = ~(1 << u) & ~(1 << v)
                common = adj[u] & adj[v] & mask
                if bin(common).count('1') > 2:
                    return False
        return True

    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(N)) // 2

    def random_graph(p):
        adj = [0] * N
        for u in range(N):
            for v in range(u+1, N):
                if np.random.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def fix_graph(adj):
        # Remove edges until valid
        for _ in range(1000): # Max iterations
            if is_valid(adj):
                return adj
            # Find a bad pair
            bad_pair = None
            for u in range(N):
                for v in range(u+1, N):
                    mask = ~(1 << u) & ~(1 << v)
                    common = adj[u] & adj[v] & mask
                    if bin(common).count('1') > 2:
                        bad_pair = (u, v, common)
                        break
                if bad_pair:
                    break
            
            if bad_pair is None:
                return adj # Should be valid
            
            u, v, common = bad_pair
            # Pick a random common neighbor to break
            # We need to remove (u, w) or (v, w) for some w in common.
            # Or remove (u, v) itself? No, (u,v) doesn't affect common neighbors of u,v.
            # Wait, if we remove (u,v), does it change common neighbors?
            # Common neighbors are w != u,v such that (u,w) and (v,w) exist.
            # Removing (u,v) does not remove any common neighbor w.
            # So we MUST remove an edge (u,w) or (v,w) where w is a common neighbor.
            
            # Get list of common neighbors
            w_list = []
            temp = common
            while temp:
                lsb = temp & -temp
                w = lsb.bit_length() - 1
                w_list.append(w)
                temp ^= lsb
            
            # Pick a random w
            w = np.random.choice(w_list)
            
            # Decide whether to remove (u,w) or (v,w)
            # Remove the one that seems "less useful" or just random.
            if np.random.random() < 0.5:
                # Remove (u,w)
                adj[u] &= ~(1 << w)
                adj[w] &= ~(1 << u)
            else:
                # Remove (v,w)
                adj[v] &= ~(1 << w)
                adj[w] &= ~(1 << v)
        
        return adj

    def maximize_edges(adj):
        # Try to add edges
        improved = True
        while improved:
            improved = False
            # Shuffle order to avoid bias
            pairs = list(combinations(range(N), 2))
            np.random.shuffle(pairs)
            
            for u, v in pairs:
                if time.time() > deadline:
                    break
                # Check if edge exists
                if adj[u] & (1 << v):
                    continue
                
                # Check if adding (u,v) keeps it valid
                # Adding (u,v) creates new common neighbors for pairs (u,x) and (v,x) where x is neighbor of both?
                # No, adding (u,v) means:
                # 1. For any w, if (u,w) and (v,w) existed, they already had w as common neighbor.
                #    Adding (u,v) does not change common neighbors of (u,w) or (v,w) for w != u,v.
                #    Wait.
                #    Let's check K2,3 condition for all pairs.
                #    A pair (a,b) has common neighbors C(a,b).
                #    If we add (u,v):
                #    - If (a,b) is (u,v), new common neighbors are C(u,v) (old) + ... wait.
                #      Common neighbors of u and v are w such that (u,w) and (v,w) are edges.
                #      Adding (u,v) does NOT add any new common neighbors to the pair (u,v).
                #      Because w must be distinct from u and v.
                #      So C(u,v) is unchanged by adding (u,v).
                #    - If (a,b) is (u,w) where w != v:
                #      Common neighbors of u and w are x such that (u,x) and (w,x) are edges.
                #      Adding (u,v) adds v to the neighbors of u.
                #      So if (w,v) is an edge, then v becomes a new common neighbor of u and w.
                #      So C(u,w) increases by 1 if (w,v) is an edge.
                #    - Similarly, if (a,b) is (v,w) where w != u:
                #      C(v,w) increases by 1 if (u,w) is an edge.
                #    - If (a,b) is (x,y) where x,y not in {u,v}:
                #      Unchanged.
                
                # So we only need to check:
                # 1. Is C(u,v) <= 2? (It was valid before, so yes, unless it was exactly 2 and we don't change it, so still valid).
                #    Actually, C(u,v) is unchanged. So if it was valid, it remains valid for pair (u,v).
                # 2. For all w in N \ {u,v}:
                #    If (w,v) is edge, check if C(u,w) becomes > 2.
                #    If (w,u) is edge, check if C(v,w) becomes > 2.
                
                # Let's check these conditions.
                ok = True
                # Check pairs (u, w) for all w neighbor of v
                # And pairs (v, w) for all w neighbor of u
                
                # Neighbors of v (excluding u)
                mask_v = adj[v] & ~(1 << u)
                temp = mask_v
                while temp:
                    lsb = temp & -temp
                    w = lsb.bit_length() - 1
                    temp ^= lsb
                    # Check C(u, w)
                    # Current C(u, w)
                    mask_uw = ~(1 << u) & ~(1 << w)
                    common_uw = adj[u] & adj[w] & mask_uw
                    count_uw = bin(common_uw).count('1')
                    # Adding (u,v) adds v to C(u,w) if (w,v) is edge.
                    # We are iterating w from neighbors of v, so (w,v) IS an edge.
                    # So new count is count_uw + 1.
                    if count_uw + 1 > 2:
                        ok = False
                        break
                
                if not ok:
                    continue
                    
                # Check pairs (v, w) for all w neighbor of u
                mask_u = adj[u] & ~(1 << v)
                temp = mask_u
                while temp:
                    lsb = temp & -temp
                    w = lsb.bit_length() - 1
                    temp ^= lsb
                    # Check C(v, w)
                    mask_vw = ~(1 << v) & ~(1 << w)
                    common_vw = adj[v] & adj[w] & mask_vw
                    count_vw = bin(common_vw).count('1')
                    # Adding (u,v) adds u to C(v,w) if (w,u) is edge.
                    # We are iterating w from neighbors of u, so (w,u) IS an edge.
                    # So new count is count_vw + 1.
                    if count_vw + 1 > 2:
                        ok = False
                        break
                
                if ok:
                    # Add edge
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
            
            if time.time() > deadline:
                break
        
        return adj

    # Main loop: Multiple restarts
    best_adj = None
    best_count = -1
    
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Start with random graph
        p = 0.5 + 0.1 * (i % 5) # Vary density
        adj = random_graph(p)
        
        # Fix it
        adj = fix_graph(adj)
        
        # Maximize
        adj = maximize_edges(adj)
        
        # Check validity again (paranoia)
        if not is_valid(adj):
            # Should not happen, but if it does, skip
            continue
            
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
            
    if best_adj is None:
        # Fallback to seed
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges

    # Convert adj to list of edges
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges
