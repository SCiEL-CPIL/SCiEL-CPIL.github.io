"""K3,3-free graph construction on 20 vertices."""
import random
import time

def construct():
    start = time.time()
    deadline = start + 110  # safety margin before 150s kill

    n = 20
    # All possible edges as (u, v) with u < v
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def popcount(x):
        cnt = 0
        while x:
            x &= x - 1
            cnt += 1
        return cnt

    def is_k33_free(adj):
        """Check if graph with adjacency bitmasks is K3,3-free."""
        for a in range(n):
            for b in range(a + 1, n):
                common_ab = adj[a] & adj[b]
                if popcount(common_ab) < 3:
                    continue
                for c in range(b + 1, n):
                    common_abc = common_ab & adj[c]
                    if popcount(common_abc) >= 3:
                        return False
        return True

    def try_add_edge(adj, u, v):
        """Try to add edge (u,v) to adj. Returns True if valid.
        
        A new K3,3 must use the new edge (u,v). WLOG u is in left part S, v in right part T.
        S = {u, a, b}, T = {v, c, d}
        Required edges: u-v (new), u-c, u-d, a-v, a-c, a-d, b-v, b-c, b-d
        So:
        - c, d must be neighbors of u (in N_u).
        - a, b must be neighbors of v (in N_v).
        - a, b must be neighbors of c and d.
        - c, d must be neighbors of a and b (implied by a,b being neighbors of c,d? No, undirected).
        
        Let's re-verify:
        If we pick c, d from N_u:
        Then we need a, b such that:
        1. a, b in N_v
        2. a, b in N_c
        3. a, b in N_d
        
        So candidates for a, b are N_v & N_c & N_d.
        We need at least 2 such distinct vertices (a, b).
        Also a, b must be distinct from u, v, c, d.
        Since a, b in N_v, a,b != v.
        Since c, d in N_u, and u,v not adjacent yet, c,d != v.
        Since a, b in N_v, and u not in N_v, a,b != u.
        Since c, d in N_u, c,d != u.
        Could a or b be c or d?
        If a = c, then c in N_v. But c in N_u. So u-c and v-c exist.
        If a=c, then S={u, c, b}, T={v, c, d}. But S and T must be disjoint.
        So we must ensure a, b are not in {c, d}.
        Similarly, c, d are not in {a, b}.
        
        So candidates = N_v & N_c & N_d
        Remove u, v, c, d from candidates.
        If popcount(candidates) >= 2, then we can pick a, b from candidates.
        Do we need to check if a, b are adjacent? No, K3,3 subgraph allows edges within parts.
        Do we need to check if c, d are adjacent? No.
        
        So the check is: for all pairs (c, d) in N_u, is popcount((N_v & N_c & N_d) & ~(1<<u) & ~(1<<v) & ~(1<<c) & ~(1<<d)) >= 2?
        If yes for any pair, adding (u,v) creates a K3,3.
        """
        N_u = adj[u]
        N_v = adj[v]
        
        # Iterate over pairs c, d in N_u
        cu = N_u
        while cu:
            lsb_c = cu & (-cu)
            cu ^= lsb_c
            c = lsb_c.bit_length() - 1
            
            cd = N_u & (~lsb_c)
            while cd:
                lsb_d = cd & (-cd)
                cd ^= lsb_d
                d = lsb_d.bit_length() - 1
                
                # Candidates for a, b
                candidates = N_v & adj[c] & adj[d]
                # Exclude u, v, c, d
                candidates &= ~(1 << u)
                candidates &= ~(1 << v)
                candidates &= ~(1 << c)
                candidates &= ~(1 << d)
                
                if popcount(candidates) >= 2:
                    return False
        return True

    def greedy_construct():
        adj = [0] * n
        random.shuffle(all_edges)
        for u, v in all_edges:
            if try_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        # Count edges
        cnt = 0
        for i in range(n):
            cnt += popcount(adj[i])
        return cnt // 2, adj

    def local_search(adj, max_iters=200000):
        adj = adj[:]
        current_edges = sum(popcount(adj[i]) for i in range(n)) // 2
        
        # Precompute list of non-edges for faster sampling?
        # Or just sample random pairs.
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Try to add a random non-edge
            u, v = random.choice(all_edges)
            if not (adj[u] & (1 << v)):
                if try_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges += 1
                    continue
            
            # Try a swap: remove (a,b), add (c,d)
            # Only if we can't add directly, or to explore different structures
            # Let's try a 1-for-1 swap
            if random.random() < 0.1:
                # Pick a random edge to remove
                a, b = random.choice(all_edges)
                if adj[a] & (1 << b):
                    # Pick a random non-edge to add
                    c, d = random.choice(all_edges)
                    if not (adj[c] & (1 << d)):
                        # Check if removing (a,b) and adding (c,d) is valid
                        # Temporarily remove (a,b)
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)
                        
                        if try_add_edge(adj, c, d):
                            adj[c] |= (1 << d)
                            adj[d] |= (1 << c)
                            # current_edges unchanged
                        else:
                            # Revert add, restore remove
                            adj[a] |= (1 << b)
                            adj[b] |= (1 << a)
                    else:
                        # Revert remove
                        pass # didn't actually remove yet in code flow? 
                        # Wait, I did adj[a] &= ... above.
                        # I need to restore if I don't add.
                        # Let's restructure swap logic to be safer.
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)

        return current_edges, adj

    best_edges = 0
    best_adj = None
    num_trials = 50
    
    # Run multiple greedy constructions
    for trial in range(num_trials):
        if time.time() > deadline - 15:
            break
        edges, adj = greedy_construct()
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            
    # Local search on the best
    if best_adj is not None and time.time() < deadline - 10:
        best_edges, best_adj = local_search(best_adj)
        
    # Final verification and polish
    # Try to add any remaining edges in a deterministic order
    # This helps if local search missed some easy additions
    for u, v in all_edges:
        if time.time() > deadline - 2:
            break
        if not (best_adj[u] & (1 << v)):
            if try_add_edge(best_adj, u, v):
                best_adj[u] |= (1 << v)
                best_adj[v] |= (1 << u)
                best_edges += 1

    # Convert adj to edge list
    edges_list = []
    for u in range(n):
        v_mask = best_adj[u]
        while v_mask:
            lsb = v_mask & (-v_mask)
            v_mask ^= lsb
            v = lsb.bit_length() - 1
            if u < v:
                edges_list.append([u, v])
    
    return edges_list
