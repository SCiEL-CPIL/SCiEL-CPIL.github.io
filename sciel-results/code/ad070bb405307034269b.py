import random
import time

def construct():
    n = 40
    start_time = time.time()
    limit = 115  # seconds

    best_edges = 0
    best_edges_list = []

    def creates_c6_fast(u, v, adj):
        """
        Check if adding edge (u, v) creates a C6.
        We need to check for paths of length 4 between u and v that are internally disjoint and avoid u,v.
        Path: u - a - b - c - d - v
        Constraints:
        a in N(u), b in N(a), c in N(b), d in N(c)
        d in N(v)
        a,b,c,d distinct and not in {u,v}
        a != d (implied by distinctness)
        b != c (implied by simple graph)
        
        Efficient check:
        For each a in N(u):
            For each b in N(a) \ {u}:
                For each c in N(b) \ {u, a}:
                    # We need d in N(c) ∩ N(v) \ {u, a, b, c, v}
                    # If such d exists, C6 is formed.
                    candidates = adj[c] & adj[v]
                    # Exclude u, a, b, c, v
                    # Note: c is in adj[c] but not in adj[v] usually unless c==v (excluded) or c adjacent to v.
                    # But d must be distinct from c.
                    # Also d must not be u, a, b.
                    if candidates:
                        for d in candidates:
                            if d not in (u, a, b, c, v):
                                return True
        return False
    """
        pass # Placeholder to be replaced

    # Better implementation of creates_c6
    def creates_c6(u, v, adj):
        # We look for u-a-b-c-d-v
        # a in N(u)
        # b in N(a)
        # c in N(b)
        # d in N(c) ∩ N(v)
        # All distinct: u, a, b, c, d, v
        for a in adj[u]:
            if a == v: continue
            for b in adj[a]:
                if b in (u, v, a): continue
                for c in adj[b]:
                    if c in (u, v, a, b): continue
                    # d must be in adj[c] and adj[v]
                    # d not in {u, v, a, b, c}
                    # Iterate over intersection for speed
                    inter = adj[c] & adj[v]
                    for d in inter:
                        if d not in (u, v, a, b, c):
                            return True
        return False

    def try_add_edge(u, v, adj):
        if v in adj[u]:
            return False
        if creates_c6(u, v, adj):
            return False
        adj[u].add(v)
        adj[v].add(u)
        return True

    def count_edges(adj):
        return sum(len(s) for s in adj) // 2

    def get_edges_list(adj):
        return [[u, v] for u in range(n) for v in adj[u] if u < v]

    # Strategy:
    # 1. Multiple restarts.
    # 2. Initial construction: Randomized greedy with degree balancing? Or just random order.
    #    To get high edge count, we want to avoid creating C6s early.
    #    A common heuristic for C6-free graphs is to keep degrees low initially.
    #    However, we want to maximize edges, so we should try to add edges aggressively.
    #    Let's try a "conflict-avoiding" greedy:
    #    Pick edges in random order, add if no C6.
    
    # 3. Local Search:
    #    - Add missing edges if possible (saturation).
    #    - Swaps: Remove 1 edge, try to add 2.
    
    for restart in range(100):
        if time.time() - start_time > limit - 2:
            break
            
        adj = [set() for _ in range(n)]
        
        # Initialization: Randomized Greedy
        # To improve results, we can try different initialization strategies.
        # Strategy A: Random order
        # Strategy B: Degree-biased (prefer low degree vertices?)
        # Let's stick to random order but with a twist:
        # We can try to add edges that are "less likely" to create C6s?
        # Hard to know without checking.
        
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            try_add_edge(u, v, adj)
            
        # Saturation Phase
        # Keep adding missing edges until none can be added.
        # We maintain a list of missing edges to avoid scanning all pairs repeatedly.
        # But since graph changes, missing list changes.
        # Simple approach: Loop until no change.
        
        max_iters = 100
        for _ in range(max_iters):
            if time.time() - start_time > limit - 2:
                break
            added = False
            # Get current missing edges
            missing = [(u, v) for u in range(n) for v in range(u+1, n) if v not in adj[u]]
            random.shuffle(missing)
            for u, v in missing:
                if try_add_edge(u, v, adj):
                    added = True
            if not added:
                break
                
        current_edges = count_edges(adj)
        
        # Update best
        if current_edges > best_edges:
            best_edges = current_edges
            best_edges_list = get_edges_list(adj)
            
        # Local Search with Swaps (1 out, 2 in)
        # This can improve the edge count if the graph is not "maximal" in a strong sense.
        
        swap_attempts = 0
        max_swaps = 5000 # Limit per restart
        
        while swap_attempts < max_swaps and time.time() - start_time < limit - 2:
            swap_attempts += 1
            
            existing = [(u, v) for u in range(n) for v in adj[u] if u < v]
            if not existing:
                break
                
            missing = [(u, v) for u in range(n) for v in range(u+1, n) if v not in adj[u]]
            if not missing:
                break
                
            # Pick a random edge to remove
            ru, rv = random.choice(existing)
            
            # Remove it
            adj[ru].remove(rv)
            adj[rv].remove(ru)
            
            # Try to add 2 edges
            added_count = 0
            added_edges = []
            
            # Shuffle missing to avoid bias
            random.shuffle(missing)
            
            # Try to add up to 2 edges
            # We should prioritize edges that don't conflict with each other?
            # If we add (u1, v1) and (u2, v2), they might create a C6 together?
            # No, C6 check is local to the addition. 
            # However, adding (u1, v1) might make (u2, v2) invalid if they form a C6 with existing paths.
            # The `creates_c6` check handles this.
            
            for u, v in missing:
                if try_add_edge(u, v, adj):
                    added_count += 1
                    added_edges.append((u, v))
                    if added_count == 2:
                        break
                        
            if added_count == 2:
                # Net gain +1. Keep.
                current_edges += 1
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_edges_list = get_edges_list(adj)
            elif added_count == 1:
                # Net gain 0. 
                # Accept with probability 0.5 to explore?
                # Or always accept to change structure?
                # Let's accept to move to a different local optimum.
                pass
            else:
                # Net gain -1. Revert.
                adj[ru].add(rv)
                adj[rv].add(ru)
                
                # Revert added edges? 
                # If added_count == 0, no edges were added.
                # If added_count == 1, we added one edge. We need to revert it if we want to revert the whole move?
                # Wait, if added_count == 1, net is 0. We accepted it. So we keep the state with 1 new edge.
                # If added_count == 0, we revert the removal.
                
                # But what if added_count == 1 and we want to revert? 
                # In the `else` block, added_count is 0. So we just re-add (ru, rv).
                
                # If added_count == 1, we are in `elif`, so we keep the state.
                
        # End swap phase
        
        # Final saturation after swaps
        for _ in range(50):
            if time.time() - start_time > limit - 2:
                break
            added = False
            missing = [(u, v) for u in range(n) for v in range(u+1, n) if v not in adj[u]]
            random.shuffle(missing)
            for u, v in missing:
                if try_add_edge(u, v, adj):
                    added = True
            if not added:
                break
                
        current_edges = count_edges(adj)
        if current_edges > best_edges:
            best_edges = current_edges
            best_edges_list = get_edges_list(adj)

    return best_edges_list
