import time
import random

def construct():
    deadline = time.time() + 140  # Leave 10s for safety
    
    # Helper to check if graph is valid
    def is_valid(adj, n):
        # Check K4
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    # Check if any pair in common is connected
                    # Iterate bits in common
                    # Efficient way: 
                    # For each w in common, check if adj[w] intersects common (excluding w? no, just check if there is an edge between any two in common)
                    # Actually, K4 is u,v,w,x. u-v, u-w, u-x, v-w, v-x, w-x.
                    # We have u-v.
                    # We need w,x in common(u,v) such that w-x is an edge.
                    # So we need a clique of size 2 in the subgraph induced by common(u,v).
                    # i.e., an edge in common(u,v).
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        # Check if w is connected to any other in common
                        if adj[w] & (common & ~(1 << w)):
                            return False
        # Check Ind6
        # Backtracking search for independent set of size 6
        def find_ind6():
            candidates = list(range(n))
            random.shuffle(candidates)
            
            def bt(cands, cur):
                if len(cur) == 6:
                    return cur
                if len(cands) < 6 - len(cur):
                    return None
                # Heuristic: try vertices with lower degree in the induced subgraph?
                # For now, just iterate
                for i in range(len(cands)):
                    v = cands[i]
                    # Check independence from cur
                    ok = True
                    for u in cur:
                        if adj[v] & (1 << u):
                            ok = False
                            break
                    if ok:
                        # Filter candidates: must be independent from v
                        new_cands = []
                        for u in cands[i+1:]:
                            if not (adj[v] & (1 << u)):
                                new_cands.append(u)
                        res = bt(new_cands, cur + [v])
                        if res:
                            return res
                return None
            
            return bt(candidates, [])
        
        if find_ind6():
            return False
        
        return True

    def get_edges(adj, n):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges

    # Start with n=17 Paley Graph
    n = 17
    adj = [0] * n
    residues = {1, 2, 4, 8, 9, 13, 15, 16}
    for i in range(n):
        for j in range(i+1, n):
            diff = (j - i) % n
            if diff in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    best_n = n
    best_adj = adj[:]
    
    # Try to extend to 35
    for target_n in range(18, 36):
        if time.time() > deadline:
            break
        
        # Time budget for this extension step
        # Decrease budget as n increases? Or keep constant?
        # Let's give each step a chunk of the remaining time.
        # Simpler: just run local search until deadline or success.
        # But we need to try multiple initializations if it fails.
        
        success = False
        # Try a few random initializations for the new vertex
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Initialize new vertex
            new_adj = [0] * target_n
            for i in range(n):
                new_adj[i] = adj[i]
            
            # Connect new vertex (index n) to existing vertices randomly
            for i in range(n):
                if random.random() < 0.5:
                    new_adj[n] |= (1 << i)
                    new_adj[i] |= (1 << n)
            
            # Local Search
            # We want to fix violations.
            # We will use a WalkSAT-like approach.
            
            current_adj = new_adj
            current_n = target_n
            best_local_adj = current_adj[:]
            best_local_viol = float('inf') # We don't count exactly, so we just track success
            
            # Run local search for a limited time or steps
            local_deadline = min(deadline, time.time() + 2.0) # 2 seconds per attempt?
            
            while time.time() < local_deadline:
                # Find a violation
                violation = None
                v_type = None
                
                # Check K4
                # Find one K4
                found_k4 = False
                for u in range(current_n):
                    for v in range(u+1, current_n):
                        if current_adj[u] & (1 << v):
                            common = current_adj[u] & current_adj[v]
                            c = common
                            while c:
                                lsb = c & -c
                                w = lsb.bit_length() - 1
                                c ^= lsb
                                if current_adj[w] & (common & ~(1 << w)):
                                    # Found K4: u, v, w, x
                                    # Need x
                                    x_mask = current_adj[w] & (common & ~(1 << w))
                                    if x_mask:
                                        lsb_x = x_mask & -x_mask
                                        x = lsb_x.bit_length() - 1
                                        violation = (u, v, w, x)
                                        v_type = 'K4'
                                        found_k4 = True
                                        break
                            if found_k4:
                                break
                    if found_k4:
                        break
                
                if not found_k4:
                    # Check Ind6
                    # Reuse the logic from is_valid but return the set
                    def find_ind6_fast(adj, n):
                        candidates = list(range(n))
                        random.shuffle(candidates)
                        
                        def bt(cands, cur):
                            if len(cur) == 6:
                                return cur
                            if len(cands) < 6 - len(cur):
                                return None
                            for i in range(len(cands)):
                                v = cands[i]
                                ok = True
                                for u in cur:
                                    if adj[v] & (1 << u):
                                        ok = False
                                        break
                                if ok:
                                    new_cands = [u for u in cands[i+1:] if not (adj[v] & (1 << u))]
                                    res = bt(new_cands, cur + [v])
                                    if res:
                                        return res
                            return None
                        
                        return bt(candidates, [])
                    
                    ind6_set = find_ind6_fast(current_adj, current_n)
                    if ind6_set:
                        violation = ind6_set
                        v_type = 'Ind6'
                
                if violation is None:
                    # Valid!
                    success = True
                    break
                
                # Flip an edge
                if v_type == 'K4':
                    u, v, w, x = violation
                    edges = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
                    a, b = random.choice(edges)
                else:
                    # Ind6: add an edge to break independence
                    a, b = random.sample(violation, 2)
                
                # Flip
                current_adj[a] ^= (1 << b)
                current_adj[b] ^= (1 << a)
                
                # Optional: Check if we made it worse? 
                # Hard to check quickly. Just proceed.
                # To avoid oscillation, we could track recent moves, but for now, random walk is okay.
                
            if success:
                best_adj = current_adj
                best_n = current_n
                adj = current_adj
                n = current_n
                break
        
        if not success:
            # If we couldn't extend, try to recover or just stop
            # We can try to search harder for this n, but if we failed 20 attempts, maybe n is too hard.
            # Let's try one more time with a longer time limit if we have time left
            if time.time() < deadline - 10:
                 # Retry with more attempts
                 pass # For simplicity, we just break and return best_n
            else:
                break

    # Final validation
    if is_valid(best_adj, best_n):
        return {'n': best_n, 'edges': get_edges(best_adj, best_n)}
    else:
        # Fallback to n=17
        return {'n': 17, 'edges': get_edges(adj, 17)}

