import random
import time
import math

def construct():
    n = 20
    # All possible edges as (u, v) with u < v
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_index = {e: i for i, e in enumerate(all_edges)}
    
    # Bitmask for edges
    # We'll use a list of booleans or an integer bitmask.
    # Python integers are arbitrary precision, so a bitmask of 190 bits is fine.
    
    def get_adj(mask):
        adj = [0] * n
        for i, (u, v) in enumerate(all_edges):
            if (mask >> i) & 1:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def evaluate(mask):
        adj = get_adj(mask)
        edges = mask.bit_count()
        violations = 0
        for u in range(n):
            for v in range(u+1, n):
                common = (adj[u] & adj[v]).bit_count()
                if common > 2:
                    violations += (common - 2)
        return edges, violations

    # Greedy initialization
    def greedy_init():
        adj = [0] * n
        mask = 0
        random.shuffle(all_edges)
        for u, v in all_edges:
            # Check if adding (u,v) is valid
            # Common neighbors of u and v must be <= 2
            # New common neighbors will be old common + (neighbors of u that are neighbors of v? No)
            # If we add (u,v), the common neighbors of any pair (x,y) change only if x=u or y=u or x=v or y=v?
            # Actually, adding (u,v) increases the common neighbor count for:
            # 1. The pair (u,v) itself? No, u and v are not neighbors of each other in the "common" sense unless we consider the edge.
            # The condition is: For any pair {a,b}, |N(a) \cap N(b)| <= 2.
            # Adding (u,v) adds v to N(u) and u to N(v).
            # This affects:
            # - Pair (u, x): N(u) gains v. So |N(u) \cap N(x)| increases by 1 if v in N(x).
            # - Pair (v, x): N(v) gains u. So |N(v) \cap N(x)| increases by 1 if u in N(x).
            # - Pair (u, v): N(u) and N(v) don't intersect via the new edge directly in the "common neighbor" definition?
            #   N(u) \cap N(v) are vertices w such that w~u and w~v.
            #   Adding (u,v) does not change N(u) \cap N(v) for w != u,v.
            #   But wait, does u become a common neighbor of v and something? No.
            #   So we just need to check:
            #   1. For all w, if adj[v] has w, then common(u,w) increases by 1.
            #   2. For all w, if adj[u] has w, then common(v,w) increases by 1.
            #   3. common(u,v) is unchanged? 
            #      N(u) \cap N(v) consists of w != u,v.
            #      Adding (u,v) doesn't add any w to N(u) \cap N(v).
            #      So we just need to ensure that for all w ~ v, common(u,w) + 1 <= 2 => common(u,w) <= 1.
            #      And for all w ~ u, common(v,w) + 1 <= 2 => common(v,w) <= 1.
            
            valid = True
            # Check neighbors of v
            # Iterate w where adj[v] has bit w
            v_neigh = adj[v]
            while v_neigh:
                lsb = v_neigh & -v_neigh
                w = lsb.bit_length() - 1
                v_neigh ^= lsb
                # Count common neighbors of u and w
                # common(u,w) = popcount(adj[u] & adj[w])
                if (adj[u] & adj[w]).bit_count() > 1:
                    valid = False
                    break
            if not valid:
                continue
                
            u_neigh = adj[u]
            while u_neigh:
                lsb = u_neigh & -u_neigh
                w = lsb.bit_length() - 1
                u_neigh ^= lsb
                if (adj[v] & adj[w]).bit_count() > 1:
                    valid = False
                    break
            
            if valid:
                idx = edge_index[(u,v)]
                mask |= (1 << idx)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        return mask

    start_time = time.time()
    time_limit = 115.0
    
    # Start with greedy
    best_mask = greedy_init()
    best_edges, best_violations = evaluate(best_mask)
    current_mask = best_mask
    current_edges, current_violations = evaluate(current_mask)
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    
    # Penalty weight: We want to strongly penalize violations.
    # If we have 1 violation, we lose 10 edges worth of score?
    # Max edges ~ 190. Violations can be high.
    # Let's use Score = Edges - 50 * Violations
    PENALTY = 50.0
    
    current_score = current_edges - PENALTY * current_violations
    best_score = current_score
    
    while time.time() - start_time < time_limit:
        # Generate move: flip a random bit
        idx = random.randint(0, len(all_edges) - 1)
        new_mask = current_mask ^ (1 << idx)
        
        # Evaluate new state
        new_edges, new_violations = evaluate(new_mask)
        new_score = new_edges - PENALTY * new_violations
        
        # Acceptance criterion
        if new_score > current_score:
            current_mask = new_mask
            current_edges = new_edges
            current_violations = new_violations
            current_score = new_score
            
            if new_violations == 0 and new_edges > best_edges:
                best_mask = new_mask
                best_edges = new_edges
                best_violations = 0
                best_score = best_edges
        else:
            delta = new_score - current_score
            # Avoid overflow in exp
            if delta < -100:
                prob = 0.0
            else:
                prob = math.exp(delta / T)
            
            if random.random() < prob:
                current_mask = new_mask
                current_edges = new_edges
                current_violations = new_violations
                current_score = new_score
        
        T *= cooling
        if T < 0.01:
            T = 0.01
            
    # Return best valid graph found
    # Ensure we return a valid graph. If best_violations > 0, we might need to clean it up?
    # But SA should find valid ones. 
    # If best_violations > 0, we can try a quick local search to remove edges causing violations.
    if best_violations > 0:
         # Fallback: use greedy again or just return best (invalid) -> Score 0?
         # The task says "invalid result ... scores 0".
         # So we MUST return a valid graph.
         # Let's run a quick cleanup: remove edges involved in violations until valid.
         mask = best_mask
         adj = get_adj(mask)
         # Find violations
         while True:
             viol_found = False
             for u in range(n):
                 for v in range(u+1, n):
                     common = (adj[u] & adj[v]).bit_count()
                     if common > 2:
                         viol_found = True
                         # Remove a random edge incident to u or v that is part of the common neighborhood?
                         # Or just remove a random edge from the graph to break the K2,3.
                         # Simple: remove edge (u,v) if it exists? No, u,v might not be connected.
                         # Remove an edge (u, w) where w is a common neighbor.
                         common_bits = adj[u] & adj[v]
                         # Pick a random bit from common_bits
                         lsb = common_bits & -common_bits
                         w = lsb.bit_length() - 1
                         # Remove edge (u, w)
                         if (adj[u] >> w) & 1:
                             idx_uw = edge_index[(min(u,w), max(u,w))]
                             mask &= ~(1 << idx_uw)
                             adj = get_adj(mask)
                         else:
                             # Remove (v, w)
                             idx_vw = edge_index[(min(v,w), max(v,w))]
                             mask &= ~(1 << idx_vw)
                             adj = get_adj(mask)
                         break
                 if viol_found:
                     break
             if not viol_found:
                 break
         best_mask = mask
         best_edges, best_violations = evaluate(best_mask)
         
    # Convert mask to list of edges
    res = []
    for i, (u, v) in enumerate(all_edges):
        if (best_mask >> i) & 1:
            res.append([u, v])
            
    return res
