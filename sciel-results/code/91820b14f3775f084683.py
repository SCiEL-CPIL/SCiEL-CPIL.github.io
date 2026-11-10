import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    n = 20
    vertices = list(range(n))
    
    # Precompute all triples (u, v, w) for K3,3 check
    triples = list(combinations(vertices, 3))
    num_triples = len(triples)
    
    # Precompute for each vertex, which triples contain it
    triples_containing = [[] for _ in range(n)]
    for idx, (u, v, w) in enumerate(triples):
        triples_containing[u].append(idx)
        triples_containing[v].append(idx)
        triples_containing[w].append(idx)
    
    def bit(i):
        return 1 << i
    
    def popcount(x):
        return x.bit_count()
    
    def is_k33_free(adj_mask):
        """Check if graph is K3,3-free."""
        for u, v, w in triples:
            if popcount(adj_mask[u] & adj_mask[v] & adj_mask[w]) > 2:
                return False
        return True
    
    def edge_count(adj_mask):
        return sum(m.bit_count() for m in adj_mask) // 2
    
    def mask_to_edges(adj_mask):
        edges = []
        for i in range(n):
            m = adj_mask[i]
            for j in range(i + 1, n):
                if m & bit(j):
                    edges.append([i, j])
        return edges
    
    def can_add(u, v, adj_mask):
        """
        Check if adding edge (u, v) keeps the graph K3,3-free.
        Only checks triples containing u or v.
        """
        bit_v = bit(v)
        bit_u = bit(u)
        new_u = adj_mask[u] | bit_v
        new_v = adj_mask[v] | bit_u
        
        # Check triples containing u
        for tidx in triples_containing[u]:
            a, b, c = triples[tidx]
            ma = new_u if a == u else (new_v if a == v else adj_mask[a])
            mb = new_u if b == u else (new_v if b == v else adj_mask[b])
            mc = new_u if c == u else (new_v if c == v else adj_mask[c])
            if popcount(ma & mb & mc) > 2:
                return False
        
        # Check triples containing v but not u
        for tidx in triples_containing[v]:
            a, b, c = triples[tidx]
            if a == u or b == u or c == u:
                continue  # Already checked
            ma = new_v if a == v else adj_mask[a]
            mb = new_v if b == v else adj_mask[b]
            mc = new_v if c == v else adj_mask[c]
            if popcount(ma & mb & mc) > 2:
                return False
        
        return True
    
    def greedy_add(adj_mask):
        """Add all possible edges greedily."""
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj_mask[i] & bit(j)):
                        if can_add(i, j, adj_mask):
                            adj_mask[i] |= bit(j)
                            adj_mask[j] |= bit(i)
                            changed = True
        return adj_mask
    
    def find_violation(adj_mask):
        """Find a K3,3 violation, return (u,v,w,common_neighbors) or None."""
        for u, v, w in triples:
            common = adj_mask[u] & adj_mask[v] & adj_mask[w]
            if popcount(common) > 2:
                return (u, v, w, common)
        return None
    
    def find_violations_fast(adj_mask):
        """Find all K3,3 violations quickly."""
        violations = []
        for u, v, w in triples:
            common = adj_mask[u] & adj_mask[v] & adj_mask[w]
            if popcount(common) > 2:
                violations.append((u, v, w, common))
        return violations
    
    def repair(adj_mask):
        """Remove edges to fix violations."""
        for _ in range(200):
            if time.time() > deadline:
                break
            viol = find_violation(adj_mask)
            if viol is None:
                break
            u, v, w, common = viol
            common_list = [x for x in range(n) if common & bit(x)]
            random.shuffle(common_list)
            removed = False
            for c in common_list:
                for a, b in [(u, c), (v, c), (w, c)]:
                    if adj_mask[a] & bit(b):
                        adj_mask[a] &= ~bit(b)
                        adj_mask[b] &= ~bit(a)
                        if find_violation(adj_mask) is None:
                            removed = True
                            break
                        else:
                            adj_mask[a] |= bit(b)
                            adj_mask[b] |= bit(a)
                if removed:
                    break
            if not removed:
                if common_list:
                    c = common_list[0]
                    adj_mask[u] &= ~bit(c)
                    adj_mask[c] &= ~bit(u)
        return adj_mask
    
    def repair_aggressive(adj_mask):
        """Remove edges more aggressively to fix violations."""
        for _ in range(300):
            if time.time() > deadline:
                break
            violations = find_violations_fast(adj_mask)
            if not violations:
                break
            
            # Group violations by common neighbors to remove efficiently
            removal_candidates = {}
            for u, v, w, common in violations:
                common_list = [x for x in range(n) if common & bit(x)]
                for c in common_list:
                    key = (min(u, v, w), max(u, v, w), c)
                    removal_candidates[key] = removal_candidates.get(key, 0) + 1
            
            # Remove the most impactful edge
            if removal_candidates:
                best_key = max(removal_candidates, key=removal_candidates.get)
                a, b, c = best_key
                if adj_mask[a] & bit(c):
                    adj_mask[a] &= ~bit(c)
                    adj_mask[c] &= ~bit(a)
                elif adj_mask[b] & bit(c):
                    adj_mask[b] &= ~bit(c)
                    adj_mask[c] &= ~bit(b)
            
            # Continue with standard repair
            adj_mask = repair(adj_mask)
        
        return adj_mask
    
    def try_swap_2_for_3(adj_mask, rng):
        """
        Try to remove 2 edges and add 3 different edges (net gain of 1).
        This helps escape local optima.
        Returns (new_adj_mask, success)
        """
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n) 
                    if adj_mask[i] & bit(j)]
        missing = [(i, j) for i in range(n) for j in range(i + 1, n) 
                  if not (adj_mask[i] & bit(j))]
        
        if len(all_edges) < 2 or len(missing) < 3:
            return adj_mask, False
        
        # Try several random pairs of edges to remove
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Pick 2 edges to remove
            e1_idx, e2_idx = rng.sample(range(len(all_edges)), 2)
            e1 = all_edges[e1_idx]
            e2 = all_edges[e2_idx]
            
            # Remove them temporarily
            adj2 = list(adj_mask)
            adj2[e1[0]] &= ~bit(e1[1])
            adj2[e1[1]] &= ~bit(e1[0])
            adj2[e2[0]] &= ~bit(e2[1])
            adj2[e2[1]] &= ~bit(e2[0])
            
            # Try to add up to 3 edges
            added = 0
            for u, v in rng.sample(missing, min(3, len(missing))):
                if not (adj2[u] & bit(v)):
                    if can_add(u, v, adj2):
                        adj2[u] |= bit(v)
                        adj2[v] |= bit(u)
                        added += 1
            
            if added >= 3:
                # Net gain of at least 1
                adj2 = greedy_add(adj2)
                return adj2, True
        
        return adj_mask, False
    
    def local_search(adj_mask, max_iters=100, rng=None):
        """Fast local search with edge additions, random removals, and 2-for-3 swaps."""
        if rng is None:
            rng = random.Random()
        
        adj_mask = repair(adj_mask)
        adj_mask = greedy_add(adj_mask)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try to add missing edges
            missing = [(i, j) for i in range(n) for j in range(i + 1, n) 
                      if not (adj_mask[i] & bit(j))]
            
            if not missing:
                break
            
            rng.shuffle(missing)
            improved = False
            
            for u, v in missing:
                if can_add(u, v, adj_mask):
                    adj_mask[u] |= bit(v)
                    adj_mask[v] |= bit(u)
                    improved = True
                    break
            
            if improved:
                adj_mask = greedy_add(adj_mask)
                continue
            
            # If no edge can be added, try 2-for-3 swap first
            new_adj, success = try_swap_2_for_3(adj_mask, rng)
            if success:
                adj_mask = new_adj
                continue
            
            # If swap failed, remove a random edge to escape local optimum
            all_edges = [(i, j) for i in range(n) for j in range(i + 1, n) 
                        if adj_mask[i] & bit(j)]
            
            if not all_edges:
                break
            
            r_u, r_v = rng.choice(all_edges)
            adj_mask[r_u] &= ~bit(r_v)
            adj_mask[r_v] &= ~bit(r_u)
            adj_mask = greedy_add(adj_mask)
        
        return adj_mask
    
    # Use numpy for vectorized operations where possible
    best_adj = None
    best_edges = 0
    rng = random.Random(42)
    
    # Deterministic base construction using structured pattern
    # Start with a high-density base: connect vertices based on modular arithmetic
    # This creates a structured graph that's close to K3,3-free
    
    def construct_base():
        """Construct a high-density base graph using structured pattern."""
        adj = [0] * n
        # Connect vertices with difference 1, 2, 3, 4, 5 (mod 20)
        # This creates a circulant-like graph
        for d in range(1, 6):
            for i in range(n):
                j = (i + d) % n
                if i < j:
                    adj[i] |= bit(j)
                    adj[j] |= bit(i)
        return adj
    
    # Try deterministic base first
    base_adj = construct_base()
    base_adj = repair_aggressive(base_adj)
    base_adj = greedy_add(base_adj)
    base_cnt = edge_count(base_adj)
    
    if base_cnt > 0:
        best_adj = list(base_adj)
        best_edges = base_cnt
    
    # Try random restarts with different densities
    for restart in range(40):
        if time.time() > deadline - 5:
            break
        
        # Random initialization with varying density
        density = 0.3 + rng.random() * 0.4
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < density:
                    adj[i] |= bit(j)
                    adj[j] |= bit(i)
        
        # Repair and optimize
        adj = local_search(adj, max_iters=150, rng=rng)
        
        cnt = edge_count(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = list(adj)
        
        # Early termination if we find a very good solution
        if best_edges > 145:
            break
    
    # One more pass with higher iteration count if time allows
    if best_adj is not None and time.time() < deadline - 15:
        adj = list(best_adj)
        adj = local_search(adj, max_iters=300, rng=rng)
        cnt = edge_count(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = list(adj)
    
    # Final polish: try to add any remaining edges
    if best_adj is not None and time.time() < deadline - 2:
        adj = list(best_adj)
        adj = greedy_add(adj)
        cnt = edge_count(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = list(adj)
    
    if best_adj is None:
        # Fallback: empty graph
        best_adj = [0] * n
        best_edges = 0
    
    edges = mask_to_edges(best_adj)
    return edges
