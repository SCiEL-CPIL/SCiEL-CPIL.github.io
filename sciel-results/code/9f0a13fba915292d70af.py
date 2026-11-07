import time
import random
import sys

def construct():
    n = 15
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    def compute_degrees(adj):
        return [bin(adj[i]).count('1') for i in range(n)]

    def is_valid_adj(adj, degrees=None):
        if degrees is None:
            degrees = compute_degrees(adj)
        # For each edge (u, v), check all paths s-u-v-t where s != t
        for u in range(n):
            au = adj[u]
            for v in range(n):
                if u >= v:
                    continue
                if not (au & (1 << v)):
                    continue
                au_minus_v = au & ~(1 << v)
                av_minus_u = adj[v] & ~(1 << u)
                
                s_mask = au_minus_v
                while s_mask:
                    s_bit = s_mask & -s_mask
                    s = s_bit.bit_length() - 1
                    s_mask ^= s_bit
                    
                    t_mask = av_minus_u & ~(1 << s)
                    while t_mask:
                        t_bit = t_mask & -t_mask
                        t = t_bit.bit_length() - 1
                        t_mask ^= t_bit
                        
                        if degrees[s] == degrees[t]:
                            return False
        return True

    def adj_to_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    def count_edges(adj):
        total = 0
        for i in range(n):
            total += bin(adj[i]).count('1')
        return total // 2

    # Strategy: Constructive approach based on degree classes
    # Key insight: For any edge (u,v), all neighbors of u (except v) and all neighbors of v (except u)
    # must have pairwise distinct degrees.
    # 
    # We'll try to build a graph where vertices are partitioned into degree classes,
    # and we construct edges to maximize density while maintaining the constraint.
    #
    # Approach: 
    # 1. Try different degree assignments (partitions of 15 vertices into degree classes)
    # 2. For each assignment, greedily add edges that are consistent with the constraint
    # 3. Use local search to improve
    
    best_adj = [0] * n
    best_count = 0
    
    random.seed(12345)
    
    # Phase 1: Try structured degree assignments
    # A key structure: if we have degree classes D1, D2, ..., Dk, then for any edge (u,v),
    # all neighbors of u must be in different degree classes from each other (except possibly v's class)
    # and similarly for v.
    
    # Let's try: assign degrees as much as possible, then build edges
    # Start with a few known good structures and try to improve
    
    # Structure 1: Try to make most vertices have unique degrees
    # With 15 vertices, degrees can range from 0 to 14
    # If all degrees are distinct, the constraint is automatically satisfied
    # (no two endpoints can have equal degree)
    
    # Let's try to construct a graph with all distinct degrees
    # This is a "graceful" type of construction
    
    # For n=15, can we have degrees 0,1,2,...,14?
    # Sum would be 0+1+...+14 = 105, which must be even (handshaking lemma)
    # 105 is odd, so we can't have exactly degrees 0 through 14
    
    # Try degrees that sum to even: e.g., skip one degree
    # Or have two vertices share a degree but ensure they're never endpoints of a 3-path
    
    # Let's try a different approach: 
    # Build graph greedily, but with a smarter edge ordering
    
    # Phase 1: Multiple greedy passes with different edge orderings
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for trial in range(200):
        trial_start = time.time()
        if trial_start - start_time > deadline - 3:
            break
            
        trial_adj = [0] * n
        trial_deg = [0] * n
        trial_count = 0
        
        # Different ordering strategies
        if trial % 4 == 0:
            random.shuffle(all_pairs)
        elif trial % 4 == 1:
            # Sort by sum of indices (favor central edges first)
            all_pairs_sorted = sorted(all_pairs, key=lambda p: p[0] + p[1])
            random.shuffle(all_pairs_sorted)
            pairs = all_pairs_sorted
        elif trial % 4 == 2:
            # Favor edges between low-index and high-index vertices
            all_pairs_sorted = sorted(all_pairs, key=lambda p: abs(p[0] - p[1]), reverse=True)
            random.shuffle(all_pairs_sorted)
            pairs = all_pairs_sorted
        else:
            # Random with bias toward high-degree potential
            pairs = all_pairs[:]
            random.shuffle(pairs)
        
        for u, v in pairs:
            new_adj = trial_adj[:]
            new_deg = trial_deg[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            new_deg[u] += 1
            new_deg[v] += 1
            
            if is_valid_adj(new_adj, new_deg):
                trial_adj = new_adj
                trial_deg = new_deg
                trial_count += 1
        
        if trial_count > best_count:
            best_count = trial_count
            best_adj = trial_adj[:]
            
        if best_count >= 30:
            break
    
    # Phase 2: Intensive local search with edge additions
    current_adj = best_adj[:]
    current_deg = compute_degrees(current_adj)
    current_count = best_count
    
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        no_improve += 1
        
        # Try to add a random non-edge
        added = False
        for _ in range(50):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if current_adj[u] & (1 << v):
                continue
                
            new_adj = current_adj[:]
            new_deg = current_deg[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            new_deg[u] += 1
            new_deg[v] += 1
            
            if is_valid_adj(new_adj, new_deg):
                current_adj = new_adj
                current_deg = new_deg
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_adj = current_adj[:]
                added = True
                no_improve = 0
                break
        
        if added:
            continue
        
        # Try swap: remove one edge, add another
        swapped = False
        # Pick random edge to remove
        edge_to_remove = None
        for _ in range(50):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if current_adj[u] & (1 << v):
                edge_to_remove = (u, v)
                break
        
        if edge_to_remove is None:
            break
            
        u1, v1 = edge_to_remove
        
        # Pick random non-edge to add
        for _ in range(80):
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 == v2:
                continue
            if u2 > v2:
                u2, v2 = v2, u2
            if (u2, v2) == (u1, v1):
                continue
            if current_adj[u2] & (1 << v2):
                continue
                
            new_adj = current_adj[:]
            new_deg = current_deg[:]
            new_adj[u1] &= ~(1 << v1)
            new_adj[v1] &= ~(1 << u1)
            new_adj[u2] |= (1 << v2)
            new_adj[v2] |= (1 << u2)
            new_deg[u1] -= 1
            new_deg[v1] -= 1
            new_deg[u2] += 1
            new_deg[v2] += 1
            
            if is_valid_adj(new_adj, new_deg):
                current_adj = new_adj
                current_deg = new_deg
                # Count stays same
                swapped = True
                no_improve = 0
                break
        
        if not swapped:
            no_improve += 1
            
        if no_improve > max_no_improve:
            # Restart from best
            current_adj = best_adj[:]
            current_deg = compute_degrees(current_adj)
            current_count = best_count
            no_improve = 0
            max_no_improve = int(max_no_improve * 1.2)
            
            # Also try a fresh greedy from scratch occasionally
            if random.random() < 0.3:
                fresh_adj = [0] * n
                fresh_deg = [0] * n
                fresh_count = 0
                shuffled = all_pairs[:]
                random.shuffle(shuffled)
                for u, v in shuffled:
                    new_adj = fresh_adj[:]
                    new_deg = fresh_deg[:]
                    new_adj[u] |= (1 << v)
                    new_adj[v] |= (1 << u)
                    new_deg[u] += 1
                    new_deg[v] += 1
                    if is_valid_adj(new_adj, new_deg):
                        fresh_adj = new_adj
                        fresh_deg = new_deg
                        fresh_count += 1
                if fresh_count > best_count:
                    best_count = fresh_count
                    best_adj = fresh_adj[:]
                current_adj = best_adj[:]
                current_deg = compute_degrees(current_adj)
                current_count = best_count
    
    # Final: try systematic edge additions from best
    current_adj = best_adj[:]
    current_deg = compute_degrees(current_adj)
    current_count = best_count
    
    # Try all possible edge additions systematically
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for u in range(n):
            for v in range(u + 1, n):
                if current_adj[u] & (1 << v):
                    continue
                new_adj = current_adj[:]
                new_deg = current_deg[:]
                new_adj[u] |= (1 << v)
                new_adj[v] |= (1 << u)
                new_deg[u] += 1
                new_deg[v] += 1
                if is_valid_adj(new_adj, new_deg):
                    current_adj = new_adj
                    current_deg = new_deg
                    current_count += 1
                    best_count = current_count
                    best_adj = current_adj[:]
                    changed = True
    
    # Convert best_adj back to edge list
    final_edges = adj_to_edges(best_adj)
    
    # Ensure types are Python int
    final_edges = [[int(e[0]), int(e[1])] for e in final_edges]
    
    return final_edges
