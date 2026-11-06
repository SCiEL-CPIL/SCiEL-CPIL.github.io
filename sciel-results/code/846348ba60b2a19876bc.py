# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115
    n = 18

    def edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def has_k4_fast(adj):
        """Fast K4 detection using bitmask operations."""
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b] & ~((1 << (b + 1)) - 1)
                for c in range(b + 1, n):
                    if (common >> c) & 1:
                        if common & adj[c] & ~((1 << (c + 1)) - 1):
                            return True
        return False

    def find_all_triangles(adj):
        """Find all triangles in the graph and return them as a list of bitmasks."""
        triangles = []
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b] & ~((1 << (b + 1)) - 1)
                tmp = common
                while tmp:
                    bit = tmp & (-tmp)
                    c = bit.bit_length() - 1
                    if (adj[b] >> c) & 1: # Check if b-c is edge (implied by common? No, common is adj[a] & adj[b])
                         # Wait, common = adj[a] & adj[b] & mask > b.
                         # So c is neighbor of a and b.
                         # So a-b-c is a triangle.
                         mask = (1 << a) | (1 << b) | (1 << c)
                         triangles.append(mask)
                    tmp &= tmp - 1
        return triangles

    def compute_triangle_packing_bound(adj):
        """
        Computes a lower bound on the minimum triangle vertex cover size.
        This is the size of a maximum set of pairwise disjoint triangles.
        We use a greedy approximation for speed.
        Returns the size M. The score (largest triangle-free subgraph) is <= n - M.
        """
        triangles = find_all_triangles(adj)
        if not triangles:
            return 0
        
        # Greedy set packing
        # Sort triangles by "popularity" (number of intersecting triangles) to pick "central" ones first?
        # Or just random?
        # Let's try picking the one with the fewest intersections first to preserve options?
        # Actually, for a lower bound on cover, we want a large packing.
        # Standard greedy: pick any, remove conflicting.
        
        remaining = set(triangles)
        count = 0
        # To make it deterministic and fast, we can just iterate
        # A better heuristic: pick triangle with min degree in conflict graph?
        # For simplicity and speed, let's just pick the first one, or random.
        # Since we call this often in local search, it must be very fast.
        # n=18, max triangles ~ 800.
        
        # Convert to list for indexing
        tri_list = list(remaining)
        
        # Build conflict matrix? Too slow for repeated calls.
        # Just use bitmask intersection.
        
        # Simple greedy:
        # 1. Pick a triangle.
        # 2. Remove it and all that intersect it.
        
        # To improve the bound, we can do a few restarts or a simple local search on the packing?
        # Let's stick to a single pass for speed, maybe with randomization if called multiple times?
        # But local search calls it once per move.
        
        # Let's try sorting by vertex sum to be deterministic?
        # tri_list.sort(key=lambda m: bin(m).count('1')) # All are 3
        
        # Random shuffle for diversification in local search?
        # No, keep it deterministic for consistency in "best" tracking, or use global random state.
        # Let's use a simple deterministic heuristic: pick the triangle with the smallest min-vertex-id?
        
        selected = []
        available = set(triangles)
        
        while available:
            # Pick the first one in sorted order?
            # Let's pick the one with the smallest bit representation for determinism
            t = min(available)
            selected.append(t)
            # Remove t and all intersecting
            new_available = set()
            for u in available:
                if u == t:
                    continue
                if (u & t) == 0:
                    new_available.add(u)
            available = new_available
            
        return len(selected)

    def max_triangle_free_size(adj, time_limit=3.0, upper_bound=None):
        """Exact branch and bound for maximum triangle-free induced subgraph size."""
        best = [0]
        deg = [bin(adj[v]).count('1') for v in range(n)]
        # Order by degree descending, then by vertex index for determinism
        order = sorted(range(n), key=lambda v: (-deg[v], v))
        t0 = time.time()
        timeout = [False]
        
        if upper_bound is not None and upper_bound < best[0]: # best is 0 initially
             pass # upper_bound is an upper limit on the possible score.
             # We can use this to prune if current_size + remaining <= best[0]
             # But we don't have a lower bound on the score yet.
             # However, if we find a solution of size S, we know best >= S.
             # The upper_bound tells us we don't need to search for solutions > upper_bound.
             # This doesn't help pruning "include" branches directly unless we have a good lower bound.
             # But it helps if we use it as a target? No.
             
        def has_triangle_with_v(v, mask):
            """Check if adding v to mask creates a triangle."""
            neighbors_in_mask = mask & adj[v]
            tmp = neighbors_in_mask
            while tmp:
                bit = tmp & (-tmp)
                u = bit.bit_length() - 1
                rest = tmp & ~bit
                if (adj[u] & rest) != 0:
                    return True
                tmp = rest
            return False

        def backtrack(idx, current_mask, current_size):
            if timeout[0]:
                return
            if time.time() - t0 > time_limit:
                timeout[0] = True
                return
            if idx == n:
                if current_size > best[0]:
                    best[0] = current_size
                return
            # Pruning: even if we include all remaining vertices, can't beat best
            if current_size + (n - idx) <= best[0]:
                return
            
            # Additional pruning using upper_bound?
            # If current_size + (n - idx) <= upper_bound, we are within the bound.
            # This doesn't prune.
            
            v = order[idx]

            # Try including v first (finds large solutions faster for pruning)
            if not has_triangle_with_v(v, current_mask):
                backtrack(idx + 1, current_mask | (1 << v), current_size + 1)
                if timeout[0]:
                    return

            # Try excluding v
            # Additional pruning: if excluding v means we can't beat best
            if current_size + (n - idx - 1) > best[0]:
                backtrack(idx + 1, current_mask, current_size)

        backtrack(0, 0, 0)
        return best[0]

    def make_paley17_plus_iso():
        """Paley(17) graph on vertices 0..16, vertex 17 isolated."""
        adj = [0] * n
        # Paley graph: edge (i,j) if i-j is a quadratic residue mod 17
        # Quadratic residues mod 17: 1^2=1, 2^2=4, 3^2=9, 4^2=16, 5^2=8, 6^2=2, 7^2=15, 8^2=13
        residues = {1, 2, 4, 8, 9, 13, 15, 16}
        for i in range(17):
            for j in range(i + 1, 17):
                diff = (j - i) % 17
                if diff in residues:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        # Vertex 17 is isolated
        return adj

    def make_complete_tripartite():
        """Complete tripartite graph K(6,6,6) - known to be K4-free."""
        adj = [0] * n
        # Partition vertices into 3 groups of 6
        # Group 0: 0-5, Group 1: 6-11, Group 2: 12-17
        for i in range(n):
            for j in range(i + 1, n):
                gi = i // 6
                gj = j // 6
                if gi != gj:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def make_paley17_shuffled():
        """Paley(17) graph with a random permutation of vertices 0..16, vertex 17 isolated."""
        adj_base = make_paley17_plus_iso()
        perm = list(range(17))
        random.shuffle(perm)
        new_adj = [0] * n
        for i in range(17):
            for j in range(17):
                if i == j:
                    continue
                if (adj_base[i] >> j) & 1:
                    new_adj[perm[i]] |= (1 << perm[j])
                    new_adj[perm[j]] |= (1 << perm[i])
        return new_adj

    def remove_k4(adj):
        """Remove edges to make graph K4-free."""
        for _ in range(500):
            found = False
            for a in range(n):
                for b in range(a + 1, n):
                    if not ((adj[a] >> b) & 1):
                        continue
                    common = adj[a] & adj[b] & ~((1 << (b + 1)) - 1)
                    for c in range(b + 1, n):
                        if (common >> c) & 1:
                            remaining = common & adj[c] & ~((1 << (c + 1)) - 1)
                            if remaining:
                                d_bit = remaining & (-remaining)
                                d = d_bit.bit_length() - 1
                                pair = random.choice([(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)])
                                x, y = pair
                                adj[x] &= ~(1 << y)
                                adj[y] &= ~(1 << x)
                                found = True
                                break
                    if found:
                        break
                if found:
                    break
            if not found:
                break
        return not has_k4_fast(adj)

    random.seed(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # Primary: Complete tripartite K(6,6,6) - strong baseline
    candidates.append(make_complete_tripartite())
    
    # Paley(17) + isolated vertex
    candidates.append(make_paley17_plus_iso())
    
    # Shuffled Paley(17) variants
    for _ in range(15):
        if time.time() > deadline - 50:
            break
        candidates.append(make_paley17_shuffled())
        
    # Random dense graphs as backup
    for _ in range(20):
        if time.time() > deadline - 50:
            break
        density = 0.4 + random.random() * 0.4
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if remove_k4(adj):
            candidates.append(adj)

    # Evaluate candidates with time limit to find best starting point
    # First, compute the packing bound for all candidates to sort them
    scored_candidates = []
    for adj in candidates:
        if time.time() > deadline - 40:
            break
        m = compute_triangle_packing_bound(adj)
        ub = n - m
        scored_candidates.append((ub, adj))
        
    # Sort by upper bound (ascending) - lower UB is better
    scored_candidates.sort(key=lambda x: x[0])
    
    best_adj = None
    best_score = 18
    for ub, adj in scored_candidates:
        if time.time() > deadline - 30:
            break
        # If the upper bound is already worse than our best, we might still check if it's tight?
        # But if ub >= best_score, the true score is <= ub. It could be < best_score.
        # So we must check.
        # However, if we have many candidates, we can skip those with very high UB if we have a good best.
        # For now, evaluate all top ones.
        
        score = max_triangle_free_size(adj, time_limit=2.0)
        if score < best_score:
            best_score = score
            best_adj = adj[:]

    if best_adj is None:
        best_adj = candidates[0]
        best_score = max_triangle_free_size(best_adj, time_limit=1.5)

    # Structured local search: only accept moves that reduce the exact score
    adj = best_adj[:]
    current_score = best_score
    best_adj_sa = adj[:]
    best_score_sa = current_score
    
    no_improve_count = 0
    max_no_improve = 800
    
    while time.time() < deadline - 8 and no_improve_count < max_no_improve:
        # Pick a random edge to flip
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
            
        old_adj = adj[:]
        if (adj[i] >> j) & 1:
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)
        else:
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
        if has_k4_fast(adj):
            adj = old_adj
            no_improve_count += 1
            continue
            
        # Compute bound
        m = compute_triangle_packing_bound(adj)
        ub = n - m
        
        # Pruning: if ub >= current_score + 2, it's unlikely to be better.
        # If ub >= current_score, it *could* be better if true score < ub.
        # But if ub is much larger, skip.
        # Let's use a tighter check: if ub >= current_score, we still need to verify.
        # But if ub > current_score + 1, skip?
        # Let's just verify if ub < current_score + 2.
        
        if ub >= current_score + 2:
            adj = old_adj
            no_improve_count += 1
            continue
            
        new_score = max_triangle_free_size(adj, time_limit=0.7)
        
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score_sa:
                best_score_sa = current_score
                best_adj_sa = adj[:]
            no_improve_count = 0
        else:
            adj = old_adj
            no_improve_count += 1
            
        # If we've been stuck for a while, try a more aggressive perturbation
        if no_improve_count > 300 and random.random() < 0.15:
            # Flip 2-4 random edges
            for _ in range(random.randint(2, 4)):
                i2 = random.randint(0, n - 1)
                j2 = random.randint(0, n - 1)
                if i2 == j2:
                    continue
                if i2 > j2:
                    i2, j2 = j2, i2
                if (adj[i2] >> j2) & 1:
                    adj[i2] &= ~(1 << j2)
                    adj[j2] &= ~(1 << i2)
                else:
                    adj[i2] |= (1 << j2)
                    adj[j2] |= (1 << i2)
            if has_k4_fast(adj):
                adj = best_adj_sa[:]
                current_score = best_score_sa
            else:
                m = compute_triangle_packing_bound(adj)
                ub = n - m
                if ub >= current_score + 2:
                    adj = best_adj_sa[:]
                    current_score = best_score_sa
                    no_improve_count += 1
                    continue
                    
                new_score = max_triangle_free_size(adj, time_limit=0.7)
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score_sa:
                        best_score_sa = current_score
                        best_adj_sa = adj[:]
                    no_improve_count = 0
                else:
                    adj = best_adj_sa[:]
                    current_score = best_score_sa
                    no_improve_count += 1

    # Final polish: try all single-edge flips from best solution
    adj = best_adj_sa[:]
    for i in range(n):
        if time.time() > deadline - 3:
            break
        for j in range(i + 1, n):
            if time.time() > deadline - 3:
                break
            old_adj = adj[:]
            if (adj[i] >> j) & 1:
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
            else:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
            if has_k4_fast(adj):
                adj = old_adj
                continue
            
            m = compute_triangle_packing_bound(adj)
            ub = n - m
            if ub >= best_score_sa + 2:
                adj = old_adj
                continue
                
            score = max_triangle_free_size(adj, time_limit=0.5)
            if score < best_score_sa:
                best_score_sa = score
                best_adj_sa = adj[:]
            else:
                adj = old_adj
    
    # Second polish pass if time allows
    if time.time() < deadline - 5:
        adj = best_adj_sa[:]
        for i in range(n):
            if time.time() > deadline - 3:
                break
            for j in range(i + 1, n):
                if time.time() > deadline - 3:
                    break
                old_adj = adj[:]
                if (adj[i] >> j) & 1:
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                else:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                if has_k4_fast(adj):
                    adj = old_adj
                    continue
                    
                m = compute_triangle_packing_bound(adj)
                ub = n - m
                if ub >= best_score_sa + 2:
                    adj = old_adj
                    continue
                    
                score = max_triangle_free_size(adj, time_limit=0.4)
                if score < best_score_sa:
                    best_score_sa = score
                    best_adj_sa = adj[:]
                else:
                    adj = old_adj

    return edges_from_adj(best_adj_sa)
# EVOLVE-BLOCK-END
