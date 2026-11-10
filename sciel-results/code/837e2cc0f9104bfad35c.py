"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 115  # safety margin before 150s kill
    n = 40
    
    # Build adjacency as bitmasks for fast operations
    adj_mask = [0] * n
    
    def add_edge(u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    def remove_edge(u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
    
    def neighbors_mask(u):
        return adj_mask[u]
    
    def degree(u):
        return bin(adj_mask[u]).count('1')
    
    def total_edges():
        return sum(degree(u) for u in range(n)) // 2
    
    def has_c6_fast(u, v):
        """Check if adding edge (u,v) creates a C6 using optimized bitmasks."""
        nu = adj_mask[u] & ~(1 << v)
        nv = adj_mask[v] & ~(1 << u)
        uv_mask = (1 << u) | (1 << v)
        
        a = nu
        while a:
            la = a & (-a)
            a ^= la
            ai = la.bit_length() - 1
            ai_mask = 1 << ai
            
            # Neighbors of ai, excluding u, v, and ai itself
            nb = adj_mask[ai] & ~uv_mask & ~ai_mask
            b = nb
            while b:
                lb = b & (-b)
                b ^= lb
                bi = lb.bit_length() - 1
                bi_mask = 1 << bi
                
                # Neighbors of bi, excluding u, v, ai, bi
                nc = adj_mask[bi] & ~uv_mask & ~ai_mask & ~bi_mask
                c = nc
                while c:
                    lc = c & (-c)
                    c ^= lc
                    ci = lc.bit_length() - 1
                    
                    # Check if any neighbor of ci is in nv, excluding u, v, ai, bi, ci
                    ci_mask = 1 << ci
                    nd = adj_mask[ci] & nv & ~uv_mask & ~ai_mask & ~bi_mask & ~ci_mask
                    if nd:
                        return True
        
        return False
    
    def spectral_gap_score(u, v):
        """Compute a heuristic score for adding edge (u,v).
        
        Uses a smooth surrogate that balances:
        1. Degree balance (lower degree = more room)
        2. Common neighbor penalty (clustering risk for C6)
        3. Cluster diversity bonus
        """
        deg_u = degree(u)
        deg_v = degree(v)
        
        # Common neighbors
        common = adj_mask[u] & adj_mask[v]
        common_count = bin(common).count('1')
        
        # Smooth score: use log-sum-exp style soft penalty for common neighbors
        # Higher common_count means higher C6 risk, apply exponential penalty
        soft_penalty = np.logaddexp(0, common_count * 0.5)
        
        base_score = (n - deg_u - deg_v) * 2.0 - soft_penalty * 4.0
        
        # Bonus for connecting different K5 blocks
        block_u = u // 5
        block_v = v // 5
        if block_u != block_v:
            base_score += 5.0
        
        return base_score
    
    def compute_spectral_gap_adj_matrix():
        """Compute the actual spectral gap from the current adjacency matrix."""
        A = np.zeros((n, n))
        for u in range(n):
            for v in range(u + 1, n):
                if (1 << v) & adj_mask[u]:
                    A[u, v] = 1.0
                    A[v, u] = 1.0
        eigvals = np.linalg.eigvalsh(A)
        eigvals = eigvals[::-1]  # sort descending
        if len(eigvals) < 2:
            return 0.0
        return eigvals[0] - eigvals[1]
    
    # Start with 8 disjoint K5s
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                add_edge(5 * k + a, 5 * k + b)
    
    # Track best state
    best_adj_mask = list(adj_mask)
    best_count = total_edges()
    
    # All possible pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Spectral-guided greedy addition with multiple restarts
    max_restarts = 5
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        # Reset to best state for each restart (except first)
        if restart > 0:
            adj_mask = list(best_adj_mask)
        
        for iteration in range(200):
            if time.time() > deadline - 5:
                break
            
            # Get all missing edges
            missing_edges = [(u, v) for (u, v) in all_pairs if not ((1 << v) & adj_mask[u])]
            
            if not missing_edges:
                break
            
            # Score and sort
            scored_edges = []
            for (u, v) in missing_edges:
                score = spectral_gap_score(u, v)
                scored_edges.append((score, u, v))
            
            scored_edges.sort(key=lambda x: -x[0])
            
            added_any = False
            for (score, u, v) in scored_edges:
                if time.time() > deadline - 3:
                    break
                if (1 << v) & adj_mask[u]:
                    continue
                if not has_c6_fast(u, v):
                    add_edge(u, v)
                    added_any = True
            
            current_count = total_edges()
            if current_count > best_count:
                best_count = current_count
                best_adj_mask = list(adj_mask)
            
            if not added_any:
                # Perturbation: remove 1-2 random edges
                if time.time() > deadline - 3:
                    break
                all_current_edges = [(u, v) for u in range(n) for v in range(u+1, n) if (1 << v) & adj_mask[u]]
                if all_current_edges:
                    num_removals = random.randint(1, 2)
                    for _ in range(num_removals):
                        ru, rv = random.choice(all_current_edges)
                        remove_edge(ru, rv)
                        all_current_edges = [(u, v) for u in range(n) for v in range(u+1, n) if (1 << v) & adj_mask[u]]
                    
                    # Re-score and try to add edges
                    missing_edges = [(u, v) for (u, v) in all_pairs if not ((1 << v) & adj_mask[u])]
                    scored_edges = []
                    for (u, v) in missing_edges:
                        score = spectral_gap_score(u, v)
                        scored_edges.append((score, u, v))
                    scored_edges.sort(key=lambda x: -x[0])
                    
                    for (score, u, v) in scored_edges:
                        if time.time() > deadline - 3:
                            break
                        if (1 << v) & adj_mask[u]:
                            continue
                        if not has_c6_fast(u, v):
                            add_edge(u, v)
                    
                    current_count = total_edges()
                    if current_count > best_count:
                        best_count = current_count
                        best_adj_mask = list(adj_mask)
                else:
                    break
    
    # Final polish: multiple passes with different orderings
    for polish_pass in range(3):
        if time.time() > deadline - 2:
            break
        
        # Load best state
        adj_mask = list(best_adj_mask)
        
        # Try edges in shuffled order
        shuffled_pairs = list(all_pairs)
        random.shuffle(shuffled_pairs)
        
        for (u, v) in shuffled_pairs:
            if time.time() > deadline - 1:
                break
            if (1 << v) & adj_mask[u]:
                continue
            if not has_c6_fast(u, v):
                add_edge(u, v)
        
        current_count = total_edges()
        if current_count > best_count:
            best_count = current_count
            best_adj_mask = list(adj_mask)
    
    # Build final edge list from the best state
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (1 << v) & best_adj_mask[u]:
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
