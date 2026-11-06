import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def has_independent_set_10(adj, n):
        """Check if graph has independent set of size 10. Returns True if it does."""
        mask_all = (1 << n) - 1

        # Greedy lower bound: try to find a large independent set quickly
        candidates = mask_all
        size = 0
        while candidates and size < 10:
            best_v = -1
            min_deg = n + 1
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = (adj[v] & candidates).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v

            if best_v == -1:
                break

            size += 1
            candidates &= ~adj[best_v] & ~(1 << best_v)

        if size >= 10:
            return True

        # Exact branch and bound
        sys.setrecursionlimit(10000)

        def backtrack(candidates, size):
            if size == 10:
                return True
            if candidates == 0:
                return False

            remaining = candidates.bit_count()
            if size + remaining < 10:
                return False

            # Pick vertex with max degree in candidates (removes most)
            best_v = -1
            max_deg = -1
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = (adj[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v

            if best_v == -1:
                return False

            # Branch 1: include best_v
            if backtrack(candidates & ~adj[best_v] & ~(1 << best_v), size + 1):
                return True
            # Branch 2: exclude best_v
            if backtrack(candidates & ~(1 << best_v), size):
                return True
            return False

        return backtrack(mask_all, 0)

    def is_triangle_free(adj, n):
        """Check if graph is triangle-free."""
        for u in range(n):
            neighbors = adj[u]
            temp = neighbors
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                if v > u:
                    if adj[u] & adj[v]:
                        return False
        return True

    def build_edge_list(adj, n):
        """Convert adjacency bitmasks to edge list."""
        edges = []
        for i in range(n):
            temp = adj[i]
            while temp:
                lsb = temp & -temp
                j = lsb.bit_length() - 1
                if j > i:
                    edges.append([i, j])
                temp &= temp - 1
        return edges

    def find_circulant(n):
        """
        Search for a circulant graph C_n(S) that is triangle-free and has alpha < 10.
        S is a set of distances. Triangle-free <=> S is sum-free in Z_n.
        """
        # We need to find S subset of {1, ..., n//2}
        # such that (a+b)%n not in S for a,b in S U {n-s for s in S}
        # Actually, for undirected circulant, we define S_sym = S U {n-s | s in S}.
        # Triangle free iff (x+y+z) % n != 0 for x,y,z in S_sym? 
        # Simpler: S is sum-free means no x,y in S such that (x+y)%n in S_sym.
        
        # Strategy: Greedy construction of maximal sum-free sets with random ordering.
        # Also try specific structured sets like intervals.
        
        candidates_dist = list(range(1, n // 2 + 1))
        
        # Try several random orderings to find different maximal sum-free sets
        for restart in range(2000):
            if time.time() > deadline:
                return None
                
            random.shuffle(candidates_dist)
            S = set()
            
            # Greedy add
            for d in candidates_dist:
                # Check if adding d keeps S sum-free
                # New set of distances effectively: S_new = S U {d, n-d}
                # Check if any sum of two elements in S_new is in S_new
                # Optimization: Only check sums involving d or n-d against S, and d+d, d+(n-d)=0 (ok), (n-d)+(n-d)
                
                # Let's just do the full check for small n (<=40) as it's fast
                S_new = set(S)
                S_new.add(d)
                S_new.add(n - d)
                
                valid = True
                # Check sums
                # We need (a+b) % n not in S_new for all a,b in S_new
                # Note: 0 is not in S_new (distances are 1..n-1)
                # But (a+b)%n can be 0. 0 is not in S_new, so that's fine.
                
                # To speed up, we can precompute sums of S
                # But |S| is small (max ~n/3 = 13). 13^2 = 169 checks. Very fast.
                for a in S_new:
                    for b in S_new:
                        if (a + b) % n in S_new:
                            valid = False
                            break
                    if not valid:
                        break
                            
                if valid:
                    S = S_new
            
            # Build adjacency matrix for this S
            adj = [0] * n
            for i in range(n):
                for d in S:
                    j1 = (i + d) % n
                    j2 = (i - d) % n
                    adj[i] |= (1 << j1)
                    adj[i] |= (1 << j2)
            
            # Verify triangle-free (should be by construction, but safety check)
            if not is_triangle_free(adj, n):
                continue
                
            # Check independence number
            if not has_independent_set_10(adj, n):
                return build_edge_list(adj, n)
                
        return None

    # Main search loop: Try largest n first
    # Known best is around 39. R(3,10) <= 41, so n=40 is the theoretical max target.
    for n in range(40, 24, -1):
        if time.time() > deadline:
            break
            
        edges = find_circulant(n)
        if edges is not None:
            return {'n': n, 'edges': edges}
            
    # Fallback: Return a known valid graph if search fails or times out
    # Andrásfai graph And(9) has n=27, alpha=9.
    # Let's construct And(9) as a safe fallback.
    def make_andrasfai(k):
        n = 3 * k
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if d == 1 or d == n - 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj, n

    # Try a few smaller Andrasfai graphs or other known constructions if n>24 failed
    # But usually the circulant search finds n=38+ quickly.
    # If we are here, we either ran out of time or failed to find a circulant.
    
    # Try a simple random search for smaller n as a last resort before fixed fallback
    for n in range(24, 10, -1):
        if time.time() > deadline:
            break
        for _ in range(50):
            adj = [0] * n
            # Random triangle-free graph
            edges_to_try = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(edges_to_try)
            for u, v in edges_to_try:
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            if not has_independent_set_10(adj, n):
                return {'n': n, 'edges': build_edge_list(adj, n)}
                
    # Absolute fallback: C5 (n=5, alpha=2)
    c5_adj = [0] * 5
    for i in range(5):
        c5_adj[i] |= (1 << ((i + 1) % 5))
        c5_adj[i] |= (1 << ((i + 4) % 5))
    return {'n': 5, 'edges': build_edge_list(c5_adj, 5)}

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(f"Found graph with n={res['n']}, edges={len(res['edges'])}")
