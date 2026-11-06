import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # The task is to find the largest n such that K_n can be 5-colored with no monochromatic C4.
    # Known bound from Lazebnik-Woldar is n=26.
    # The seed program found n=11.
    # We will try to construct n=26 using a robust Simulated Annealing approach.
    # If that fails, we try n=25, 24, etc.

    def get_idx(i, j, n):
        return i * (2 * n - i - 1) // 2 + (j - i - 1)

    def is_valid(n, colors):
        # Check for monochromatic C4 using bitmasks
        N = [[0] * n for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[get_idx(i, j, n)]
                N[c][i] |= (1 << j)
                N[c][j] |= (1 << i)
        
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[get_idx(i, j, n)]
                common = N[c][i] & N[c][j]
                if bin(common).count('1') >= 2:
                    return False
        return True

    def count_violations(n, colors, N):
        # N is precomputed neighbor bitmasks
        violations = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[get_idx(i, j, n)]
                common = N[c][i] & N[c][j]
                if bin(common).count('1') >= 2:
                    violations += 1
        return violations

    def simulate_annealing(n, max_time=10.0):
        n_edges = n * (n - 1) // 2
        colors = [random.randint(0, 4) for _ in range(n_edges)]
        
        # Precompute edge endpoints
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Build initial neighbor bitmasks
        N = [[0] * n for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[get_idx(i, j, n)]
                N[c][i] |= (1 << j)
                N[c][j] |= (1 << i)
        
        violations = count_violations(n, colors, N)
        
        if violations == 0:
            return colors
        
        T = 1.0
        cooling = 0.999
        start_t = time.time()
        
        while T > 0.01 and time.time() - start_t < max_time:
            # Pick a random edge
            idx = random.randint(0, n_edges - 1)
            i, j = edges[idx]
            c = colors[idx]
            
            # Try a different color
            new_c = random.randint(0, 4)
            if new_c == c:
                continue
            
            # Calculate change in violations
            # Removing (i,j) from color c
            # This affects pairs (i, k) and (j, k) where k is a common neighbor in c?
            # Actually, it's easier to just recalculate violations for affected edges.
            # Or use the delta method.
            
            # Delta method:
            # Violations are counted for each pair (u, v) if they share >= 2 neighbors in color c(u,v).
            # Changing (i, j) from c to new_c affects:
            # 1. The pair (i, j) itself.
            # 2. Pairs (i, k) where k is a neighbor of j in new_c? No.
            # 3. Pairs (u, v) where the common neighbor count changes.
            
            # Let's use a simpler local search step: 
            # Find a violating edge and try to fix it.
            # Or just accept random moves based on energy change.
            
            # Energy = number of violations.
            # Let's compute delta in violations.
            
            # For edge (i, j):
            # Old color c. New color new_c.
            
            # 1. Pair (i, j):
            # Old common neighbors in c: N[c][i] & N[c][j]
            # But (i, j) is not a neighbor of itself.
            # The count of common neighbors in c for (i, j) is popcount(N[c][i] & N[c][j]).
            # Note: N[c][i] includes j if color(i,j)==c.
            # So N[c][i] & N[c][j] includes the bits for common neighbors.
            # If we remove (i, j) from c, N[c][i] loses j, N[c][j] loses i.
            # The intersection N[c][i] & N[c][j] is unchanged by removing i from N[c][j] and j from N[c][i]?
            # Yes, because i is not in N[c][i] and j is not in N[c][j].
            # So the common neighbor count for (i, j) in color c is unchanged by removing the edge?
            # Wait. The definition of common neighbors of u and v in color c is {w | (u,w) in c and (v,w) in c}.
            # w cannot be u or v.
            # So the set is indeed N[c][u] & N[c][v] \ {u, v}.
            # Since u is not in N[c][u] and v is not in N[c][v], the intersection is exactly the common neighbors.
            # So removing (i, j) does not change the common neighbor count for pair (i, j) in color c.
            # However, it changes the common neighbor counts for OTHER pairs.
            
            # Which pairs change?
            # A pair (u, v) changes its common neighbor count in color c if u or v is i or j.
            # Specifically, if u=i, then the common neighbors of (i, v) in color c change because j is no longer a neighbor of i in color c.
            # So j is removed from the common neighbors of (i, v) in color c IF (j, v) is in color c.
            # Similarly, i is removed from common neighbors of (j, v) in color c IF (i, v) is in color c.
            
            # This is getting complicated. Let's just recalculate violations for a subset of edges.
            # Affected edges:
            # 1. (i, j)
            # 2. (i, k) for all k
            # 3. (j, k) for all k
            
            # We can compute the violation status for these edges before and after.
            
            def get_violation_count_for_edges(edge_subset):
                count = 0
                for e_idx in edge_subset:
                    u, v = edges[e_idx]
                    c_uv = colors[e_idx]
                    common = N[c_uv][u] & N[c_uv][v]
                    if bin(common).count('1') >= 2:
                        count += 1
                return count
            
            # This is O(n) per move. With n=26, this is fast.
            # Number of affected edges is ~ 2n.
            # So O(n^2) per move.
            
            # Let's optimize by just checking the change.
            
            # Actually, let's just do a simple hill climbing / local search first.
            # Find a random violating edge, try to recolor it to fix.
            
            # Find a violating edge
            viol_idx = None
            for _ in range(10): # Try a few random edges to find a violation
                e_idx = random.randint(0, n_edges - 1)
                u, v = edges[e_idx]
                c_uv = colors[e_idx]
                common = N[c_uv][u] & N[c_uv][v]
                if bin(common).count('1') >= 2:
                    viol_idx = e_idx
                    break
            
            if viol_idx is None:
                # No violation found in sample, maybe it's valid?
                if count_violations(n, colors, N) == 0:
                    return colors
                continue
            
            u, v = edges[viol_idx]
            c_old = colors[viol_idx]
            
            # Try to recolor to a color that reduces violations
            best_new_c = None
            best_delta = 0
            
            for new_c in range(5):
                if new_c == c_old:
                    continue
                
                # Calculate delta in violations
                # We need to count violations involving (u, v) or neighbors affected by (u, v) color change.
                
                # Let's just compute the total violations for affected edges.
                
                # Save state? No, just calculate.
                
                # This is too slow to do naively.
                
                # Let's use the property that we only care about pairs (u, x) and (v, x).
                
                # Let's try a simpler heuristic:
                # Pick a random new color.
                # Accept if it doesn't increase violations.
                
                pass
            
            # Let's just pick a random new color and accept if it reduces violations.
            new_c = random.randint(0, 4)
            if new_c == c_old:
                continue
            
            # Calculate new violations
            # We need to update N and check.
            
            # To keep it simple and fast, let's just use a standard SA with full violation count update.
            # But full update is O(n^2).
            
            # Let's try a different strategy:
            # Start with a random coloring.
            # Iteratively pick a random violating edge and change its color to one that minimizes local violations.
            
            # Let's implement "Fix Violations" local search.
            
            # For now, let's just return the random coloring if we can't do better quickly.
            # But we need to find n=26.
            
            # Let's try the algebraic construction for n=26.
            # Lazebnik-Woldar construction.
            
            # Vertices are (x, y) in Z_5 x Z_5? No, that's 25 vertices.
            # The construction for n=26 involves 26 vertices.
            # One known construction uses a projective plane or similar structure.
            
            # Actually, the score 1.4545 corresponds to n=16 (16/11 = 1.4545).
            # The inspiration found n=26 (score 2.36).
            # So we MUST find n=26.
            
            # Since I don't have the exact algebraic construction memorized, I will rely on the local search to find it.
            # The local search in the inspiration worked.
            
            # Let's implement a robust local search.
            
            break # Break out of SA loop, use a dedicated local search function below
        
        return None # Failed

    def local_search(n, max_time=10.0):
        n_edges = n * (n - 1) // 2
        colors = [random.randint(0, 4) for _ in range(n_edges)]
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Precompute indices for edges
        # edge_idx[(i, j)] = idx
        
        # Build N
        N = [[0] * n for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[get_idx(i, j, n)]
                N[c][i] |= (1 << j)
                N[c][j] |= (1 << i)
        
        def get_violation_count():
            v_count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[get_idx(i, j, n)]
                    if bin(N[c][i] & N[c][j]).count('1') >= 2:
                        v_count += 1
            return v_count
        
        current_violations = get_violation_count()
        
        if current_violations == 0:
            return colors
        
        start_t = time.time()
        T = 1.0
        cooling = 0.999
        
        while T > 0.01 and time.time() - start_t < max_time:
            # Pick a random edge
            idx = random.randint(0, n_edges - 1)
            i, j = edges[idx]
            c_old = colors[idx]
            
            new_c = random.randint(0, 4)
            if new_c == c_old:
                continue
            
            # Calculate delta in violations
            # Affected pairs: (i, j), (i, k), (j, k)
            
            def count_violations_for_pairs(pairs):
                cnt = 0
                for u, v in pairs:
                    c_uv = colors[get_idx(u, v, n)]
                    if bin(N[c_uv][u] & N[c_uv][v]).count('1') >= 2:
                        cnt += 1
                return cnt
            
            # This is inefficient. Let's just update N and count all violations? No.
            
            # Let's use the delta method properly.
            # Change (i, j) from c_old to new_c.
            
            # 1. Pair (i, j):
            # Common neighbors in c_old: N[c_old][i] & N[c_old][j]
            # Common neighbors in new_c: N[new_c][i] & N[new_c][j]
            # Note: N includes the edge itself? No, N[c][i] is neighbors of i in color c.
            # So (i, j) is in N[c_old][i] and N[c_old][j].
            # But common neighbors are w != i, j.
            # So the count is popcount(N[c][i] & N[c][j]).
            
            # Before change:
            viol_old_ij = 1 if bin(N[c_old][i] & N[c_old][j]).count('1') >= 2 else 0
            
            # After change:
            # We need to simulate the change in N.
            # N[c_old][i] loses j. N[c_old][j] loses i.
            # N[new_c][i] gains j. N[new_c][j] gains i.
            
            # The common neighbors for (i, j) in c_old are unchanged (as argued before).
            # So viol_new_ij = viol_old_ij.
            
            # 2. Pairs (i, k) for k != j:
            # For each k, the common neighbors of (i, k) in color c(i, k) might change.
            # The color of (i, k) is c_ik.
            # If c_ik == c_old:
            #   j was a common neighbor of (i, k) if (j, k) was c_old.
            #   Now j is no longer a neighbor of i in c_old.
            #   So if (j, k) is c_old, the count decreases by 1.
            # If c_ik == new_c:
            #   j was NOT a common neighbor (since (i, j) was c_old, not new_c).
            #   Now j is a neighbor of i in new_c.
            #   So if (j, k) is new_c, the count increases by 1.
            
            # Similarly for pairs (j, k).
            
            # This is doable but complex.
            
            # Given the time limit, let's just do a simple hill climb:
            # Try all 5 colors for a random violating edge, pick the one with min local violations.
            
            # Find a violating edge
            viol_idx = None
            for _ in range(20):
                e_idx = random.randint(0, n_edges - 1)
                u, v = edges[e_idx]
                c_uv = colors[e_idx]
                if bin(N[c_uv][u] & N[c_uv][v]).count('1') >= 2:
                    viol_idx = e_idx
                    break
            
            if viol_idx is None:
                if get_violation_count() == 0:
                    return colors
                continue
            
            u, v = edges[viol_idx]
            c_old = colors[viol_idx]
            
            best_c = c_old
            best_score = float('inf')
            
            for new_c in range(5):
                if new_c == c_old:
                    continue
                
                # Simulate change
                # We need to count violations for (u, v), (u, k), (v, k).
                
                # This is O(n) per new_c. 5 * O(n) = O(n).
                
                # Let's implement a helper to count violations for a specific edge set given the current N and a proposed change.
                
                pass
            
            # Let's just pick a random new color and accept if it reduces total violations.
            # To do this, we need to calculate total violations, which is O(n^2).
            # But we can approximate.
            
            # Let's try a different approach:
            # Use the "Fix Violations" strategy from the inspiration.
            
            # For now, let's just return None and rely on the fact that the inspiration found n=26.
            # I will copy the logic from the inspiration that worked.
            
            break
        
        return None

    # Since I can't perfectly reconstruct the working local search from the inspiration text without seeing the full code,
    # and I don't have the algebraic construction, I will implement a simple but effective local search.
    
    def solve_n(n, time_limit=10.0):
        n_edges = n * (n - 1) // 2
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Random initialization
        colors = [random.randint(0, 4) for _ in range(n_edges)]
        
        # Build N
        N = [[0] * n for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[get_idx(i, j, n)]
                N[c][i] |= (1 << j)
                N[c][j] |= (1 << i)
        
        def get_violations():
            v = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[get_idx(i, j, n)]
                    if bin(N[c][i] & N[c][j]).count('1') >= 2:
                        v += 1
            return v
        
        current_v = get_violations()
        
        if current_v == 0:
            return colors
        
        start_t = time.time()
        T = 1.0
        cooling = 0.9995
        
        while T > 0.01 and time.time() - start_t < time_limit:
            idx = random.randint(0, n_edges - 1)
            i, j = edges[idx]
            c_old = colors[idx]
            
            new_c = random.randint(0, 4)
            if new_c == c_old:
                continue
            
            # Calculate delta in violations
            # Affected edges: (i, j), (i, k), (j, k)
            
            # We will compute the violation count for these edges before and after.
            
            def count_violations_subset(edge_indices):
                cnt = 0
                for e_idx in edge_indices:
                    u, v = edges[e_idx]
                    c_uv = colors[e_idx]
                    if bin(N[c_uv][u] & N[c_uv][v]).count('1') >= 2:
                        cnt += 1
                return cnt
            
            # This is too slow.
            
            # Let's just update N and count all violations. O(n^2).
            # With n=26, n^2=676. 10000 iterations * 676 = 6.7M operations. Fast enough.
            
            # Save old N
            N_old = [row[:] for row in N]
            
            # Update N
            N[c_old][i] &= ~(1 << j)
            N[c_old][j] &= ~(1 << i)
            N[new_c][i] |= (1 << j)
            N[new_c][j] |= (1 << i)
            
            # Update colors
            colors[idx] = new_c
            
            new_v = get_violations()
            
            delta = new_v - current_v
            
            if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                current_v = new_v
            else:
                # Revert
                N = N_old
                colors[idx] = c_old
            
            T *= cooling
        
        if current_v == 0:
            return colors
        return None

    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    for n in range(26, 10, -1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple restarts
        for restart in range(5):
            if time.time() > deadline - 2:
                break
            
            res = solve_n(n, time_limit=5.0)
            if res is not None:
                best_n = n
                best_colors = res
                break
        
        if best_n == n:
            break
    
    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    construct()
