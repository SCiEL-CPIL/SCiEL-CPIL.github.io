import time
import random

def construct():
    deadline = time.time() + 145.0
    
    def is_valid(n, colors):
        """
        Verifies if the coloring has no monochromatic C4.
        """
        if len(colors) != n * (n - 1) // 2:
            return False
        
        neighbors = [[0] * n for _ in range(5)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                if c < 0 or c > 4:
                    return False
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
                idx += 1
        
        for c in range(5):
            for u in range(n):
                nu = neighbors[c][u]
                if nu == 0:
                    continue
                for v in range(u + 1, n):
                    nv = neighbors[c][v]
                    common = nu & nv
                    if common & (common - 1):
                        return False
        return True

    def count_c4_violations(neighbors):
        """
        Count the number of C4 violations across all colors.
        A violation is a pair (u, v) in color c that share >= 2 common neighbors.
        We count 1 for each such pair.
        """
        violations = 0
        for c in range(5):
            ncbits = neighbors[c]
            for u in range(n):
                nu = ncbits[u]
                # We only need to check pairs (u, v) where u < v.
                # To avoid double counting, we iterate v > u.
                # But iterating all v is O(n^2).
                # We can optimize by only iterating v in the set of neighbors?
                # No, we need to check common neighbors of u and v in color c.
                # The condition is popcount(nu & nv) >= 2.
                # This requires iterating v.
                # However, we can stop early if we find violations, but we need the count for SA.
                # Let's just do the standard O(n^2 * 5) check but optimized with bit operations.
                for v in range(u + 1, n):
                    nv = ncbits[v]
                    common = nu & nv
                    if common & (common - 1): # popcount >= 2
                        violations += 1
        return violations

    def solve_bitset(n, time_limit):
        """
        Local search using bitsets for fast neighborhood checks.
        """
        # Initialize random coloring
        colors = [random.randint(0, 4) for _ in range(n * (n - 1) // 2)]
        
        # Build initial bitsets
        neighbors = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
                idx += 1
        
        current_viol = count_c4_violations(neighbors)
        best_viol = current_viol
        best_colors = colors.copy()
        
        if best_viol == 0:
            return best_colors
        
        T = 1.0
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            # Pick a random edge
            idx = random.randint(0, n * (n - 1) // 2 - 1)
            
            # Find (i, j) from idx
            # k = i*n - i*(i+1)/2 + (j-i-1)
            # This is slow. Precompute edge indices?
            # Or just pick i, j randomly and find idx?
            # Picking i, j randomly is O(1).
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            # Find current color
            # We need a fast way to get color(i,j).
            # We can maintain a 2D array color[i][j] or a flat array and compute index.
            # Flat index: i*(n-1) - i*(i+1)//2 + (j-i-1)
            # Let's maintain a 2D list for speed? n is small (<=29).
            # Or just recompute? No, we need to update it.
            # Let's maintain a 2D array `col` of size n x n (upper triangle used).
            # But we already have `colors` flat list.
            # Let's maintain `col_2d` alongside.
            
            # Actually, let's restructure to use a 2D array for colors to make lookup O(1).
            # But we need to return the flat list at the end.
            
            # Let's restart the solver with 2D array.
            return None # Placeholder, will implement below

    def solve_bitset_v2(n, time_limit):
        """
        Local search using bitsets and 2D color array.
        """
        col = [[-1] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                col[i][j] = random.randint(0, 4)
                col[j][i] = col[i][j]
        
        neighbors = [[0] * n for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                c = col[i][j]
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
        
        current_viol = count_c4_violations(neighbors)
        best_viol = current_viol
        best_col = [row[:] for row in col]
        
        if best_viol == 0:
            return best_col
        
        T = 1.0
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old_c = col[i][j]
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
            
            # Update bitsets
            neighbors[old_c][i] &= ~(1 << j)
            neighbors[old_c][j] &= ~(1 << i)
            neighbors[new_c][i] |= (1 << j)
            neighbors[new_c][j] |= (1 << i)
            
            # Calculate delta in violations
            # Only violations involving (i, j) or paths through i, j can change?
            # Actually, C4s are cycles of length 4.
            # Changing color of (i, j) affects:
            # 1. C4s that use edge (i, j).
            # 2. C4s where i and j are opposite vertices? No, edges are colored.
            # A C4 is a set of 4 vertices {a, b, c, d} and 4 edges.
            # If we change color of (i, j), any C4 that includes edge (i, j) is affected.
            # Also, if i and j are in the same color class, they might share neighbors.
            # The violation condition is: for any pair (u, v) in color c, popcount(N_c(u) & N_c(v)) >= 2.
            # Changing (i, j) from c1 to c2:
            # - For c1: (i, j) is removed from the set of edges in c1.
            #   This removes violations where (i, j) was the pair.
            #   Also, for any other pair (u, v) in c1, the common neighbors might change?
            #   No, N_c1(i) and N_c1(j) change.
            #   So pairs (i, u) and (j, u) for u in V might have changed common neighbors.
            #   Specifically, u is a common neighbor of i and j in c1?
            #   If u was a common neighbor of i and j in c1, then (i, u) and (j, u) were edges in c1? No.
            #   N_c1(i) contains neighbors of i via c1 edges.
            #   So if (i, u) is c1 and (j, u) is c1, then u is in N_c1(i) & N_c1(j).
            #   So the pair (i, j) in c1 has common neighbors N_c1(i) & N_c1(j).
            #   If popcount >= 2, it's a violation.
            #   When we remove (i, j) from c1:
            #   - The pair (i, j) no longer exists in c1. So we don't check it.
            #   - But N_c1(i) loses j, and N_c1(j) loses i.
            #   - This affects the common neighbor sets of pairs involving i or j.
            #   - Specifically, for any u, the pair (i, u) in c1:
            #     Common neighbors are N_c1(i) & N_c1(u).
            #     N_c1(i) changed (lost j).
            #     So if j was in N_c1(u), then j was a common neighbor.
            #     So the count of common neighbors for (i, u) might decrease by 1 if (j, u) was c1.
            #     Similarly for (j, u).
            #   - This is getting complex to calculate delta exactly.
            #   Given n is small (<=29), and we have bitsets, we can just recalculate the total violations?
            #   O(n^2 * 5) = 29^2 * 5 ~ 4000 ops.
            #   This is very fast.
            #   Let's just recalculate total violations.
            
            new_viol = count_c4_violations(neighbors)
            delta = new_viol - current_viol
            
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                current_viol = new_viol
                col[i][j] = new_c
                col[j][i] = new_c
                if current_viol < best_viol:
                    best_viol = current_viol
                    best_col = [row[:] for row in col]
                    if best_viol == 0:
                        return best_col
            else:
                # Revert
                neighbors[old_c][i] |= (1 << j)
                neighbors[old_c][j] |= (1 << i)
                neighbors[new_c][i] &= ~(1 << j)
                neighbors[new_c][j] &= ~(1 << i)
            
            T *= 0.999
            if T < 0.01:
                T = 0.5
        
        return best_col

    def col_to_flat(col, n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(col[i][j])
        return colors

    # Fallback solution n=11
    def get_seed_solution():
        n = 11
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = min(j - i, n - (j - i))
                colors.append(d - 1)
        return n, colors

    best_n, best_colors = get_seed_solution()
    
    # Try n=26
    if time.time() < deadline - 10:
        res_col = solve_bitset_v2(26, deadline - time.time() - 5)
        if res_col is not None:
            colors_flat = col_to_flat(res_col, 26)
            if is_valid(26, colors_flat):
                return {'n': 26, 'colors': colors_flat}
    
    # Try n=25
    if time.time() < deadline - 5:
        res_col = solve_bitset_v2(25, deadline - time.time() - 2)
        if res_col is not None:
            colors_flat = col_to_flat(res_col, 25)
            if is_valid(25, colors_flat):
                return {'n': 25, 'colors': colors_flat}
    
    # Try n=24
    if time.time() < deadline - 2:
        res_col = solve_bitset_v2(24, deadline - time.time() - 1)
        if res_col is not None:
            colors_flat = col_to_flat(res_col, 24)
            if is_valid(24, colors_flat):
                return {'n': 24, 'colors': colors_flat}

    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
