import time
import random
import sys

def construct():
    """
    Finds a 5-coloring of K_n without monochromatic C4.
    Maximizes n.
    """
    deadline = time.time() + 110.0  # Safety margin for 120s limit
    
    # Helper to get edge index in lexicographic order
    def get_edge_index(i, j, n):
        if i > j:
            i, j = j, i
        return i * (i - 1) // 2 + (j - i - 1)

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

    def build_n25_solution():
        """
        Builds the explicit Lazebnik-Woldar style solution for n=25.
        """
        n = 25
        colors = []
        # Map vertex index to (x, y) in Z5 x Z5
        def get_coords(idx):
            return (idx % 5, idx // 5)
        
        for i in range(n):
            xi, yi = get_coords(i)
            for j in range(i + 1, n):
                xj, yj = get_coords(j)
                dx = (xj - xi) % 5
                dy = (yj - yi) % 5
                if dx == 0:
                    c = 0
                else:
                    # Inverse of dx mod 5
                    # 1->1, 2->3, 3->2, 4->4
                    inv_dx = pow(dx, 4, 5)
                    m = (dy * inv_dx) % 5
                    c = m
                colors.append(c)
        return n, colors

    def extend_to_n26(n25_colors):
        """
        Attempts to extend the n=25 solution to n=26 by adding a vertex.
        Uses SA on the 25 edges incident to the new vertex.
        """
        n_core = 25
        n_new = 26
        
        # Precompute conflict graphs Q_k for the core
        # Q_k[a][c] is True if a and c share a common neighbor in color k in the core
        Q = [[False] * n_core for _ in range(n_core)] # Reuse for all k? No, need per k.
        Q_k = [[0] * n_core for _ in range(5)] # Bitmasks for speed? Or just boolean matrices.
        # Let's use bitmasks for Q_k to check conflicts fast.
        # Q_mask[k][a] is bitmask of vertices c that conflict with a in color k.
        Q_mask = [[0] * n_core for _ in range(5)]
        
        # Build neighbor bitsets for core
        neighbors = [[0] * n_core for _ in range(5)]
        idx = 0
        for i in range(n_core):
            for j in range(i + 1, n_core):
                c = n25_colors[idx]
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
                idx += 1
        
        # Compute Q_mask
        for k in range(5):
            for a in range(n_core):
                mask = 0
                na = neighbors[k][a]
                # Iterate through neighbors of a
                # To find common neighbors with c, we can just compute (na & nc) popcount
                # But we need Q_mask[k][a] to contain c if popcount(na & nc) >= 1?
                # No, C4 in extension: 25-a-b-c-25.
                # a, c in N_k(25).
                # Path a-b-c in core color k.
                # So a and c share a common neighbor b in core color k.
                # So Q_mask[k][a] should contain c if they share a common neighbor.
                # We can compute this by:
                # For each neighbor b of a in color k, add all neighbors of b in color k to Q_mask[k][a].
                nb_list = []
                temp = na
                while temp:
                    lsb = temp & -temp
                    b = lsb.bit_length() - 1
                    nb_list.append(b)
                    temp -= lsb
                
                for b in nb_list:
                    Q_mask[k][a] |= neighbors[k][b]
                # Remove self? a is not a neighbor of itself in simple graph.
                # But b is neighbor of a, and a is neighbor of b.
                # neighbors[k][b] includes a.
                # So Q_mask[k][a] includes a.
                # We don't care about a==c in the conflict check if we ensure a!=c in the pair loop.
        
        # SA on 25 variables (colors for edges (25, i))
        # cost = sum over k of number of pairs (a,c) in S_k that are in Q_mask[k]
        
        def calc_cost(ext_colors):
            cost = 0
            # Group vertices by color
            # S_k = [i for i, c in enumerate(ext_colors) if c == k]
            # For each k, check pairs in S_k
            # Faster: iterate pairs (a,c) and check if same color and conflict
            for a in range(n_core):
                ca = ext_colors[a]
                # Only check c > a to avoid double counting?
                # Cost is number of conflicting pairs.
                mask_conflicts = Q_mask[ca][a]
                # Iterate c > a
                for c in range(a + 1, n_core):
                    if ext_colors[c] == ca:
                        if mask_conflicts & (1 << c):
                            cost += 1
            return cost

        # Initialize ext_colors randomly
        ext_colors = [random.randint(0, 4) for _ in range(n_core)]
        current_cost = calc_cost(ext_colors)
        
        best_cost = current_cost
        best_ext = ext_colors[:]
        
        T = 1.0
        cooling = 0.999
        start_time = time.time()
        time_limit = 10.0 # 10 seconds for this targeted search
        
        while time.time() - start_time < time_limit:
            if best_cost == 0:
                break
            i = random.randint(0, n_core - 1)
            old_c = ext_colors[i]
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
            
            ext_colors[i] = new_c
            new_cost = calc_cost(ext_colors)
            
            delta = new_cost - current_cost
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_ext = ext_colors[:]
                    if best_cost == 0:
                        break
            else:
                ext_colors[i] = old_c
            T *= cooling
        
        if best_cost == 0:
            # Construct full n=26 coloring
            full_colors = n25_colors[:]
            # Append edges (25, i) in lexicographic order?
            # Lex order: (0,1), (0,2)... (0,24), (1,2)... (1,24)... (24,25)
            # Wait, the new vertex is 25.
            # Edges are (i, 25) for i=0..24.
            # In lexicographic order of (u,v) with u<v:
            # (0,25), (1,25), ..., (24,25) are NOT at the end.
            # The edges are interleaved.
            # (0,1)..(0,24) are indices 0..23.
            # (0,25) is index 24.
            # (1,2)..(1,24) are next.
            # (1,25) is next.
            # So we need to insert them correctly.
            
            # Easier: create a dict or list of size n*(n-1)/2
            n = 26
            total_edges = n * (n - 1) // 2
            result_colors = [0] * total_edges
            
            # Fill core edges
            idx = 0
            for i in range(n_core):
                for j in range(i + 1, n_core):
                    result_colors[idx] = n25_colors[idx]
                    idx += 1
            
            # Fill new edges
            # We need to map (i, 25) to the correct index.
            # Index of (i, j) is i*(i-1)//2 + (j-i-1)
            for i in range(n_core):
                j = 25
                if i > j: i, j = j, i
                e_idx = i * (i - 1) // 2 + (j - i - 1)
                result_colors[e_idx] = best_ext[i]
            
            return result_colors
        return None

    # 1. Get n=25 base
    n25, colors25 = build_n25_solution()
    
    # Verify n=25
    if is_valid(n25, colors25):
        # 2. Try to extend to n=26
        res26 = extend_to_n26(colors25)
        if res26:
            if is_valid(26, res26):
                return {'n': 26, 'colors': res26}
    
    # 3. If extension failed, run general SA for n=26
    # We can reuse the logic from the original prompt but with better time management
    # Since we already spent 10s, we have ~100s left.
    
    def solve_sa_general(n, time_limit, init_colors=None):
        total_edges = n * (n - 1) // 2
        if init_colors:
            colors = init_colors[:]
        else:
            colors = [random.randint(0, 4) for _ in range(total_edges)]
            
        # Use the same incremental logic? 
        # For simplicity and speed, we'll use the full recalculation but it's O(n^2).
        # n=26, n^2=676. 676 ops is fast.
        
        def count_violations(n, colors):
            neighbors = [[0] * n for _ in range(5)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx]
                    neighbors[c][i] |= (1 << j)
                    neighbors[c][j] |= (1 << i)
                    idx += 1
            cost = 0
            for c in range(5):
                for u in range(n):
                    nu = neighbors[c][u]
                    if nu == 0: continue
                    for v in range(u + 1, n):
                        nv = neighbors[c][v]
                        common = nu & nv
                        if common & (common - 1):
                            cost += 1
            return cost

        current_cost = count_violations(n, colors)
        best_cost = current_cost
        best_colors = colors[:]
        
        T = 1.0
        cooling = 0.9995
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            if best_cost == 0:
                break
            e_idx = random.randint(0, total_edges - 1)
            # Find u, v for e_idx
            # Inverse of i*(i-1)//2 + (j-i-1)
            # Approx i = sqrt(2*e_idx)
            i = int((2 * e_idx + 1) ** 0.5)
            # Adjust i
            base = i * (i - 1) // 2
            if e_idx < base:
                i -= 1
                base = i * (i - 1) // 2
            j = i + 1 + (e_idx - base)
            
            old_c = colors[e_idx]
            new_c = random.randint(0, 4)
            if new_c == old_c: continue
            
            colors[e_idx] = new_c
            new_cost = count_violations(n, colors)
            
            delta = new_cost - current_cost
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
                    if best_cost == 0:
                        break
            else:
                colors[e_idx] = old_c
            T *= cooling
            
        return best_colors if best_cost == 0 else None

    # Try n=26 with SA
    remaining_time = deadline - time.time() - 2.0
    if remaining_time > 5:
        # Start from random or from the failed extension attempt?
        # Random is safer to avoid bias.
        res = solve_sa_general(26, remaining_time)
        if res:
            if is_valid(26, res):
                return {'n': 26, 'colors': res}
                
    # Fallback to n=25
    return {'n': n25, 'colors': colors25}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
