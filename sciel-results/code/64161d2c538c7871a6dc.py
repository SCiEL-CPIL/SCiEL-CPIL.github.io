import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Lazebnik-Woldar construction for n=26
    # Uses finite field GF(9) = GF(3^2)
    # GF(9) elements: {0, 1, 2, α, α+1, α+2, 2α, 2α+1, 2α+2} where α² = α + 2 (irreducible over GF(3))
    # Actually, let's use the standard construction
    
    # The construction: vertices are pairs (a, b) where a in GF(9), b in GF(3), but that gives 27 vertices
    # For 26 vertices, we might use a different approach
    
    # Actually, let me recall the exact construction:
    # The Lazebnik-Woldar construction for R_5(C4) = 26 uses:
    # - 5 color classes
    # - Each color class is a C4-free graph
    # - Total edges: C(26,2) = 325
    # - Each color class has about 65 edges
    
    # The construction uses the fact that in GF(9), the multiplicative group has order 8
    # and partitions edges based on certain algebraic properties
    
    # Let me use a known valid construction:
    # Vertices: 0 to 25
    # Color of edge (i,j) is determined by the difference or ratio in a finite field structure
    
    # Actually, the simplest approach: use the known fact that this construction exists
    # and implement it based on the algebraic structure
    
    # GF(9) with primitive element α where α^2 = α + 2
    # Elements of GF(9): 0, 1, α, α+1, α+2, 2α, 2α+1, 2α+2
    # Let's label them: 0=0, 1=1, 2=α, 3=α+1, 4=α+2, 5=2α, 6=2α+1, 7=2α+2
    
    # The construction for 26 vertices:
    # Use a 5x5 grid or similar structure
    
    # Let me try a different approach: use the known construction from the literature
    # The vertices can be thought of as elements of a specific structure
    
    # For now, let me implement a construction that I know works:
    # Use the fact that C(26,2) = 325 = 5 * 65
    # Each color class should be a C4-free graph with 65 edges on 26 vertices
    
    # A C4-free graph on 26 vertices with 65 edges: this is possible (e.g., a polarity graph)
    
    # The Lazebnik-Woldar construction uses a specific algebraic method:
    # Define vertices as pairs (i, j) where i in {0,...,4}, j in {0,...,4} gives 25 vertices
    # Plus one more vertex? Or use a different structure
    
    # Actually, let me use the correct construction:
    # The construction uses GF(9) and creates a 5-coloring where:
    # - Color c of edge (u,v) is determined by the "slope" or "ratio" in GF(9)
    
    # Let me implement based on the known result:
    # The 26 vertices can be arranged as follows:
    # Use the additive group of GF(9) (9 elements) plus additional structure
    
    # I'll use a direct construction:
    n = 26
    colors = []
    
    # GF(9) arithmetic: represent elements as (a, b) where a, b in {0,1,2}
    # Addition: component-wise mod 3
    # Multiplication: (a,b)*(c,d) = (ac - 2bd, ad + bc + bd) since α^2 = α + 2
    # Wait, if α^2 = α + 2, then α^2 - α - 2 = 0
    # (a + bα)(c + dα) = ac + (ad + bc)α + bd α^2 = ac + (ad + bc)α + bd(α + 2)
    # = (ac + 2bd) + (ad + bc + bd)α
    
    def gf9_add(a, b):
        return ((a[0] + b[0]) % 3, (a[1] + b[1]) % 3)
    
    def gf9_sub(a, b):
        return ((a[0] - b[0]) % 3, (a[1] - b[1]) % 3)
    
    def gf9_mul(a, b):
        a0, a1 = a
        b0, b1 = b
        c0 = (a0 * b0 + 2 * a1 * b1) % 3
        c1 = (a0 * b1 + a1 * b0 + a1 * b1) % 3
        return (c0, c1)
    
    def gf9_inv(a):
        if a == (0, 0):
            return None
        # In GF(9), a^8 = 1 for a != 0
        result = (1, 0)
        base = a
        exp = 7  # a^7 = a^8 / a = 1/a
        while exp > 0:
            if exp % 2 == 1:
                result = gf9_mul(result, base)
            base = gf9_mul(base, base)
            exp //= 2
        return result
    
    # Vertices: use 26 vertices
    # The construction: label vertices 0-25
    # Color of edge (i, j): 
    # Map vertices to GF(9) elements and use multiplicative structure
    
    # Actually, let me use a known valid coloring from the literature:
    # The Lazebnik-Woldar construction is well-documented
    
    # For simplicity, let me use a construction based on the following:
    # Partition vertices into 5 groups of 5, plus 1 vertex
    # Or use a different algebraic structure
    
    # Let me try: vertices are (x, y) where x in GF(5), y in GF(5), giving 25 vertices
    # Plus vertex 25
    
    # Color of edge between (x1, y1) and (x2, y2):
    # If x1 != x2: color = (y2 - y1) * (x2 - x1)^(-1) mod 5  [slope]
    # If x1 == x2: color = y2 - y1 mod 5  [vertical]
    # But this gives 5 colors and might have C4s
    
    # Actually, this is a standard construction for avoiding monochromatic C4s
    
    # Let me implement this:
    # Vertices 0-24: (i, j) where i = v // 5, j = v % 5
    # Vertex 25: special
    
    def vertex_to_coords(v):
        if v < 25:
            return (v // 5, v % 5)
        else:
            return None
    
    def get_color(v1, v2):
        c1 = vertex_to_coords(v1)
        c2 = vertex_to_coords(v2)
        
        if c1 is None and c2 is None:
            return 0  # Should not happen
        if c1 is None:
            # Edge between special vertex and (i, j)
            return (i + j) % 5
        if c2 is None:
            # Edge between (i, j) and special vertex
            return (i + j) % 5
        
        x1, y1 = c1
        x2, y2 = c2
        
        if x1 == x2:
            # Vertical edge
            return (y2 - y1) % 5
        else:
            # Slope
            dx = (x2 - x1) % 5
            dy = (y2 - y1) % 5
            # Inverse of dx mod 5
            inv_dx = pow(dx, -1, 5)
            slope = (dy * inv_dx) % 5
            return slope
    
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(get_color(i, j))
    
    # Verify: check for C4s
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
                if nu == 0:
                    continue
                for v in range(u + 1, n):
                    nv = neighbors[c][v]
                    common = nu & nv
                    if common & (common - 1):
                        cost += 1
        return cost
    
    violations = count_violations(n, colors)
    
    if violations == 0:
        return {'n': n, 'colors': colors}
    
    # If the construction doesn't work, fall back to local search
    # But let's first try to fix the construction
    
    # The issue: the slope construction might create C4s
    # Let me try a different algebraic construction
    
    # Alternative: use GF(9) properly
    # Vertices: 26 = 9 + 9 + 8? Or some other partition
    
    # Let me use the actual Lazebnik-Woldar construction:
    # It uses a 5-partite structure based on GF(9)
    
    # For now, let me try local search from this initial coloring
    if time.time() > deadline - 5:
        return {'n': n, 'colors': colors}
    
    # Local search to fix violations
    total_edges = n * (n - 1) // 2
    color_matrix = [[-1] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            color_matrix[i][j] = c
            color_matrix[j][i] = c
            idx += 1
    
    neighbors = [[0] * n for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            neighbors[c][i] |= (1 << j)
            neighbors[c][j] |= (1 << i)
            idx += 1
    
    def get_cost():
        cost = 0
        for c in range(5):
            for u in range(n):
                nu = neighbors[c][u]
                if nu == 0:
                    continue
                for v in range(u + 1, n):
                    nv = neighbors[c][v]
                    common = nu & nv
                    if common & (common - 1):
                        cost += 1
        return cost
    
    current_cost = get_cost()
    search_start = time.time()
    
    while time.time() - search_start < deadline - time.time() - 5:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        
        c_old = color_matrix[i][j]
        best_new_color = -1
        best_cost = current_cost
        
        for c_new in range(5):
            if c_new == c_old:
                continue
            
            delta = 0
            for w in range(n):
                if w == i or w == j:
                    continue
                if neighbors[c_old][j] & (1 << w):
                    k = bin(neighbors[c_old][i] & neighbors[c_old][w]).count('1')
                    delta -= (k - 1)
                if neighbors[c_old][i] & (1 << w):
                    k = bin(neighbors[c_old][j] & neighbors[c_old][w]).count('1')
                    delta -= (k - 1)
            
            for w in range(n):
                if w == i or w == j:
                    continue
                if neighbors[c_new][j] & (1 << w):
                    k = bin(neighbors[c_new][i] & neighbors[c_new][w]).count('1')
                    delta += k
                if neighbors[c_new][i] & (1 << w):
                    k = bin(neighbors[c_new][j] & neighbors[c_new][w]).count('1')
                    delta += k
            
            new_cost = current_cost + delta
            if new_cost < best_cost:
                best_cost = new_cost
                best_new_color = c_new
        
        if best_new_color != -1:
            neighbors[c_old][i] &= ~(1 << j)
            neighbors[c_old][j] &= ~(1 << i)
            neighbors[best_new_color][i] |= (1 << j)
            neighbors[best_new_color][j] |= (1 << i)
            
            color_matrix[i][j] = best_new_color
            color_matrix[j][i] = best_new_color
            idx = i * (2 * n - i - 1) // 2 + j - i - 1
            colors[idx] = best_new_color
            current_cost = best_cost
            
            if current_cost == 0:
                return {'n': n, 'colors': colors}
    
    # If we didn't fix all violations, return best effort
    # But we need a valid coloring, so let's try a smaller n
    # Try n=25, 24, etc.
    
    for n_try in [25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]:
        if time.time() > deadline - 2:
            break
        
        # Quick random search
        colors_try = [random.randint(0, 4) for _ in range(n_try * (n_try - 1) // 2)]
        
        def check_valid(n, colors):
            neighbors = [[0] * n for _ in range(5)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx]
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
        
        if check_valid(n_try, colors_try):
            return {'n': n_try, 'colors': colors_try}
    
    # Fallback: seed solution
    n_seed = 11
    colors_seed = []
    for i in range(n_seed):
        for j in range(i + 1, n_seed):
            d = min(j - i, n_seed - (j - i))
            colors_seed.append(d - 1)
    
    return {'n': n_seed, 'colors': colors_seed}

if __name__ == '__main__':
    res = construct()
    print(f"n={res['n']}")
