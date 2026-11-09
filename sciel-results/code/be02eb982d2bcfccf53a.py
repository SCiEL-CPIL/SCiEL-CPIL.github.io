import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit

    def count_triangles_fast(mat):
        """Count monochromatic triangles using numpy for speed."""
        n = mat.shape[0]
        count = 0
        for c in range(4):
            # For each color, count triangles in the subgraph of that color
            # Triangle (i,j,k) is monochromatic color c if all three edges are color c
            # Use matrix multiplication approach
            M = (mat == c).astype(np.int32)
            np.fill_diagonal(M, 0)
            # Number of triangles = trace(M^3) / 6
            M2 = M @ M
            count += np.trace(M2 @ M) // 6
        return count

    def count_triangles_vec(mat):
        """Vectorized triangle count."""
        n = mat.shape[0]
        total = 0
        for c in range(4):
            M = (mat == c).astype(np.int32)
            np.fill_diagonal(M, 0)
            # trace(M^3) / 6
            M2 = M @ M
            total += int(np.trace(M2 @ M)) // 6
        return total

    def find_mono_triangle_fast(mat):
        """Find a monochromatic triangle using vectorized approach."""
        n = mat.shape[0]
        # Try random sampling first
        for _ in range(50):
            i, j, k = random.sample(range(n), 3)
            if mat[i][j] == mat[i][k] == mat[j][k]:
                return (i, j, k)
        # Systematic search
        for c in range(4):
            M = (mat == c)
            for i in range(n):
                for j in range(i+1, n):
                    if M[i][j]:
                        # Find common neighbors
                        common = np.where(M[i] & M[j])[0]
                        for k in common:
                            if k > j:
                                return (i, j, k)
        return None

    def local_search_sa(mat, n, time_budget):
        """Simulated annealing to eliminate monochromatic triangles."""
        current_cost = count_triangles_vec(mat)
        if current_cost == 0:
            return mat, True

        best_mat = mat.copy()
        best_cost = current_cost

        T = 2.0
        cooling = 0.99995
        start = time.time()

        # Precompute adjacency for each color for fast delta computation
        while time.time() - start < time_budget:
            if current_cost == 0:
                return mat, True

            # Find a triangle to fix
            tri = find_mono_triangle_fast(mat)
            if tri:
                i, j, k = tri
                edge_idx = random.randint(0, 2)
                if edge_idx == 0:
                    u, v = i, j
                elif edge_idx == 1:
                    u, v = i, k
                else:
                    u, v = j, k
            else:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue

            old_color = mat[u][v]
            new_color = random.randint(0, 3)
            if new_color == old_color:
                continue

            # Compute delta in O(n)
            delta = 0
            for w in range(n):
                if w == u or w == v:
                    continue
                if mat[u][w] == old_color and mat[v][w] == old_color:
                    delta -= 1
                if mat[u][w] == new_color and mat[v][w] == new_color:
                    delta += 1

            new_cost = current_cost + delta

            if new_cost < current_cost:
                mat[u][v] = new_color
                mat[v][u] = new_color
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_mat = mat.copy()
            elif delta == 0:
                mat[u][v] = new_color
                mat[v][u] = new_color
                current_cost = new_cost
            else:
                if random.random() < np.exp(-delta / max(T, 1e-10)):
                    mat[u][v] = new_color
                    mat[v][u] = new_color
                    current_cost = new_cost

            T *= cooling
            T = max(T, 0.01)

            if time.time() > deadline:
                break

        return best_mat, best_cost == 0

    def gf49_construct():
        """
        Construct a 4-coloring of K_49 using GF(49) with a proper difference set.
        
        The key insight: we need a (49, 4, ?) difference set or use a known
        construction. For R(3,3,3,3), the known 49-vertex construction uses
        the fact that GF(49) has a Singer difference set.
        
        Actually, let's use a simpler approach: use the fact that we can
        construct a 4-coloring from a 3-coloring of K_16 (from GF(16))
        by a product construction, or use the known 49-vertex construction.
        
        A better approach: use the Paley-type construction.
        In GF(49), the multiplicative group has order 48. We need to partition
        the 48 non-zero elements into 4 sets D_0, D_1, D_2, D_3 such that
        each D_i is a (49, 4, 12) difference set or similar.
        
        Actually, for a triangle-free 4-coloring, we need that for each color c,
        the graph with edges of color c is triangle-free. This means D_c should
        be a set with no 3 elements x,y,z in D_c such that x+y=z (in the group sense).
        
        Let me use a known construction: the 49-vertex 4-coloring from
        the projective plane of order 7.
        
        Alternative: use a random initialization and local search.
        """
        n = 49
        # Initialize with random coloring
        mat = np.random.randint(0, 4, (n, n), dtype=np.int8)
        np.fill_diagonal(mat, -1)
        for i in range(n):
            for j in range(i+1, n):
                mat[j][i] = mat[i][j]
        
        return mat

    def gf49_construct_structured():
        """
        Use a structured construction for GF(49).
        
        In GF(49) with primitive element g, the multiplicative group is cyclic of order 48.
        We can use the fact that 48 = 4 * 12.
        
        For a triangle-free coloring, we need each color class to be triangle-free.
        A sufficient condition: if D is a set of elements such that no three distinct
        elements of D sum to zero (in the additive group), then D is triangle-free
        in the Cayley graph sense.
        
        Let's use a known difference set construction.
        The (49, 8, 1) difference set in GF(49) can help, but we need 4 colors.
        
        Let me try: partition the 48 non-zero elements into 4 sets of 12,
        where each set is a (49, 12, 3) difference set.
        
        Actually, let me just use a well-known construction:
        Use the fact that 49 = 7^2, and use the 7x7 grid structure.
        """
        n = 49
        mat = np.zeros((n, n), dtype=np.int8)
        
        # Represent elements as (a, b) in GF(7)^2
        # Use a structured coloring based on the difference (a1-a2, b1-b2) mod 7
        # Color based on which "sector" the difference falls in
        
        # Let's use: color = (a_diff * 3 + b_diff) mod 4 where a_diff, b_diff are differences
        # This might not work, so let's try multiple structured approaches
        
        # Approach 1: Use (a_diff mod 4, b_diff mod 4) mapped to colors
        # Actually, let's use a simple hash-like function
        for i in range(n):
            ai, bi = i // 7, i % 7
            for j in range(i+1, n):
                aj, bj = j // 7, j % 7
                da = (ai - aj) % 7
                db = (bi - bj) % 7
                # Map to 4 colors
                c = (da * 3 + db * 5) % 4
                mat[i][j] = c
                mat[j][i] = c
        
        return mat

    def try_extend(mat, n, time_budget):
        """Try to extend the coloring by adding vertices one at a time."""
        current_n = n
        current_mat = mat.copy()
        
        while time.time() < start_time + time_budget:
            new_n = current_n + 1
            if new_n > 62:
                break
            
            new_mat = np.zeros((new_n, new_n), dtype=np.int8)
            new_mat[:current_n, :current_n] = current_mat
            
            # Try to assign colors to edges from new vertex to existing vertices
            # such that no monochromatic triangle is formed
            success = True
            for i in range(current_n):
                # For each existing vertex i, we need to choose a color for edge (i, new_n-1)
                # such that for every j < i, if edge (i,j) has color c, then edge (j, new_n-1) != c
                # and edge (i, new_n-1) != c
                forbidden = set()
                for j in range(i):
                    c = current_mat[i][j]
                    if current_mat[j, new_n-1] >= 0:
                        forbidden.add(c)
                
                available = [c for c in range(4) if c not in forbidden]
                if not available:
                    success = False
                    break
                # Choose randomly from available
                c = random.choice(available)
                new_mat[i, new_n-1] = c
                new_mat[new_n-1, i] = c
            
            if success:
                current_n = new_n
                current_mat = new_mat
            else:
                break
        
        return current_n, current_mat

    # First, try structured GF(49) construction
    mat49 = gf49_construct_structured()
    tri_count = count_triangles_vec(mat49)

    if tri_count == 0:
        best_n = 49
        best_colors = []
        for i in range(49):
            for j in range(i+1, 49):
                best_colors.append(int(mat49[i][j]))
        return {'n': best_n, 'colors': best_colors}

    # Try to fix with local search
    time_budget = min(60.0, deadline - 10 - (time.time() - start_time))
    if time_budget > 5:
        res_mat, success = local_search_sa(mat49.copy(), 49, time_budget)
        if success:
            best_n = 49
            best_colors = []
            for i in range(49):
                for j in range(i+1, 49):
                    best_colors.append(int(res_mat[i][j]))
            return {'n': best_n, 'colors': best_colors}

    # Try random initialization + local search
    remaining = deadline - 10 - (time.time() - start_time)
    if remaining > 10:
        mat_rand = np.random.randint(0, 4, (49, 49), dtype=np.int8)
        np.fill_diagonal(mat_rand, -1)
        for i in range(49):
            for j in range(i+1, 49):
                mat_rand[j][i] = mat_rand[i][j]
        
        res_mat, success = local_search_sa(mat_rand, 49, remaining)
        if success:
            best_n = 49
            best_colors = []
            for i in range(49):
                for j in range(i+1, 49):
                    best_colors.append(int(res_mat[i][j]))
            return {'n': best_n, 'colors': best_colors}

    # Fallback: use the known 32-vertex construction
    def get_seed_solution():
        def mul(a, b):
            r = 0
            for i in range(4):
                if b >> i & 1:
                    r ^= a << i
            for i in range(7, 3, -1):
                if r >> i & 1:
                    r ^= 0b10011 << (i - 4)
            return r
        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return {'n': 32, 'colors': colors}

    seed_res = get_seed_solution()
    return {'n': seed_res['n'], 'colors': seed_res['colors']}
