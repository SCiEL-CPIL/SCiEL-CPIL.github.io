import time
import random
import math
import sys

def construct():
    """
    Find a 5-coloring of K_n with no monochromatic C4, maximizing n.
    
    Uses an efficient simulated annealing approach with incremental energy updates.
    """
    start_time = time.time()
    deadline = start_time + 115  # 115 seconds, leaving 5s margin
    
    def time_left():
        return deadline - time.time()
    
    def validate(colors, n):
        """Check if no color class contains a C4."""
        # For each color, build adjacency and check for C4
        for c in range(5):
            adj = [0] * n
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    if colors[idx] == c:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                    idx += 1
            # Check for C4: any two vertices with >= 2 common neighbors
            for i in range(n):
                for j in range(i+1, n):
                    if i == j: continue
                    # Count common neighbors
                    common = adj[i] & adj[j]
                    # If there are at least 2 common neighbors, a C4 exists
                    # We can check if (common & (common-1)) is not 0
                    if common & (common - 1):
                        return False
        return True

    def solve_for_n(n):
        """
        Attempt to find a valid 5-coloring for K_n using Simulated Annealing.
        Returns the coloring if found, else None.
        """
        num_edges = n * (n - 1) // 2
        
        # Precompute edge indices for (i, j)
        # index = i*n - i*(i+1)//2 + (j-i-1)
        
        def get_idx(i, j):
            if i > j:
                i, j = j, i
            return i * n - i * (i + 1) // 2 + (j - i - 1)
        
        # We will store colors in a list of size num_edges
        # But for fast access in energy calculation, it's better to store
        # adjacency bitmasks for each color.
        
        # State:
        # colors: list of ints, length num_edges
        # adj: list of 5 lists, each containing n integers (bitmasks)
        
        def init_state():
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            adj = [[0] * n for _ in range(5)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx]
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
                    idx += 1
            return colors, adj
            
        def calc_energy(adj):
            """Calculate total number of C4s."""
            energy = 0
            for c in range(5):
                for i in range(n):
                    for j in range(i + 1, n):
                        common = adj[c][i] & adj[c][j]
                        # Count bits in common
                        # If count >= 2, it's a violation.
                        # We count the number of pairs of common neighbors.
                        # Actually, the definition in the prompt is "no two vertices with two common neighbours".
                        # So if common has k bits, there are C(k, 2) such pairs.
                        # But usually "contains a C4" means at least one C4.
                        # The prompt says: "No colour class may contain a four-cycle (no two vertices with two common neighbours in one colour)."
                        # This implies we want 0 pairs of vertices with >= 2 common neighbors.
                        # So energy = sum over all pairs (i,j) of max(0, C(common_neighbors, 2))?
                        # Or just count the number of C4s?
                        # Let's count the number of distinct C4s.
                        # A C4 is defined by 4 vertices.
                        # If i and j have 2 common neighbors k, l, then (i,k,j,l,i) is a C4?
                        # Yes.
                        # So for each pair (i,j) in color c, if they have m common neighbors,
                        # they form C(m, 2) C4s? No, that counts each C4 multiple times?
                        # A C4 has 4 edges. It is counted for each of its 2 pairs of opposite vertices.
                        # So if we sum C(m_ij, 2) over all pairs, each C4 is counted exactly twice.
                        # So energy = 0.5 * sum C(m_ij, 2).
                        # Or we can just use the number of "bad pairs" as the energy, which is simpler.
                        # Let's use the number of pairs (i,j) with m_ij >= 2.
                        # This is a valid objective function to minimize to 0.
                        m = bin(common).count('1')
                        if m >= 2:
                            energy += 1
            return energy

        # Initial state
        colors, adj = init_state()
        current_energy = calc_energy(adj)
        
        if current_energy == 0:
            return colors
            
        best_energy = current_energy
        best_colors = colors[:]
        
        temperature = 10.0
        cooling_rate = 0.999
        min_temperature = 0.01
        
        # Precompute neighbors for each vertex to speed up delta calculation?
        # Delta calculation for changing edge (u, v) from c1 to c2:
        # 1. Remove (u, v) from c1.
        #    This affects pairs (i, j) such that i or j is u or v?
        #    No, it affects the common neighbor count for any pair (i, j) where
        #    u is a common neighbor of i and j in c1, OR v is a common neighbor.
        #    Specifically, for any pair (i, j) != (u, v), if u is adjacent to both i and j in c1,
        #    then the common neighbor count for (i, j) decreases by 1.
        #    Similarly for v.
        #    Also, the pair (u, v) itself is removed from c1, so its common neighbor count changes.
        #    Actually, the common neighbor count for (u, v) in c1 is the number of w such that
        #    (u,w) and (v,w) are in c1. Removing (u,v) doesn't change this count directly,
        #    but it changes the adjacency of u and v.
        #    Wait, the energy is based on common neighbors of pairs in the *same* color.
        #    If we change color of (u, v) from c1 to c2:
        #    - In c1: (u, v) is no longer an edge.
        #      This affects:
        #      1. The pair (u, v) itself: it is no longer in c1, so we don't count its common neighbors in c1.
        #      2. Pairs (i, j) where u is a common neighbor in c1. i.e., (i, u) in c1 and (j, u) in c1.
        #         Removing (u, v) means v is no longer a neighbor of u in c1.
        #         So for any i, the pair (i, v) in c1: u is no longer a common neighbor of i and v.
        #         So the common neighbor count for (i, v) decreases by 1.
        #      3. Similarly, pairs (i, u) in c1: v is no longer a common neighbor of i and u.
        #         So the common neighbor count for (i, u) decreases by 1.
        #    - In c2: (u, v) is now an edge.
        #      1. The pair (u, v) is now in c2. We need to count its common neighbors in c2.
        #      2. Pairs (i, v) in c2: u is now a common neighbor of i and v.
        #         So the common neighbor count for (i, v) increases by 1.
        #      3. Pairs (i, u) in c2: v is now a common neighbor of i and u.
        #         So the common neighbor count for (i, u) increases by 1.
        
        # To implement this efficiently, we need to know the current common neighbor counts for all pairs.
        # Let's maintain a matrix `cn[c][i][j]` = number of common neighbors of i and j in color c.
        # This is O(5 * n^2) memory, which is 5 * 26^2 = 3380 ints. Very small.
        
        # Let's restart with this approach.
        
        def init_state_v2():
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            adj = [[0] * n for _ in range(5)]
            cn = [[[0] * n for _ in range(n)] for _ in range(5)]
            
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx]
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
                    idx += 1
            
            # Calculate initial CNs
            for c in range(5):
                for i in range(n):
                    for j in range(i + 1, n):
                        common = adj[c][i] & adj[c][j]
                        cn[c][i][j] = bin(common).count('1')
                        cn[c][j][i] = cn[c][i][j]
            return colors, adj, cn

        def calc_energy_v2(cn):
            energy = 0
            for c in range(5):
                for i in range(n):
                    for j in range(i + 1, n):
                        if cn[c][i][j] >= 2:
                            energy += 1
            return energy

        colors, adj, cn = init_state_v2()
        current_energy = calc_energy_v2(cn)
        
        if current_energy == 0:
            return colors
            
        best_energy = current_energy
        best_colors = colors[:]
        
        temperature = 5.0
        cooling_rate = 0.9995
        min_temperature = 0.01
        
        steps_per_temp = 100
        
        while temperature > min_temperature:
            if time_left() < 2:
                break
                
            for _ in range(steps_per_temp):
                if time_left() < 2:
                    break
                    
                # Pick random edge
                i = random.randint(0, n - 2)
                j = random.randint(i + 1, n - 1)
                ei = get_idx(i, j)
                
                old_color = colors[ei]
                new_color = random.randint(0, 4)
                if new_color == old_color:
                    continue
                
                # Calculate Delta Energy
                delta = 0
                
                # 1. Remove from old_color
                # Affected pairs in old_color:
                # a. (i, j) itself: was in old_color, now not.
                #    Its contribution was 1 if cn[old][i][j] >= 2, else 0.
                #    Now it's 0.
                #    So delta -= (1 if cn[old][i][j] >= 2 else 0)
                
                # b. Pairs (k, i) where k is neighbor of i in old_color (k != j)
                #    i is a common neighbor of k and i? No, i is one of the vertices.
                #    Wait, the common neighbor count for pair (k, i) depends on neighbors of k and i.
                #    If we remove edge (i, j), then j is no longer a neighbor of i in old_color.
                #    So for any k such that (k, j) is in old_color, j was a common neighbor of k and i?
                #    No. j is a neighbor of i. k is a neighbor of j.
                #    So j is a common neighbor of i and k?
                #    Yes, if (i, j) in c and (k, j) in c, then j is a common neighbor of i and k.
                #    So removing (i, j) decreases cn[old][i][k] by 1 for all k in N_old(i) \ {j}?
                #    No, k must be a neighbor of j.
                #    So for each k in N_old(j) \ {i}, the pair (i, k) has its common neighbor count decreased by 1 (because j is no longer a common neighbor? No, j is still a neighbor of k, but i is no longer a neighbor of j? No, i is still a neighbor of j? No, we are removing (i, j). So i is no longer a neighbor of j. So j is no longer a common neighbor of i and k?
                #    Let's trace:
                #    Common neighbors of i and k in c: w such that (i, w) in c and (k, w) in c.
                #    If w = j, then (i, j) in c and (k, j) in c.
                #    If we remove (i, j), then (i, j) is not in c. So j is no longer a common neighbor.
                #    So cn[c][i][k] decreases by 1 for all k such that (k, j) in c and k != i.
                #    Similarly, cn[c][j][k] decreases by 1 for all k such that (k, i) in c and k != j.
                
                # Let's implement the update carefully.
                
                # Old Color Updates
                c_old = old_color
                # (i, j) pair itself
                if cn[c_old][i][j] >= 2:
                    delta -= 1
                # Note: cn[c_old][i][j] will be updated later, but we need the old value for delta.
                
                # Pairs involving i and neighbors of j in c_old
                # For each k in N(c_old, j) \ {i}:
                # cn[c_old][i][k] decreases by 1.
                # Check if this crosses the threshold 2 -> 1 (delta += 1) or 1 -> 0 (no change) or 2 -> 1 (delta += 1? No, 2 is bad, 1 is good. So if it was 2 and becomes 1, energy decreases by 1. If it was 3 and becomes 2, energy stays 1? No, energy is count of pairs with cn >= 2.
                # If cn was 2, energy contribution 1. New cn 1, energy contribution 0. Delta -1.
                # If cn was 3, energy contribution 1. New cn 2, energy contribution 1. Delta 0.
                # If cn was 1, energy contribution 0. New cn 0, energy contribution 0. Delta 0.
                
                mask_j_old = adj[c_old][j]
                # Iterate through neighbors of j in c_old
                # We can iterate through bits in mask_j_old
                k = 0
                m = mask_j_old
                while m:
                    if m & 1:
                        if k != i:
                            # Update cn[c_old][i][k]
                            old_cn = cn[c_old][i][k]
                            new_cn = old_cn - 1
                            if old_cn >= 2 and new_cn < 2:
                                delta -= 1
                            cn[c_old][i][k] = new_cn
                            cn[c_old][k][i] = new_cn
                    m >>= 1
                    k += 1
                
                # Pairs involving j and neighbors of i in c_old
                mask_i_old = adj[c_old][i]
                k = 0
                m = mask_i_old
                while m:
                    if m & 1:
                        if k != j:
                            old_cn = cn[c_old][j][k]
                            new_cn = old_cn - 1
                            if old_cn >= 2 and new_cn < 2:
                                delta -= 1
                            cn[c_old][j][k] = new_cn
                            cn[c_old][k][j] = new_cn
                    m >>= 1
                    k += 1
                
                # Update adj for c_old
                adj[c_old][i] &= ~(1 << j)
                adj[c_old][j] &= ~(1 << i)
                
                # New Color Updates
                c_new = new_color
                # (i, j) pair itself
                # It is now in c_new.
                # Its cn is the number of common neighbors of i and j in c_new.
                # Note: cn[c_new][i][j] is currently 0 (or whatever it was, but (i,j) was not in c_new).
                # Wait, cn is defined for all pairs, regardless of edge existence?
                # Yes, cn[c][i][j] is the number of common neighbors of i and j in color c.
                # This value exists even if (i, j) is not an edge in color c.
                # However, the energy only counts pairs (i, j) that ARE edges in color c?
                # The prompt says: "No colour class may contain a four-cycle".
                # A C4 is a cycle in the graph of that color.
                # So we only care about pairs (i, j) that are edges in color c.
                # So energy = sum over (i, j) in E_c of [cn[c][i][j] >= 2].
                
                # So for (i, j) in c_new:
                # We need to calculate its new cn.
                # The cn of (i, j) in c_new is the number of w such that (i, w) in c_new and (j, w) in c_new.
                # This is exactly bin(adj[c_new][i] & adj[c_new][j]).count('1').
                # But we are adding (i, j) to c_new. This does NOT change the common neighbors of i and j!
                # The common neighbors are w != i, j.
                # So cn[c_new][i][j] remains the same?
                # Wait, if w = i or w = j, they are not common neighbors.
                # So adding edge (i, j) does not change cn[c_new][i][j].
                # BUT, it DOES change the cn of other pairs.
                
                # So for (i, j) in c_new:
                # Its energy contribution is 1 if cn[c_new][i][j] >= 2, else 0.
                # Note: cn[c_new][i][j] is the value BEFORE adding the edge?
                # Yes, because the edge (i, j) itself is not a common neighbor.
                # So we can just check the current cn[c_new][i][j].
                
                if cn[c_new][i][j] >= 2:
                    delta += 1
                
                # Pairs involving i and neighbors of j in c_new
                # For each k in N(c_new, j) \ {i}:
                # i is now a common neighbor of k and j?
                # No. k is neighbor of j. i is now neighbor of j.
                # So j is a common neighbor of i and k?
                # Yes. (i, j) in c_new and (k, j) in c_new.
                # So cn[c_new][i][k] increases by 1.
                
                mask_j_new = adj[c_new][j]
                k = 0
                m = mask_j_new
                while m:
                    if m & 1:
                        if k != i:
                            old_cn = cn[c_new][i][k]
                            new_cn = old_cn + 1
                            if old_cn < 2 and new_cn >= 2:
                                delta += 1
                            cn[c_new][i][k] = new_cn
                            cn[c_new][k][i] = new_cn
                    m >>= 1
                    k += 1
                
                # Pairs involving j and neighbors of i in c_new
                mask_i_new = adj[c_new][i]
                k = 0
                m = mask_i_new
                while m:
                    if m & 1:
                        if k != j:
                            old_cn = cn[c_new][j][k]
                            new_cn = old_cn + 1
                            if old_cn < 2 and new_cn >= 2:
                                delta += 1
                            cn[c_new][j][k] = new_cn
                            cn[c_new][k][j] = new_cn
                    m >>= 1
                    k += 1
                
                # Update adj for c_new
                adj[c_new][i] |= (1 << j)
                adj[c_new][j] |= (1 << i)
                
                # Apply move
                if delta < 0 or random.random() < math.exp(-delta / temperature):
                    colors[ei] = new_color
                    current_energy += delta
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_colors = colors[:]
                        if best_energy == 0:
                            return best_colors
                else:
                    # Revert changes
                    # This is complex. It's easier to just re-init or store state.
                    # Given the complexity, let's just re-init state if we reject? No, too slow.
                    # Let's implement revert.
                    
                    # Revert c_new
                    adj[c_new][i] &= ~(1 << j)
                    adj[c_new][j] &= ~(1 << i)
                    
                    mask_j_new = adj[c_new][j]
                    k = 0
                    m = mask_j_new
                    while m:
                        if m & 1:
                            if k != i:
                                cn[c_new][i][k] -= 1
                                cn[c_new][k][i] -= 1
                        m >>= 1
                        k += 1
                    
                    mask_i_new = adj[c_new][i]
                    k = 0
                    m = mask_i_new
                    while m:
                        if m & 1:
                            if k != j:
                                cn[c_new][j][k] -= 1
                                cn[c_new][k][j] -= 1
                        m >>= 1
                        k += 1
                        
                    # Revert c_old
                    adj[c_old][i] |= (1 << j)
                    adj[c_old][j] |= (1 << i)
                    
                    mask_j_old = adj[c_old][j]
                    k = 0
                    m = mask_j_old
                    while m:
                        if m & 1:
                            if k != i:
                                cn[c_old][i][k] += 1
                                cn[c_old][k][i] += 1
                        m >>= 1
                        k += 1
                        
                    mask_i_old = adj[c_old][i]
                    k = 0
                    m = mask_i_old
                    while m:
                        if m & 1:
                            if k != j:
                                cn[c_old][j][k] += 1
                                cn[c_old][k][j] += 1
                        m >>= 1
                        k += 1

            temperature *= cooling_rate
            
        return best_colors if best_energy == 0 else None

    # Try n from 26 down to 1
    for n in range(26, 0, -1):
        if time_left() < 5:
            break
        result = solve_for_n(n)
        if result is not None:
            return {'n': n, 'colors': result}
            
    # Fallback
    return {'n': 1, 'colors': [0]}

if __name__ == '__main__':
    result = construct()
    print(f"Found n = {result['n']}")
