import time
import random
import sys

def construct():
    deadline = time.time() + 140.0  # Safety margin for 150s limit
    
    def verify(n, colors):
        """Fast verification using bitmasks"""
        m = n * (n - 1) // 2
        if len(colors) != m:
            return False
        masks = [[0] * 4 for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                masks[i][c] |= (1 << j)
                masks[j][c] |= (1 << i)
                idx += 1
        for u in range(n):
            for c in range(4):
                mask_u = masks[u][c]
                if mask_u == 0:
                    continue
                temp = mask_u
                while temp:
                    lsb = temp & (-temp)
                    v = lsb.bit_length() - 1
                    if masks[v][c] & mask_u:
                        return False
                    temp ^= lsb
        return True
    
    def count_triangles(n, colors):
        """Count monochromatic triangles using bitmasks"""
        m = n * (n - 1) // 2
        masks = [[0] * 4 for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                masks[i][c] |= (1 << j)
                masks[j][c] |= (1 << i)
                idx += 1
        tri_count = 0
        for u in range(n):
            for c in range(4):
                mask_u = masks[u][c]
                if mask_u == 0:
                    continue
                temp = mask_u
                while temp:
                    lsb = temp & (-temp)
                    v = lsb.bit_length() - 1
                    tri_count += (masks[v][c] & mask_u).bit_count()
                    temp ^= lsb
        return tri_count // 3
    
    def paley_coloring_25():
        """
        Paley graph coloring of K_25 using quadratic residues mod 25.
        In Z_25, the quadratic residues are {1,4,9,16,6,24,21,10,11,22,19,14,5,20,15}
        Wait, let me compute properly. For p=25 (not prime), we use a different approach.
        
        Actually, for a triangle-free 3-coloring of K_25, we can use the following:
        Color edge (i,j) based on (j-i) mod 25.
        
        The Paley graph of order 25 (which is 5^2) has vertices in F_25.
        The quadratic residues in F_25 form a triangle-free graph when used as adjacency.
        
        Let's use a simpler approach: the complete graph K_25 can be 3-colored
        triangle-free using the following construction based on Z_5 x Z_5.
        
        Actually, let me use a known construction: the 25-vertex 3-coloring
        from the Paley graph of order 25.
        
        In F_25 = GF(25), the quadratic residues give a self-complementary graph.
        The Paley graph P(25) has 25 vertices and is triangle-free in each color class
        when we use 3 colors based on cubic residues or similar.
        
        For simplicity, let's use a different known construction:
        The 25-vertex 3-coloring can be derived from the 5x5 grid.
        
        Color (i,j) where i,j in {0,1,2,3,4}:
        - If (j-i) mod 5 is in {1,2}: color 0
        - If (j-i) mod 5 is in {3,4}: color 1  
        - If i==j (same row) or special: color 2
        
        This is getting complex. Let me use a simpler fallback.
        """
        # Use a known valid 3-coloring of K_25
        # Based on the Paley construction for order 25
        # Vertices are pairs (a,b) in Z_5 x Z_5
        # Edge color between (a1,b1) and (a2,b2) is determined by:
        # d = (a2-a1, b2-b1) mod 5
        # If d is a "quadratic residue" direction: color 0
        # If d is a "non-residue" direction: color 1
        # Special handling for color 2
        
        # Actually, let me just use a computational approach:
        # Generate a random 3-coloring and check, or use a known one.
        
        # For now, return a placeholder that will be replaced by local search
        # We'll construct it properly below
        pass
    
    def build_paley_25():
        """
        Construct a triangle-free 3-coloring of K_25.
        Using the construction from the Paley graph of order 25.
        
        F_25 can be represented as a + b*sqrt(2) where a,b in Z_5, sqrt(2)^2 = 2.
        The quadratic residues in F_25 are elements that are squares.
        
        For a triangle-free 3-coloring, we partition the 12 non-zero elements
        of F_25 into 3 sets of 4 such that no set contains x, y, z with x+y=z.
        
        Actually, the standard approach: In F_25, the norm map N(x) = x * x^5 gives values in F_5.
        The elements with N(x) = 1 form a set of 12 elements, N(x) = 2, 3, 4 form sets of 5 each? No...
        
        Let me use a different, verified construction.
        """
        # Use the following known 3-coloring of K_25:
        # Label vertices 0..24.
        # Color edge (i,j) based on (j-i) mod 25.
        # The 24 non-zero differences are partitioned into 3 sets of 8.
        # Each set must be sum-free (no two elements sum to a third mod 25).
        
        # One valid partition of {1,...,24} into 3 sum-free sets of 8:
        # Set A: {1, 2, 4, 8, 16, 3, 6, 12} - powers of 2 mod 25? 
        # 1,2,4,8,16,32=7,14,28=3... not quite.
        
        # Let me just compute a valid one by search or use a known result.
        # For the purpose of this program, I'll use a local search approach
        # to find a 3-coloring of K_25, then extend it.
        
        # Actually, for the disjoint union approach to work, we need:
        # - Two copies of a 3-coloring of K_k (each color class triangle-free)
        # - All inter-copy edges in color 3
        # - No monochromatic triangles: within each copy, colors 0,1,2 are triangle-free by construction.
        #   Inter-copy: all edges are color 3, so a monochromatic triangle in color 3 would need
        #   3 vertices all in the same copy (impossible, only 2 copies) or 2 in one copy and 1 in the other.
        #   But if 2 vertices are in copy A and 1 in copy B, the edge between the two in A is NOT color 3
        #   (it's color 0, 1, or 2), so no monochromatic color-3 triangle.
        #   Similarly for color 0,1,2: a triangle with vertices from both copies has at least one
        #   inter-copy edge which is color 3, so it can't be monochromatic in 0,1,2.
        #   Within a single copy, the 3-coloring is triangle-free by construction.
        # So the disjoint union of two copies of a triangle-free 3-coloring of K_k,
        # with inter-copy edges in color 3, gives a triangle-free 4-coloring of K_{2k}.
        
        # We need a triangle-free 3-coloring of K_k for some k.
        # The maximum k for which this is known is... 
        # R(3,3,3) = 17, so a 3-coloring of K_16 is triangle-free in each color.
        # So we can get K_32 this way, which matches our current fallback.
        
        # For larger n, we need a better construction.
        # Chung's 50-vertex coloring uses a more sophisticated approach.
        
        # Let's try a hybrid: use the disjoint union for a base, then local search to extend.
        
        # First, let's build a valid 3-coloring of K_16 (known to exist since R(3,3,3)=17).
        # We can find it by local search or use a known construction.
        
        return None  # Will be handled by local search
    
    def verify_3color(n, colors):
        """Verify a 3-coloring has no monochromatic triangles"""
        m = n * (n - 1) // 2
        if len(colors) != m:
            return False
        masks = [[0] * 3 for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                if c >= 3:
                    return False
                masks[i][c] |= (1 << j)
                masks[j][c] |= (1 << i)
                idx += 1
        for u in range(n):
            for c in range(3):
                mask_u = masks[u][c]
                if mask_u == 0:
                    continue
                temp = mask_u
                while temp:
                    lsb = temp & (-temp)
                    v = lsb.bit_length() - 1
                    if masks[v][c] & mask_u:
                        return False
                    temp ^= lsb
        return True
    
    def local_search_3color(n, time_budget):
        """Find a triangle-free 3-coloring of K_n using local search"""
        start_time = time.time()
        m = n * (n - 1) // 2
        
        idx = [[0] * n for _ in range(n)]
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                idx[i][j] = k
                idx[j][i] = k
                k += 1
        
        for restart in range(50):
            if time.time() > start_time + time_budget:
                break
            
            colors = [random.randint(0, 2) for _ in range(m)]
            
            masks = [[0] * 3 for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx[i][j]]
                    masks[i][c] |= (1 << j)
                    masks[j][c] |= (1 << i)
            
            tri_count = 0
            for u in range(n):
                for c in range(3):
                    mask_u = masks[u][c]
                    if mask_u == 0:
                        continue
                    temp = mask_u
                    while temp:
                        lsb = temp & (-temp)
                        v = lsb.bit_length() - 1
                        tri_count += (masks[v][c] & mask_u).bit_count()
                        temp ^= lsb
            tri_count //= 3
            
            if tri_count == 0:
                return colors
            
            T = 1.0
            cooling = 0.999
            min_T = 0.001
            no_improve = 0
            best_tri = tri_count
            
            while T > min_T and time.time() < start_time + time_budget - 1:
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                
                i_idx = idx[u][v]
                old_c = colors[i_idx]
                new_c = random.randint(0, 2)
                if old_c == new_c:
                    continue
                
                lost = (masks[u][old_c] & masks[v][old_c]).bit_count()
                gained = (masks[u][new_c] & masks[v][new_c]).bit_count()
                delta = gained - lost
                
                if delta < 0 or random.random() < 2.718 ** (-delta / T):
                    colors[i_idx] = new_c
                    tri_count += delta
                    masks[u][old_c] &= ~(1 << v)
                    masks[v][old_c] &= ~(1 << u)
                    masks[u][new_c] |= (1 << v)
                    masks[v][new_c] |= (1 << u)
                    
                    if tri_count < best_tri:
                        best_tri = tri_count
                        no_improve = 0
                    else:
                        no_improve += 1
                    
                    if tri_count == 0:
                        return colors
                    
                    if no_improve > 30000:
                        break
                
                T *= cooling
            
            if best_tri <= 3 and time.time() < start_time + time_budget - 1:
                for _ in range(50000):
                    if time.time() > start_time + time_budget - 1:
                        break
                    found = False
                    for u in range(n):
                        for c in range(3):
                            mask_u = masks[u][c]
                            if mask_u == 0:
                                continue
                            temp = mask_u
                            while temp:
                                lsb = temp & (-temp)
                                v = lsb.bit_length() - 1
                                common = masks[v][c] & mask_u
                                if common:
                                    lsb2 = common & (-common)
                                    w = lsb2.bit_length() - 1
                                    for eu, ev in [(u, v), (u, w), (v, w)]:
                                        if eu > ev:
                                            eu, ev = ev, eu
                                        e_idx = idx[eu][ev]
                                        old_ec = colors[e_idx]
                                        for new_ec in range(3):
                                            if new_ec == old_ec:
                                                continue
                                            lost_e = (masks[eu][old_ec] & masks[ev][old_ec]).bit_count()
                                            gained_e = (masks[eu][new_ec] & masks[ev][new_ec]).bit_count()
                                            delta_e = gained_e - lost_e
                                            if delta_e < 0:
                                                colors[e_idx] = new_ec
                                                tri_count += delta_e
                                                masks[eu][old_ec] &= ~(1 << ev)
                                                masks[ev][old_ec] &= ~(1 << eu)
                                                masks[eu][new_ec] |= (1 << ev)
                                                masks[ev][new_ec] |= (1 << eu)
                                                if tri_count == 0:
                                                    return colors
                                                found = True
                                                break
                                        if found:
                                            break
                                    if found:
                                        break
                                if found:
                                    break
                                temp ^= lsb
                            if found:
                                break
                        if found:
                            break
                    if not found:
                        break
        
        return None
    
    def local_search_4color(n, time_budget, initial_colors=None):
        """Find a triangle-free 4-coloring of K_n using local search"""
        start_time = time.time()
        m = n * (n - 1) // 2
        
        idx = [[0] * n for _ in range(n)]
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                idx[i][j] = k
                idx[j][i] = k
                k += 1
        
        for restart in range(30):
            if time.time() > start_time + time_budget:
                break
            
            if restart == 0 and initial_colors is not None:
                colors = list(initial_colors)
            else:
                colors = [random.randint(0, 3) for _ in range(m)]
            
            masks = [[0] * 4 for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx[i][j]]
                    masks[i][c] |= (1 << j)
                    masks[j][c] |= (1 << i)
            
            tri_count = 0
            for u in range(n):
                for c in range(4):
                    mask_u = masks[u][c]
                    if mask_u == 0:
                        continue
                    temp = mask_u
                    while temp:
                        lsb = temp & (-temp)
                        v = lsb.bit_length() - 1
                        tri_count += (masks[v][c] & mask_u).bit_count()
                        temp ^= lsb
            tri_count //= 3
            
            if tri_count == 0:
                return colors
            
            T = 1.0
            cooling = 0.9995
            min_T = 0.005
            no_improve = 0
            best_tri = tri_count
            
            while T > min_T and time.time() < start_time + time_budget - 2:
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                
                i_idx = idx[u][v]
                old_c = colors[i_idx]
                new_c = random.randint(0, 3)
                if old_c == new_c:
                    continue
                
                lost = (masks[u][old_c] & masks[v][old_c]).bit_count()
                gained = (masks[u][new_c] & masks[v][new_c]).bit_count()
                delta = gained - lost
                
                if delta < 0 or random.random() < 2.718 ** (-delta / T):
                    colors[i_idx] = new_c
                    tri_count += delta
                    masks[u][old_c] &= ~(1 << v)
                    masks[v][old_c] &= ~(1 << u)
                    masks[u][new_c] |= (1 << v)
                    masks[v][new_c] |= (1 << u)
                    
                    if tri_count < best_tri:
                        best_tri = tri_count
                        no_improve = 0
                    else:
                        no_improve += 1
                    
                    if tri_count == 0:
                        return colors
                    
                    if no_improve > 50000:
                        break
                
                T *= cooling
            
            if best_tri <= 5 and time.time() < start_time + time_budget - 2:
                for _ in range(100000):
                    if time.time() > start_time + time_budget - 2:
                        break
                    found = False
                    for u in range(n):
                        for c in range(4):
                            mask_u = masks[u][c]
                            if mask_u == 0:
                                continue
                            temp = mask_u
                            while temp:
                                lsb = temp & (-temp)
                                v = lsb.bit_length() - 1
                                common = masks[v][c] & mask_u
                                if common:
                                    lsb2 = common & (-common)
                                    w = lsb2.bit_length() - 1
                                    for eu, ev in [(u, v), (u, w), (v, w)]:
                                        if eu > ev:
                                            eu, ev = ev, eu
                                        e_idx = idx[eu][ev]
                                        old_ec = colors[e_idx]
                                        for new_ec in range(4):
                                            if new_ec == old_ec:
                                                continue
                                            lost_e = (masks[eu][old_ec] & masks[ev][old_ec]).bit_count()
                                            gained_e = (masks[eu][new_ec] & masks[ev][new_ec]).bit_count()
                                            delta_e = gained_e - lost_e
                                            if delta_e < 0:
                                                colors[e_idx] = new_ec
                                                tri_count += delta_e
                                                masks[eu][old_ec] &= ~(1 << ev)
                                                masks[ev][old_ec] &= ~(1 << eu)
                                                masks[eu][new_ec] |= (1 << ev)
                                                masks[ev][new_ec] |= (1 << eu)
                                                if tri_count == 0:
                                                    return colors
                                                found = True
                                                break
                                        if found:
                                            break
                                    if found:
                                        break
                                if found:
                                    break
                                temp ^= lsb
                            if found:
                                break
                        if found:
                            break
                    if not found:
                        break
        
        return None
    
    # Strategy 1: Try to find a 3-coloring of K_16, then use disjoint union for K_32
    # Strategy 2: Try direct 4-coloring local search for n=40, 45, 50
    # Strategy 3: Use the known n=32 fallback as a base and try to extend
    
    # First, try to find a 3-coloring of K_16
    t3 = time.time()
    coloring_16 = local_search_3color(16, 30)
    t3_elapsed = time.time() - t3
    
    if coloring_16 is not None and verify_3color(16, coloring_16):
        # Build K_32 via disjoint union
        colors_32 = []
        for i in range(32):
            for j in range(i + 1, 32):
                if i // 16 == j // 16:
                    # Same copy
                    local_i = i % 16
                    local_j = j % 16
                    # Find the edge index in the 16-vertex coloring
                    # Edges are in lex order: (0,1),(0,2),...,(0,15),(1,2),...
                    eidx = 0
                    for a in range(16):
                        for b in range(a + 1, 16):
                            if a == local_i and b == local_j:
                                break
                            eidx += 1
                    colors_32.append(coloring_16[eidx])
                else:
                    # Different copies
                    colors_32.append(3)
        
        if verify(32, colors_32):
            # Try to extend to larger n
            pass
    
    # Try direct 4-coloring search for larger n
    # Allocate remaining time
    remaining = deadline - time.time()
    
    # Try n from 50 down to 33
    n_targets = [50, 48, 45, 42, 40, 38, 36, 34, 33]
    # Weight by difficulty
    total_weight = sum(n * n for n in n_targets)
    
    best_result = None
    for n in n_targets:
        if time.time() > deadline - 10:
            break
        t_budget = min(remaining * (n * n) / total_weight * 0.7, deadline - time.time() - 10)
        if t_budget < 3:
            break
        
        result = local_search_4color(n, t_budget)
        if result is not None and verify(n, result):
            return {'n': n, 'colors': result}
    
    # Fallback: use the known n=32 solution
    # Greenwood-Gleason doubled construction
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
    fallback_colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            fallback_colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    
    return {'n': 32, 'colors': fallback_colors}
