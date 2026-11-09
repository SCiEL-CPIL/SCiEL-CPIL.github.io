import time
import random

def construct():
    deadline = time.time() + 145.0
    
    # GF(3^4) construction for 3-coloring of K_81
    # GF(81) = GF(3^4), multiplicative group has order 80
    # Cubes subgroup has order 80/gcd(3,80) = 80/1 = 80... wait
    # Actually gcd(3,80)=1, so cubes = all nonzero elements. That's not right.
    # Let me reconsider: we need a subgroup H of index 3 in GF(81)*
    # |GF(81)*| = 80. We need subgroup of order 80/3, but 3 doesn't divide 80.
    # So this approach doesn't work directly with GF(3^4).
    
    # Alternative: Use GF(27) = GF(3^3), |GF(27)*| = 26. gcd(3,26)=1, so cubes = all.
    # Hmm, this is tricky. Let me use a different approach.
    
    # Actually, for 3-coloring without monochromatic triangles, we can use:
    # Partition GF(q) into cosets of a multiplicative subgroup H where:
    # - h1 + h2 notin H for h1,h2 in H (or more precisely, no solution to x+y=z in H)
    # This requires specific field properties.
    
    # Simpler approach: Use the known 32-vertex construction and extend via local search.
    # Or use random 4-coloring + local search from scratch for higher n.
    
    # Let's use a hybrid: start with good 32-vertex seed, then try to extend.
    
    def mul_gf16(a, b):
        """Multiply in GF(16)"""
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    
    log_gf16, x = {}, 1
    for e in range(15):
        log_gf16[x] = e
        x = mul_gf16(x, 2)
    
    # Build 32-vertex seed coloring
    def make_seed_32():
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                if i // 16 == j // 16:
                    colors.append(log_gf16[(i % 16) ^ (j % 16)] % 3)
                else:
                    colors.append(3)
        return colors
    
    best_n = 32
    best_colors = make_seed_32()
    
    def get_idx(i, j, n):
        if i > j: i, j = j, i
        return i*n - i*(i+1)//2 + (j-i-1)
    
    def count_triangles(colors, n):
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                idx_ij = get_idx(i, j, n)
                c_ij = colors[idx_ij]
                for k in range(j+1, n):
                    idx_ik = get_idx(i, k, n)
                    idx_jk = get_idx(j, k, n)
                    if colors[idx_ik] == c_ij and colors[idx_jk] == c_ij:
                        count += 1
        return count
    
    def local_search_extend(base_colors, base_n, target_n, time_limit):
        """Try to extend a valid coloring from base_n to target_n vertices"""
        N = target_n*(target_n-1)//2
        edges = []
        for i in range(target_n):
            for j in range(i+1, target_n):
                edges.append((i, j))
        
        # Precompute edge neighbors for triangle counting
        edge_neighbors = []
        for idx, (i, j) in enumerate(edges):
            neighbors = []
            for w in range(target_n):
                if w == i or w == j: continue
                idx_iw = get_idx(i, w, target_n)
                idx_jw = get_idx(j, w, target_n)
                neighbors.append((idx_iw, idx_jw))
            edge_neighbors.append(neighbors)
        
        # Initialize: copy base colors, random for new edges
        colors = [0]*N
        for i in range(base_n):
            for j in range(i+1, base_n):
                idx = get_idx(i, j, target_n)
                base_idx = get_idx(i, j, base_n)
                colors[idx] = base_colors[base_idx]
        
        # Random colors for edges involving new vertices
        for i in range(target_n):
            for j in range(i+1, target_n):
                idx = get_idx(i, j, target_n)
                if i >= base_n or j >= base_n:
                    colors[idx] = random.randint(0, 3)
        
        curr_count = count_triangles(colors, target_n)
        if curr_count == 0:
            return colors
        
        start_time = time.time()
        no_improve = 0
        
        while time.time() < start_time + time_limit:
            # Pick random edge, prefer edges involving new vertices
            if random.random() < 0.7:
                # Pick edge involving at least one new vertex
                e_idx = random.randint(0, N-1)
                i, j = edges[e_idx]
                if i < base_n and j < base_n:
                    # Retry
                    e_idx = random.randint(0, N-1)
                    i, j = edges[e_idx]
                    if i < base_n and j < base_n:
                        continue
            else:
                e_idx = random.randint(0, N-1)
                i, j = edges[e_idx]
            
            old_c = colors[e_idx]
            best_new_c = -1
            best_new_count = curr_count
            
            for new_c in range(4):
                if new_c == old_c: continue
                delta = 0
                for idx_iw, idx_jw in edge_neighbors[e_idx]:
                    c_iw = colors[idx_iw]
                    c_jw = colors[idx_jw]
                    if old_c == c_iw and old_c == c_jw:
                        delta -= 1
                    if new_c == c_iw and new_c == c_jw:
                        delta += 1
                new_count = curr_count + delta
                if new_count < best_new_count:
                    best_new_count = new_count
                    best_new_c = new_c
            
            if best_new_c != -1 and best_new_count < curr_count:
                colors[e_idx] = best_new_c
                curr_count = best_new_count
                no_improve = 0
                if curr_count == 0:
                    return colors
            else:
                # Accept random move to escape local minima
                if no_improve > 500:
                    new_c = random.randint(0, 3)
                    if new_c != old_c:
                        delta = 0
                        for idx_iw, idx_jw in edge_neighbors[e_idx]:
                            c_iw = colors[idx_iw]
                            c_jw = colors[idx_jw]
                            if old_c == c_iw and old_c == c_jw:
                                delta -= 1
                            if new_c == c_iw and new_c == c_jw:
                                delta += 1
                        colors[e_idx] = new_c
                        curr_count += delta
                    no_improve = 0
                else:
                    no_improve += 1
        
        return None
    
    # Try to extend from 32 to higher n
    for target_n in range(33, 63):
        if time.time() > deadline - 10:
            break
        
        remaining = deadline - time.time()
        time_slice = min(3.0, remaining * 0.3)
        
        # Try multiple restarts
        found = False
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            result = local_search_extend(best_colors, best_n, target_n, time_slice/3)
            if result is not None:
                best_n = target_n
                best_colors = result
                found = True
                break
        
        if not found:
            # Give up on this n and higher
            break
    
    # Final verification
    if best_n > 1:
        final_count = count_triangles(best_colors, best_n)
        if final_count > 0:
            # Fallback to seed
            best_n = 32
            best_colors = make_seed_32()
    
    return {'n': best_n, 'colors': best_colors}
