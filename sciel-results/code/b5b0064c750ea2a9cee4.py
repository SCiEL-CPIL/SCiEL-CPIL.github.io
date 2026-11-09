import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    N = 256
    
    # Build conflict graph using bitsets
    # conflict[v] is an integer bitset where bit (u-1) is set if v and u conflict
    conflict = [0] * (N + 1)
    
    for a in range(1, N + 1):
        a_sq = a * a
        limit = int(math.isqrt(a_sq))
        for k in range(1, limit + 1):
            if a_sq % k == 0:
                d1 = k
                d2 = a_sq // k
                for d in (d1, d2):
                    if a < d <= a + N:
                        b = d - a
                        if 1 <= b <= N and b != a:
                            conflict[a] |= (1 << (b - 1))
                            conflict[b] |= (1 << (a - 1))
    
    # Precompute bitsets for vertices 1..N
    vertex_bit = [0] * (N + 1)
    for v in range(1, N + 1):
        vertex_bit[v] = 1 << (v - 1)
    
    # All vertices bitset
    all_vertices = (1 << N) - 1
    
    # Build compatibility graph (edges where NO conflict exists)
    # compat[v] contains bits for u such that u != v and no conflict between u and v
    compat = [0] * (N + 1)
    for v in range(1, N + 1):
        compat[v] = all_vertices ^ vertex_bit[v] ^ conflict[v]
        
    best_clique = []
    best_size = 0
    
    # Greedy initialization
    cand_init = all_vertices
    greedy_clique = []
    temp_cand = cand_init
    while temp_cand:
        best_v = 0
        max_deg = -1
        t = temp_cand
        while t:
            lsb = t & -t
            v_idx = lsb.bit_length() - 1
            v = v_idx + 1
            deg = (compat[v] & temp_cand).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            t ^= lsb
        
        if best_v == 0:
            break
            
        greedy_clique.append(best_v)
        temp_cand &= compat[best_v]
        
    best_size = len(greedy_clique)
    best_clique = greedy_clique[:]
    
    def expand_mis(cand, size, clique):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
            
        if size == 0:
            return
            
        if size > best_size:
            best_size = size
            best_clique = clique[:]
            
        if size + cand.bit_count() <= best_size:
            return
            
        if cand == 0:
            return

        verts = []
        temp_cand = cand
        while temp_cand:
            lsb = temp_cand & -temp_cand
            v_idx = lsb.bit_length() - 1
            v = v_idx + 1
            verts.append(v)
            temp_cand ^= lsb
            
        def get_degree(v):
            return (compat[v] & cand).bit_count()
            
        verts.sort(key=get_degree, reverse=True)
        
        for v in verts:
            if time.time() > deadline:
                return
                
            v_bit = vertex_bit[v]
            
            if not (cand & v_bit):
                continue
                
            new_cand = cand & compat[v]
            clique.append(v)
            expand_mis(new_cand, size + 1, clique)
            clique.pop()
            
            cand ^= v_bit
            
            if size + cand.bit_count() <= best_size:
                return

    expand_mis(all_vertices, 0, [])
    
    return best_clique
