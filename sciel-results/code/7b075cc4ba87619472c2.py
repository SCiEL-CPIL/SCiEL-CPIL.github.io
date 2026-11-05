import time
import random
import sys

def construct():
    deadline = time.time() + 145.0  # Safety margin for 150s limit
    
    # Known valid n=32 solution (Greenwood-Gleason doubled)
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
    
    def verify(n, colors):
        """Fast verification using bitmasks"""
        m = n * (n - 1) // 2
        if len(colors) != m:
            return False
        # Build masks
        masks = [[0] * 4 for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                masks[i][c] |= (1 << j)
                masks[j][c] |= (1 << i)
                idx += 1
        # Check for monochromatic triangles
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
    
    def solve_for_n(n, time_budget):
        """Simulated annealing with bitmask delta computation and structured initialization"""
        start_time = time.time()
        m = n * (n - 1) // 2
        
        # Precompute edge index lookup: idx[i][j] = index in flat array
        idx = [[0] * n for _ in range(n)]
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                idx[i][j] = k
                idx[j][i] = k
                k += 1
        
        # Multiple restarts with different strategies
        best_result = None
        best_tri = float('inf')
        
        for restart in range(30):
            if time.time() > start_time + time_budget - 2:
                break
            
            # Different initialization strategies
            if restart < 5:
                # Strategy 1: Random
                colors = [random.randint(0, 3) for _ in range(m)]
            elif restart < 10:
                # Strategy 2: Based on n=32 pattern extended
                colors = [0] * m
                idx2 = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        if i < 32 and j < 32:
                            colors[idx2] = fallback_colors[idx2]
                        else:
                            colors[idx2] = random.randint(0, 3)
                        idx2 += 1
            elif restart < 15:
                # Strategy 3: Cyclic pattern
                colors = [0] * m
                idx2 = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        colors[idx2] = (i + j) % 4
                        idx2 += 1
            elif restart < 20:
                # Strategy 4: Two-partition pattern
                colors = [0] * m
                idx2 = 0
                mid = n // 2
                for i in range(n):
                    for j in range(i + 1, n):
                        if i < mid and j < mid:
                            colors[idx2] = (i - j) % 3
                        elif i >= mid and j >= mid:
                            colors[idx2] = (i - j) % 3
                        else:
                            colors[idx2] = 3
                        idx2 += 1
            else:
                # Strategy 5: Random with bias
                colors = [random.randint(0, 3) for _ in range(m)]
            
            # Build masks
            masks = [[0] * 4 for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx[i][j]]
                    masks[i][c] |= (1 << j)
                    masks[j][c] |= (1 << i)
            
            # Calculate initial triangle count using bitmasks
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
            
            # Simulated annealing with adaptive temperature
            T = 1.0
            cooling = 0.9997
            min_T = 0.005
            no_improve_count = 0
            best_tri_local = tri_count
            steps = 0
            max_steps = 500000
            
            while T > min_T and time.time() < start_time + time_budget - 2 and steps < max_steps:
                # Pick random edge
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
                
                # Calculate delta in triangle count
                lost = (masks[u][old_c] & masks[v][old_c]).bit_count()
                gained = (masks[u][new_c] & masks[v][new_c]).bit_count()
                delta = gained - lost
                
                if delta < 0 or random.random() < pow(2.718, -delta / T):
                    # Accept
                    colors[i_idx] = new_c
                    tri_count += delta
                    
                    # Update masks
                    masks[u][old_c] &= ~(1 << v)
                    masks[v][old_c] &= ~(1 << u)
                    masks[u][new_c] |= (1 << v)
                    masks[v][new_c] |= (1 << u)
                    
                    if tri_count < best_tri_local:
                        best_tri_local = tri_count
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    
                    if tri_count == 0:
                        return colors
                    
                    # Early termination if stuck too long
                    if no_improve_count > 30000:
                        break
                
                T *= cooling
                steps += 1
            
            # If we got close, do a final polish
            if best_tri_local <= 10 and time.time() < start_time + time_budget - 2:
                # Try to eliminate remaining triangles
                polish_steps = 0
                max_polish = 200000
                while polish_steps < max_polish and time.time() < start_time + time_budget - 2:
                    # Find a triangle
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
                                    # Found triangle (u, v, w) where w in common
                                    lsb2 = common & (-common)
                                    w = lsb2.bit_length() - 1
                                    # Try to break it by changing one edge
                                    for edge_u, edge_v in [(u, v), (u, w), (v, w)]:
                                        if edge_u > edge_v:
                                            edge_u, edge_v = edge_v, edge_u
                                        e_idx = idx[edge_u][edge_v]
                                        old_ec = colors[e_idx]
                                        for new_ec in range(4):
                                            if new_ec == old_ec:
                                                continue
                                            lost_e = (masks[edge_u][old_ec] & masks[edge_v][old_ec]).bit_count()
                                            gained_e = (masks[edge_u][new_ec] & masks[edge_v][new_ec]).bit_count()
                                            delta_e = gained_e - lost_e
                                            if delta_e < 0:
                                                # Accept this flip
                                                colors[e_idx] = new_ec
                                                tri_count += delta_e
                                                masks[edge_u][old_ec] &= ~(1 << edge_v)
                                                masks[edge_v][old_ec] &= ~(1 << edge_u)
                                                masks[edge_u][new_ec] |= (1 << edge_v)
                                                masks[edge_v][new_ec] |= (1 << edge_u)
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
                    polish_steps += 1
            
            if best_tri_local < best_tri:
                best_tri = best_tri_local
                best_result = colors[:]
        
        return best_result
    
    # Try n from 50 down to 33
    # Time budget allocation: give more time to higher n
    total_budget = deadline - time.time()
    n_values = list(range(50, 32, -1))
    # Allocate time: higher n gets more time, but cap it
    # n=50: 30%, n=49: 20%, n=48: 15%, n=47: 10%, n=46: 8%, n=45: 5%, n=44: 4%, n=43: 3%, n=42: 2%, n=41: 2%, n=40: 1%, n=39: 1%, n=38: 1%, n=37: 1%, n=36: 1%, n=35: 1%, n=34: 1%, n=33: 1%
    weights = {50: 30, 49: 20, 48: 15, 47: 10, 46: 8, 45: 5, 44: 4, 43: 3, 42: 2, 41: 2, 40: 1, 39: 1, 38: 1, 37: 1, 36: 1, 35: 1, 34: 1, 33: 1}
    total_weight = sum(weights.values())
    
    for n in n_values:
        if time.time() > deadline - 5:
            break
        w = weights.get(n, 1)
        t_budget = total_budget * w / total_weight * 0.85  # 85% for search, 15% buffer
        t_budget = min(t_budget, deadline - time.time() - 5)
        if t_budget < 2:
            break
        result = solve_for_n(n, t_budget)
        if result is not None:
            # Verify
            if verify(n, result):
                return {'n': n, 'colors': result}
    
    # Fallback to n=32
    return {'n': 32, 'colors': fallback_colors}
