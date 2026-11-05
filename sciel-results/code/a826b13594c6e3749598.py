import random
import time
import numpy as np

def construct():
    n = 50
    num_colors = 4
    time_limit = 115.0
    start_time = time.time()
    
    # Precompute edge indices
    edge_to_ij = []
    idx_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            idx = len(edge_to_ij)
            edge_to_ij.append((i, j))
            idx_map[(i, j)] = idx
            idx_map[(j, i)] = idx

    num_edges = len(edge_to_ij)

    # Precompute triangles for each edge
    # triangles_by_edge[e] = list of (k, idx_ik, idx_jk)
    triangles_by_edge = [[] for _ in range(num_edges)]
    for e, (i, j) in enumerate(edge_to_ij):
        for k in range(n):
            if k == i or k == j:
                continue
            if i < k:
                idx_ik = idx_map[(i, k)]
            else:
                idx_ik = idx_map[(k, i)]
            if j < k:
                idx_jk = idx_map[(j, k)]
            else:
                idx_jk = idx_map[(k, j)]
            triangles_by_edge[e].append((k, idx_ik, idx_jk))

    def solve_n(n_target, time_budget):
        # Precompute for n_target
        # Since n is fixed at 50 in the main block, we can reuse the precomputed data above 
        # if n_target == 50. But for generality or if we change n, we'd need to recompute.
        # The task asks for max n. We try 50 first.
        
        if n_target != n:
            # Recompute structures for n_target
            # (Implementation omitted for brevity in thought process, but will be in code)
            pass
        
        mono_set = set()
        mono_list = []
        colors = [random.randint(0, num_colors - 1) for _ in range(num_edges)]
        
        # Initialize mono_set
        for e in range(num_edges):
            c_e = colors[e]
            for k, idx_ik, idx_jk in triangles_by_edge[e]:
                if colors[idx_ik] == c_e and colors[idx_jk] == c_e:
                    key = (e, k)
                    if key not in mono_set:
                        mono_set.add(key)
                        mono_list.append(key)
                        
        # Local Search
        max_iters = 5000
        for it in range(max_iters):
            if time.time() - start_time > time_budget:
                return None
            
            if not mono_list:
                return colors.copy()
            
            # Pick a random mono triangle
            idx = random.randint(0, len(mono_list) - 1)
            e, k = mono_list[idx]
            
            c_old = colors[e]
            
            # Find best new color
            best_delta = float('inf')
            best_c_new = -1
            
            for c_new in range(num_colors):
                if c_new == c_old:
                    continue
                
                delta = 0
                for k_prime, idx_ik, idx_jk in triangles_by_edge[e]:
                    c_ik = colors[idx_ik]
                    c_jk = colors[idx_jk]
                    
                    was_mono = (c_ik == c_old and c_jk == c_old)
                    is_mono = (c_ik == c_new and c_jk == c_new)
                    
                    if was_mono and not is_mono:
                        delta -= 1
                    elif not was_mono and is_mono:
                        delta += 1
                        
                if delta < best_delta:
                    best_delta = delta
                    best_c_new = c_new
            
            # If no improvement (delta > 0) or stuck, we might still apply the best move 
            # to escape local minima? 
            # Usually, we want to minimize. If all deltas are positive, we are in a local minimum 
            # with respect to single edge flips.
            # Strategy: If best_delta > 0, restart or perturb.
            # For now, if best_delta > 0, we break and restart.
            if best_c_new == -1 or best_delta > 0:
                # Perturb or Restart
                # Let's try a small perturbation first
                if random.random() < 0.5:
                    # Perturb 5 random edges
                    for _ in range(5):
                        re = random.randint(0, num_edges - 1)
                        old_c = colors[re]
                        new_c = random.randint(0, num_colors - 1)
                        if new_c != old_c:
                            # Update mono_set for re
                            for k_prime, idx_ik, idx_jk in triangles_by_edge[re]:
                                key = (re, k_prime)
                                was_mono = (colors[idx_ik] == old_c and colors[idx_jk] == old_c)
                                is_mono = (colors[idx_ik] == new_c and colors[idx_jk] == new_c)
                                if was_mono and not is_mono:
                                    if key in mono_set:
                                        mono_set.remove(key)
                                        try:
                                            mono_list.remove(key)
                                        except ValueError:
                                            pass
                                elif not was_mono and is_mono:
                                    if key not in mono_set:
                                        mono_set.add(key)
                                        mono_list.append(key)
                            colors[re] = new_c
                else:
                    break # Restart
            else:
                # Apply best move
                c_new = best_c_new
                # Update mono_set
                for k_prime, idx_ik, idx_jk in triangles_by_edge[e]:
                    key = (e, k_prime)
                    was_mono = (colors[idx_ik] == c_old and colors[idx_jk] == c_old)
                    is_mono = (colors[idx_ik] == c_new and colors[idx_jk] == c_new)
                    
                    if was_mono and not is_mono:
                        if key in mono_set:
                            mono_set.remove(key)
                            try:
                                mono_list.remove(key)
                            except ValueError:
                                pass
                    elif not was_mono and is_mono:
                        if key not in mono_set:
                            mono_set.add(key)
                            mono_list.append(key)
                
                colors[e] = c_new
                
        return None

    # Try to solve for n=50
    result = solve_n(50, time_limit)
    if result:
        return {'n': 50, 'colors': result}
        
    # Fallback to n=32 construction
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
    colors_32 = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors_32.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': colors_32}
