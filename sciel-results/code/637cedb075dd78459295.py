import numpy as np
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def generate_edge_indices(n):
        M = n * (n-1) * (n-2) // 6
        edge_idx = {}
        idx = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    edge_idx[(a,b,c)] = idx
                    idx += 1
        return M, edge_idx

    def precompute_structures(n, edge_idx):
        quads_list = []
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        e1 = edge_idx[(a,b,c)]
                        e2 = edge_idx[(a,b,d)]
                        e3 = edge_idx[(a,c,d)]
                        e4 = edge_idx[(b,c,d)]
                        quads_list.append([e1, e2, e3, e4])
        
        pents_list = []
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        for e in range(d+1, n):
                            e1 = edge_idx[(a,b,c)]
                            e2 = edge_idx[(a,b,d)]
                            e3 = edge_idx[(a,b,e)]
                            e4 = edge_idx[(a,c,d)]
                            e5 = edge_idx[(a,c,e)]
                            e6 = edge_idx[(a,d,e)]
                            e7 = edge_idx[(b,c,d)]
                            e8 = edge_idx[(b,c,e)]
                            e9 = edge_idx[(b,d,e)]
                            e10 = edge_idx[(c,d,e)]
                            pents_list.append([e1, e2, e3, e4, e5, e6, e7, e8, e9, e10])
        
        quads = np.array(quads_list, dtype=np.int32)
        pents = np.array(pents_list, dtype=np.int32)
        
        M = len(edge_idx)
        # Build reverse maps as lists of arrays for vectorized updates
        e2q = [np.array([], dtype=np.int32) for _ in range(M)]
        for i in range(len(quads)):
            for e in quads[i]:
                e2q[e] = np.append(e2q[e], i)
                
        e2p = [np.array([], dtype=np.int32) for _ in range(M)]
        for i in range(len(pents)):
            for e in pents[i]:
                e2p[e] = np.append(e2p[e], i)
                
        return quads, pents, e2q, e2p

    def solve_n(n):
        if time.time() > deadline:
            return None
            
        M, edge_idx = generate_edge_indices(n)
        quads, pents, e2q, e2p = precompute_structures(n, edge_idx)
        
        colors = np.random.randint(0, 2, size=M, dtype=np.uint8)
        
        quad_red_counts = np.array([np.sum(colors[q] == 0) for q in quads], dtype=np.int32)
        pent_blue_counts = np.array([np.sum(colors[p] == 1) for p in pents], dtype=np.int32)
        
        red_violations = np.sum(quad_red_counts == 4)
        blue_violations = np.sum(pent_blue_counts == 10)
        total_violations = red_violations + blue_violations
        
        best_total = total_violations
        best_colors = colors.copy()
        
        if total_violations == 0:
            return n, best_colors.tolist()
            
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        while time.time() < deadline - 5.0:
            if T < min_T:
                colors = best_colors.copy()
                flip_mask = np.random.random(M) < 0.1
                colors[flip_mask] = 1 - colors[flip_mask]
                quad_red_counts = np.array([np.sum(colors[q] == 0) for q in quads], dtype=np.int32)
                pent_blue_counts = np.array([np.sum(colors[p] == 1) for p in pents], dtype=np.int32)
                red_violations = np.sum(quad_red_counts == 4)
                blue_violations = np.sum(pent_blue_counts == 10)
                total_violations = red_violations + blue_violations
                T = 1.0
                continue

            k = np.random.randint(0, M)
            old_color = colors[k]
            
            # Vectorized delta calculation
            # Update quad counts
            q_indices = e2q[k]
            if len(q_indices) > 0:
                if old_color == 0:
                    quad_red_counts[q_indices] -= 1
                else:
                    quad_red_counts[q_indices] += 1
                    
            p_indices = e2p[k]
            if len(p_indices) > 0:
                if old_color == 1:
                    pent_blue_counts[p_indices] -= 1
                else:
                    pent_blue_counts[p_indices] += 1
            
            new_red_violations = np.sum(quad_red_counts == 4)
            new_blue_violations = np.sum(pent_blue_counts == 10)
            new_total = new_red_violations + new_blue_violations
            
            delta = new_total - total_violations
            
            if delta < 0 or (T > 0 and random.random() < np.exp(-delta / T)):
                colors[k] = 1 - old_color
                total_violations = new_total
                red_violations = new_red_violations
                blue_violations = new_blue_violations
                
                if total_violations < best_total:
                    best_total = total_violations
                    best_colors = colors.copy()
                    if best_total == 0:
                        return n, best_colors.tolist()
            else:
                # Revert counts
                if len(q_indices) > 0:
                    if old_color == 0:
                        quad_red_counts[q_indices] += 1
                    else:
                        quad_red_counts[q_indices] -= 1
                if len(p_indices) > 0:
                    if old_color == 1:
                        pent_blue_counts[p_indices] += 1
                    else:
                        pent_blue_counts[p_indices] -= 1
                # total_violations remains same
                
            T *= cooling
            
        return None

    best_result = None
    for n in range(34, 19, -1):
        if time.time() > deadline - 10.0:
            break
        res = solve_n(n)
        if res:
            best_result = res
            break
    
    if best_result:
        n, colors = best_result
        return {'n': n, 'colors': colors}
    
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n), 3)]}
