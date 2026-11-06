import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 128
    
    aps = []
    ap_indices = [[] for _ in range(N)]
    for d in range(1, N // 3 + 1):
        for i in range(N - 3 * d):
            ap = (i, i + d, i + 2 * d, i + 3 * d)
            ap_idx = len(aps)
            aps.append(ap)
            for idx in ap:
                ap_indices[idx].append(ap_idx)
                
    num_aps = len(aps)
    aps_np = np.array(aps, dtype=np.int32)
    
    def run_seed():
        colors=[]
        for x in range(1,129):
            for color in range(128):
                if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                    and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                    colors.append(color);break
        return colors

    best_colors = run_seed()
    best_k = len(set(best_colors))
    
    def build_viol_table(k):
        table = np.zeros((k, k, k, k), dtype=bool)
        for c0 in range(k):
            for c1 in range(k):
                for c2 in range(k):
                    for c3 in range(k):
                        if c0 == c1 and c1 == c2 and c2 == c3:
                            table[c0, c1, c2, c3] = True
                        else:
                            if c0 == c1 and c2 == c3 and c0 != c2: table[c0, c1, c2, c3] = True
                            if c0 == c2 and c1 == c3 and c0 != c1: table[c0, c1, c2, c3] = True
                            if c0 == c3 and c1 == c2 and c0 != c1: table[c0, c1, c2, c3] = True
                            if c1 == c2 and c2 == c3 and c0 != c1: table[c0, c1, c2, c3] = True
                            if c0 == c2 and c2 == c3 and c1 != c0: table[c0, c1, c2, c3] = True
                            if c0 == c1 and c1 == c3 and c2 != c0: table[c0, c1, c2, c3] = True
                            if c0 == c1 and c1 == c2 and c3 != c0: table[c0, c1, c2, c3] = True
        return table

    for k in range(3, best_k):
        if time.time() > deadline - 5:
            break
            
        time_limit = min(deadline - 2, time.time() + 8.0)
        viol_table = build_viol_table(k)
        
        # Start from random or seed mapped to k colors?
        # Random is better for SA.
        colors = [random.randint(0, k-1) for _ in range(N)]
        
        v_flags = np.zeros(num_aps, dtype=bool)
        for ai in range(num_aps):
            ap = aps_np[ai]
            v_flags[ai] = viol_table[colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]]
        num_violations = int(np.sum(v_flags))
        
        best_v = num_violations
        best_c = colors[:]
        
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        while T > T_min and time.time() < time_limit:
            i = random.randint(0, N-1)
            old_c = colors[i]
            new_c = random.randint(0, k-1)
            if new_c == old_c:
                continue
                
            delta = 0
            old_flags = []
            for ai in ap_indices[i]:
                old_flag = v_flags[ai]
                old_flags.append((ai, old_flag))
                
            colors[i] = new_c
            
            for ai, old_flag in old_flags:
                ap = aps_np[ai]
                new_flag = viol_table[colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]]
                v_flags[ai] = new_flag
                if new_flag and not old_flag:
                    delta += 1
                elif not new_flag and old_flag:
                    delta -= 1
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                num_violations += delta
                if num_violations < best_v:
                    best_v = num_violations
                    best_c = colors[:]
            else:
                colors[i] = old_c
                for ai, old_flag in old_flags:
                    v_flags[ai] = old_flag
            
            T *= cooling
        
        if best_v == 0:
            used_colors = set(best_c)
            if len(used_colors) < k:
                 mapping = {c: i for i, c in enumerate(sorted(used_colors))}
                 compressed = [mapping[c] for c in best_c]
                 best_c = compressed
            best_colors = best_c
            best_k = len(set(best_c))
            
            if best_k > 3:
                cols = list(set(best_c))
                for i_c in range(len(cols)):
                    for j_c in range(i_c+1, len(cols)):
                        c1, c2 = cols[i_c], cols[j_c]
                        test_c = [c1 if c == c2 else c for c in best_c]
                        valid = True
                        for ai in range(num_aps):
                            ap = aps_np[ai]
                            c0, c1v, c2v, c3v = test_c[ap[0]], test_c[ap[1]], test_c[ap[2]], test_c[ap[3]]
                            if len({c0, c1v, c2v, c3v}) < 3:
                                valid = False
                                break
                        if valid:
                            best_c = test_c
                            best_k = len(set(best_c))
                            if best_k == 3:
                                break
                    if best_k == 3:
                        break
            
            if best_k == 3:
                break

    return best_colors
