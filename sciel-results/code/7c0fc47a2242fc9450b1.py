import time
import random
import math

def construct():
    n = 256
    ap_list = []
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(ap_list)
    
    bit_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap_list):
        for pos, bit in enumerate(ap):
            bit_aps[bit].append((idx, pos))

    def get_cost_delta(colors, i):
        delta = 0
        for ap_idx, pos in bit_aps[i]:
            ap = ap_list[ap_idx]
            c0 = colors[ap[0]]
            c1 = colors[ap[1]]
            c2 = colors[ap[2]]
            c3 = colors[ap[3]]
            
            mono_before = (c0 == c1 and c1 == c2 and c2 == c3)
            
            new_c = 1 - colors[i]
            
            if pos == 0:
                mono_after = (c1 == c2 and c2 == c3 and c3 == new_c)
            elif pos == 1:
                mono_after = (c0 == c2 and c2 == c3 and c3 == new_c)
            elif pos == 2:
                mono_after = (c0 == c1 and c1 == c3 and c3 == new_c)
            else:
                mono_after = (c0 == c1 and c1 == c2 and c2 == new_c)
                
            if mono_before and not mono_after:
                delta -= 1
            elif not mono_before and mono_after:
                delta += 1
                
        return delta

    def calculate_total_cost(colors):
        cnt = 0
        for ap in ap_list:
            if colors[ap[0]] == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]:
                cnt += 1
        return cnt

    best_score = float('inf')
    best_colors = None
    
    deadline = time.time() + 145
    
    def thue_morse(n):
        return [bin(i).count('1') % 2 for i in range(n)]
        
    def random_init(n):
        return [random.getrandbits(1) for _ in range(n)]

    while time.time() < deadline:
        if random.random() < 0.5:
            colors = thue_morse(n)
        else:
            colors = random_init(n)
            
        current_cost = calculate_total_cost(colors)
        if current_cost < best_score:
            best_score = current_cost
            best_colors = colors[:]
            
        if best_score == 0:
            break

        T = 5.0
        T_min = 0.01
        cooling_rate = 0.9995
        
        while time.time() < deadline - 2 and T > T_min:
            i = random.getrandbits(8)
            
            delta = get_cost_delta(colors, i)
            
            if delta < 0:
                colors[i] = 1 - colors[i]
                current_cost += delta
                if current_cost < best_score:
                    best_score = current_cost
                    best_colors = colors[:]
            else:
                if delta == 0:
                    colors[i] = 1 - colors[i]
                else:
                    if delta < 30 * T:
                        if random.random() < math.exp(-delta / T):
                            colors[i] = 1 - colors[i]
                            current_cost += delta
                            if current_cost < best_score:
                                best_score = current_cost
                                best_colors = colors[:]
            
            T *= cooling_rate
            
            if best_score == 0:
                break
                
    return best_colors
