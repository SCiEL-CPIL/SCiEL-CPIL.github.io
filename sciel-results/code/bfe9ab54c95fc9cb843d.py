import time
import random
import math

def construct():
    n = 256
    # Precompute APs
    ap_list = []
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Precompute other_bits: for each bit i, list of (other_idx1, other_idx2, other_idx3)
    # This allows fast delta calculation
    other_bits = [[] for _ in range(n)]
    for ap in ap_list:
        a, b, c, d = ap
        other_bits[a].append((b, c, d))
        other_bits[b].append((a, c, d))
        other_bits[c].append((a, b, d))
        other_bits[d].append((a, b, c))

    def calculate_total_cost(colors):
        cnt = 0
        for ap in ap_list:
            if colors[ap[0]] == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]:
                cnt += 1
        return cnt

    def get_cost_delta(colors, i):
        """
        Returns the change in number of mono APs if we flip bit i.
        """
        c_i = colors[i]
        new_c = 1 - c_i
        
        count_before = 0
        count_after = 0
        
        for o1, o2, o3 in other_bits[i]:
            c1 = colors[o1]
            c2 = colors[o2]
            c3 = colors[o3]
            
            if c1 == c2 == c3 == c_i:
                count_before += 1
                
            if c1 == c2 == c3 == new_c:
                count_after += 1
                
        return count_after - count_before

    best_score = float('inf')
    best_colors = None
    
    # Safety margin: 110s limit to stay under 120s
    deadline = time.time() + 110
    
    def thue_morse(n):
        return [bin(i).count('1') % 2 for i in range(n)]
        
    def random_init(n):
        return [random.randint(0, 1) for _ in range(n)]

    # Run SA multiple times
    while time.time() < deadline:
        # Start with a good init
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

        # SA Parameters
        T = 5.0
        T_min = 0.01
        cooling_rate = 0.995 
        
        start_time = time.time()
        steps = 0
        while time.time() < deadline - 5 and T > T_min:
            # Pick random bit
            i = random.randint(0, n - 1)
            
            delta = get_cost_delta(colors, i)
            
            # Acceptance probability
            if delta < 0:
                accept = True
            else:
                if delta == 0:
                    accept = True
                else:
                    prob = math.exp(-delta / T)
                    accept = random.random() < prob
            
            if accept:
                colors[i] = 1 - colors[i]
                current_cost += delta
                if current_cost < best_score:
                    best_score = current_cost
                    best_colors = colors[:]
                
            T *= cooling_rate
            steps += 1
            
            if best_score == 0:
                break
                
    return best_colors
