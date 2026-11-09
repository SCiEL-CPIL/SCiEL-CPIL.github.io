import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = 128
    aps = []
    for d in range(1, (n - 1) // 3 + 1):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    def calc_cost(colors):
        bad_count = 0
        for ap in aps:
            c0 = colors[ap[0]]
            c1 = colors[ap[1]]
            c2 = colors[ap[2]]
            c3 = colors[ap[3]]
            
            # Check if <= 2 distinct colors
            # Case 1: All same
            if c0 == c1 == c2 == c3:
                bad_count += 1
            else:
                # Case 2: Exactly 2 colors
                # This happens if the multiset has counts (3,1) or (2,2)
                # (3,1): three are same, one is different
                if c0 == c1 == c2 or c0 == c1 == c3 or c0 == c2 == c3 or c1 == c2 == c3:
                    bad_count += 1
                # (2,2): two pairs
                elif (c0 == c1 and c2 == c3) or (c0 == c2 and c1 == c3) or (c0 == c3 and c1 == c2):
                    bad_count += 1
        return bad_count

    best_global_cost = float('inf')
    best_global_colors = None

    for k in range(3, 8):
        if time.time() > deadline:
            break
            
        # Simulated Annealing
        colors = [random.randint(0, k - 1) for _ in range(n)]
        current_cost = calc_cost(colors)
        
        best_local_cost = current_cost
        best_local_colors = colors[:]
        
        if current_cost == 0:
            return colors
            
        T = 1.0
        T_min = 0.001
        cooling = 0.999
        
        steps = 0
        while time.time() < deadline:
            i = random.randint(0, n - 1)
            old_color = colors[i]
            new_color = random.randint(0, k - 1)
            
            if new_color == old_color:
                continue
                
            colors[i] = new_color
            new_cost = calc_cost(colors)
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                current_cost = new_cost
                if current_cost < best_local_cost:
                    best_local_cost = current_cost
                    best_local_colors = colors[:]
                    if best_local_cost == 0:
                        return best_local_colors
            else:
                colors[i] = old_color
                
            T *= cooling
            if T < T_min:
                T = T_min
                if random.random() < 0.05:
                    colors = [random.randint(0, k - 1) for _ in range(n)]
                    current_cost = calc_cost(colors)
                    if current_cost < best_local_cost:
                        best_local_cost = current_cost
                        best_local_colors = colors[:]
                        if best_local_cost == 0:
                            return best_local_colors
                    T = 1.0

        if best_local_cost < best_global_cost:
            best_global_cost = best_local_cost
            best_global_colors = best_local_colors

    if best_global_colors is not None and best_global_cost == 0:
        return best_global_colors

    # Fallback to greedy
    def greedy_construct():
        colors = []
        for x in range(1, n + 1):
            for color in range(128):
                if all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3 for d in range(1, (x - 1) // 3 + 1)):
                    colors.append(color)
                    break
            else:
                colors.append(0)
        return colors

    return greedy_construct()
