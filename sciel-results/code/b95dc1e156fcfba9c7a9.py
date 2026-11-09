import time
import random
import math

def construct():
    n = 256
    # Safety margin: Evaluator kills at 150s. We use 110s.
    deadline = time.time() + 110.0
    
    # Thue-Morse initialization
    # c_i = popcount(i) % 2
    colors = [bin(i).count('1') % 2 for i in range(n)]
    
    # Precompute neighbors for each index
    # neighbors[i] contains tuples of the other 3 indices for each 4-AP containing i
    neighbors = [[] for _ in range(n)]
    max_d = (n - 3) // 3
    for d in range(1, max_d + 1):
        for a in range(n - 3 * d):
            # The 4 indices in the AP
            idxs = [a, a + d, a + 2 * d, a + 3 * d]
            for i in idxs:
                # Find the other 3
                others = []
                for j in idxs:
                    if j != i:
                        others.append(j)
                neighbors[i].append(tuple(others))
    
    def count_mono():
        count = 0
        for d in range(1, max_d + 1):
            for a in range(n - 3 * d):
                if colors[a] == colors[a + d] == colors[a + 2 * d] == colors[a + 3 * d]:
                    count += 1
        return count

    current_cost = count_mono()
    best_cost = current_cost
    best_colors = colors.copy()
    
    T = 5.0
    min_T = 0.01
    cooling = 0.9999
    
    # Main SA loop
    # We use a batch of 100 moves before checking time to reduce overhead
    while time.time() < deadline - 5.0:
        for _ in range(100):
            i = random.randint(0, n - 1)
            old_val = colors[i]
            new_val = 1 - old_val
            
            delta = 0
            # Calculate delta
            for j, k, l in neighbors[i]:
                vj = colors[j]
                vk = colors[k]
                vl = colors[l]
                if vj == vk == vl:
                    if vj == old_val:
                        delta -= 1
                    else:
                        delta += 1
            
            # Acceptance criterion
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = True
            else:
                # delta > 0
                # Avoid exp if T is very small and delta large?
                # math.exp(-1000) is 0.
                if T < 1e-6:
                    accept = False
                else:
                    accept = random.random() < math.exp(-delta / T)
            
            if accept:
                colors[i] = new_val
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
        
        T *= cooling
        if T < min_T:
            T = min_T
            
    return best_colors
