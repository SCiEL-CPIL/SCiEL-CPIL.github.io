import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 110.0  # 10s margin
    
    # Thue-Morse initialization
    colors = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    
    def count_mono():
        count = 0
        for d in range(1, (n - 3) // 3 + 1):
            for a in range(n - 3 * d):
                if colors[a] == colors[a+d] == colors[a+2*d] == colors[a+3*d]:
                    count += 1
        return count

    current_cost = count_mono()
    best_cost = current_cost
    best_colors = colors.copy()
    
    # Precompute max d
    max_d = (n - 3) // 3
    
    def delta_flip(x):
        """Calculate change in cost if we flip color at x."""
        delta = 0
        for d in range(1, max_d + 1):
            # Check if x is in an AP with difference d
            # x = a + k*d for k in 0..3
            # a = x - k*d
            for k in range(4):
                a = x - k * d
                if a < 0:
                    continue
                if a + 3 * d >= n:
                    continue
                # AP is (a, a+d, a+2d, a+3d)
                # x is at position k in this AP
                # Get the 4 values
                v0 = colors[a]
                v1 = colors[a+d]
                v2 = colors[a+2*d]
                v3 = colors[a+3*d]
                
                # Check old mono
                old_mono = 1 if (v0 == v1 == v2 == v3) else 0
                
                # Flip x
                if k == 0:
                    nv0 = 1 - v0
                    nv1 = v1
                    nv2 = v2
                    nv3 = v3
                elif k == 1:
                    nv0 = v0
                    nv1 = 1 - v1
                    nv2 = v2
                    nv3 = v3
                elif k == 2:
                    nv0 = v0
                    nv1 = v1
                    nv2 = 1 - v2
                    nv3 = v3
                else:
                    nv0 = v0
                    nv1 = v1
                    nv2 = v2
                    nv3 = 1 - v3
                    
                new_mono = 1 if (nv0 == nv1 == nv2 == nv3) else 0
                delta += new_mono - old_mono
        return delta

    # Simulated Annealing
    T = 1.0
    cool_rate = 0.99
    min_T = 0.01
    steps = 0
    
    while time.time() < deadline:
        x = random.randint(0, n - 1)
        d_cost = delta_flip(x)
        
        if d_cost < 0:
            # Accept
            colors[x] = 1 - colors[x]
            current_cost += d_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
        else:
            # Accept with probability
            if T > min_T:
                p = np.exp(-d_cost / T)
                if random.random() < p:
                    colors[x] = 1 - colors[x]
                    current_cost += d_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors.copy()
        
        T *= cool_rate
        steps += 1
        
        # Restart logic if stuck?
        # For now, just keep going until time limit.
        
    return best_colors.tolist()
