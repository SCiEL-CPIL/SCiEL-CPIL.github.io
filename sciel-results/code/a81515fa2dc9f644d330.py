import random
import time
import math

def construct():
    n = 256
    # Random initialization
    color = [random.randint(0, 1) for _ in range(n)]
    
    def count_mono_4ap(color):
        count = 0
        for d in range(1, n // 3 + 1):
            for a in range(1, n - 3 * d + 1):
                if color[a-1] == color[a+d-1] == color[a+2*d-1] == color[a+3*d-1]:
                    count += 1
        return count

    def delta_from_flip(color, p, n):
        """Compute change in monochromatic 4-AP count when flipping position p (0-indexed)."""
        delta = 0
        c = color[p]
        for d in range(1, n // 3 + 1):
            # p as first element: (p, p+d, p+2d, p+3d) 0-indexed
            if p + 3*d < n:
                others = [color[p+d], color[p+2*d], color[p+3*d]]
                if all(o == c for o in others):
                    delta -= 1
                elif all(o != c for o in others):
                    delta += 1
            # p as second: (p-d, p, p+d, p+2d)
            if p - d >= 0 and p + 2*d < n:
                others = [color[p-d], color[p+d], color[p+2*d]]
                if all(o == c for o in others):
                    delta -= 1
                elif all(o != c for o in others):
                    delta += 1
            # p as third: (p-2d, p-d, p, p+d)
            if p - 2*d >= 0 and p + d < n:
                others = [color[p-2*d], color[p-d], color[p+d]]
                if all(o == c for o in others):
                    delta -= 1
                elif all(o != c for o in others):
                    delta += 1
            # p as fourth: (p-3d, p-2d, p-d, p)
            if p - 3*d >= 0:
                others = [color[p-3*d], color[p-2*d], color[p-d]]
                if all(o == c for o in others):
                    delta -= 1
                elif all(o != c for o in others):
                    delta += 1
        return delta

    current_cost = count_mono_4ap(color)
    best_cost = current_cost
    best_color = color[:]
    
    start_time = time.time()
    time_limit = 110.0  # Leave some buffer
    
    T = 10.0
    cooling_rate = 0.9999
    
    while time.time() - start_time < time_limit:
        p = random.randint(0, n - 1)
        delta = delta_from_flip(color, p, n)
        
        if delta < 0:
            color[p] = 1 - color[p]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_color = color[:]
        else:
            # Accept with probability exp(-delta / T)
            if random.random() < math.exp(-delta / T):
                color[p] = 1 - color[p]
                current_cost += delta
        
        T *= cooling_rate
        if T < 0.01:
            T = 0.01

    return best_color
