import numpy as np
import time

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 550.0
    
    best_score = -np.inf
    best_centers = None
    best_sides = None
    
    def energy(x, y, s, lam=10.0, mu=10.0):
        # x, y: bottom-left, s: side
        e = -np.sum(s)
        
        # Boundaries
        # x < 0, x+s > 1, y < 0, y+s > 1, s < 0
        v = 0
        v += np.sum(np.maximum(0, -x)**2)
        v += np.sum(np.maximum(0, x + s - 1)**2)
        v += np.sum(np.maximum(0, -y)**2)
        v += np.sum(np.maximum(0, y + s - 1)**2)
        v += np.sum(np.maximum(0, -s)**2)
        e += mu * v
        
        # Overlaps
        for i in range(n):
            for j in range(i+1, n):
                ox = min(x[i]+s[i], x[j]+s[j]) - max(x[i], x[j])
                oy = min(y[i]+s[i], y[j]+s[j]) - max(y[i], y[j])
                if ox > 0:
                    e += lam * ox**2
                if oy > 0:
                    e += lam * oy**2
        return e

    def is_valid(x, y, s, tol=1e-7):
        if np.any(x < -tol) or np.any(y < -tol) or np.any(x+s > 1+tol) or np.any(y+s > 1+tol) or np.any(s < tol):
            return False
        for i in range(n):
            for j in range(i+1, n):
                ox = min(x[i]+s[i], x[j]+s[j]) - max(x[i], x[j])
                oy = min(y[i]+s[i], y[j]+s[j]) - max(y[i], y[j])
                if ox > tol and oy > tol:
                    return False
        return True

    def init_config():
        # Start with a valid grid, then perturb
        k = 3
        side = 1.0/3
        x = np.zeros(n)
        y = np.zeros(n)
        s = np.zeros(n)
        for i in range(n):
            r = i // k
            c = i % k
            x[i] = c * side
            y[i] = r * side
            s[i] = side
        # Perturb
        x += np.random.uniform(-0.05, 0.05, n)
        y += np.random.uniform(-0.05, 0.05, n)
        s += np.random.uniform(-0.05, 0.05, n)
        # Clip to valid range roughly
        s = np.clip(s, 0.1, 1.0)
        x = np.clip(x, 0, 1-s)
        y = np.clip(y, 0, 1-s)
        return x, y, s

    # Simulated Annealing
    T = 0.1
    T_min = 0.0001
    cooling = 0.99
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        x, y, s = init_config()
        current_e = energy(x, y, s)
        
        for step in range(5000):
            if time.time() > deadline:
                break
                
            # Random move
            idx = np.random.randint(n)
            dx = np.random.normal(0, 0.05)
            dy = np.random.normal(0, 0.05)
            ds = np.random.normal(0, 0.05)
            
            new_x = x.copy()
            new_y = y.copy()
            new_s = s.copy()
            
            new_x[idx] += dx
            new_y[idx] += dy
            new_s[idx] += ds
            
            # Hard clip s to be positive
            if new_s[idx] < 0.01:
                new_s[idx] = 0.01
            
            # Hard clip x, y to keep inside [0,1] roughly
            new_x[idx] = np.clip(new_x[idx], 0, 1 - new_s[idx])
            new_y[idx] = np.clip(new_y[idx], 0, 1 - new_s[idx])
            
            new_e = energy(new_x, new_y, new_s)
            
            # Acceptance
            if new_e < current_e:
                x, y, s = new_x, new_y, new_s
                current_e = new_e
            else:
                if np.random.random() < np.exp((current_e - new_e) / T):
                    x, y, s = new_x, new_y, new_s
                    current_e = new_e
            
            T *= cooling
            
        # Check best valid solution
        if is_valid(x, y, s):
            score = np.sum(s)
            if score > best_score:
                best_score = score
                best_centers = np.stack([x + s/2, y + s/2], axis=1)
                best_sides = s.copy()
                
    if best_centers is None:
        # Fallback to grid
        k = 3
        side = 1.0/3
        x = np.zeros(n)
        y = np.zeros(n)
        s = np.zeros(n)
        for i in range(n):
            r = i // k
            c = i % k
            x[i] = c * side
            y[i] = r * side
            s[i] = side
        best_centers = np.stack([x + s/2, y + s/2], axis=1)
        best_sides = s
        
    return best_centers, best_sides
