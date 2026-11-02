import numpy as np
import time

N = 100

def compute_min_autocorr(h):
    min_val = np.inf
    for k in range(1, N):
        autocorr = np.sum(h * np.roll(h, k))
        if autocorr < min_val:
            min_val = autocorr
    return min_val

def enforce_sum(h, target):
    h = h.copy()
    for _ in range(10000):
        diff = target - np.sum(h)
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            room = 1 - h
            j = np.argmax(room)
            step = min(diff, room[j])
            if step <= 0:
                break
            h[j] += step
        else:
            room = h
            j = np.argmax(room)
            step = min(-diff, room[j])
            if step <= 0:
                break
            h[j] -= step
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_h = np.full(N, 0.5)
    best_min_ac = compute_min_autocorr(best_h)
    
    np.random.seed(42)
    
    # Try several initializations
    inits = []
    
    # 1. Quadratic irrational (golden ratio)
    alpha = (np.sqrt(5) - 1) / 2
    h1 = np.array([1.0 if (i * alpha) % 1.0 < 0.5 else 0.0 for i in range(N)])
    inits.append(enforce_sum(h1, N/2))
    
    # 2. sqrt(2) - 1
    alpha2 = np.sqrt(2) - 1
    h2 = np.array([1.0 if (i * alpha2) % 1.0 < 0.5 else 0.0 for i in range(N)])
    inits.append(enforce_sum(h2, N/2))
    
    # 3. sqrt(3) - 1
    alpha3 = np.sqrt(3) - 1
    h3 = np.array([1.0 if (i * alpha3) % 1.0 < 0.5 else 0.0 for i in range(N)])
    inits.append(enforce_sum(h3, N/2))
    
    # 4. Random binary
    h4 = np.array([1.0 if np.random.random() < 0.5 else 0.0 for _ in range(N)])
    inits.append(enforce_sum(h4, N/2))
    
    for h_init in inits:
        min_ac = compute_min_autocorr(h_init)
        if min_ac > best_min_ac:
            best_min_ac = min_ac
            best_h = h_init.copy()
    
    # Local search
    while time.time() < deadline:
        h_new = best_h.copy()
        n_perturb = np.random.randint(1, 6)
        idx = np.random.choice(N, n_perturb, replace=False)
        for i in idx:
            h_new[i] = np.random.uniform(0, 1)
        
        h_new = enforce_sum(h_new, N/2)
        min_ac = compute_min_autocorr(h_new)
        if min_ac > best_min_ac:
            best_min_ac = min_ac
            best_h = h_new.copy()
    
    return best_h
