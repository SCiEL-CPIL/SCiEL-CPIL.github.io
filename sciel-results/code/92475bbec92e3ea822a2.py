import numpy as np

def project_simplex(v):
    """Project vector v onto the probability simplex."""
    n = len(v)
    if np.sum(v) == 0:
        return np.zeros(n)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    return w

def construct():
    n = 50
    # 1. Find a good initial support (Sidon set)
    def find_sidon_set(n, max_trials=200):
        best_set = []
        for _ in range(max_trials):
            indices = list(range(n))
            np.random.shuffle(indices)
            current_set = []
            used_sums = set()
            for i in indices:
                valid = True
                new_sums = []
                for j in current_set:
                    s = i + j
                    if s in used_sums:
                        valid = False
                        break
                    new_sums.append(s)
                if valid:
                    s_ii = i + i
                    if s_ii in used_sums:
                        valid = False
                    else:
                        new_sums.append(s_ii)
                if valid:
                    current_set.append(i)
                    used_sums.update(new_sums)
            if len(current_set) > len(best_set):
                best_set = current_set
        return best_set

    sidon_set = find_sidon_set(n)
    
    # Initialize h
    h = np.zeros(n)
    if len(sidon_set) > 0:
        for idx in sidon_set:
            h[idx] = 1.0 / len(sidon_set)
    else:
        h[0] = 1.0

    def calc_score(h):
        l1 = np.sum(h)
        if l1 == 0: return 0.0
        l2_sq = np.sum(h**2)
        c = np.convolve(h, h, mode='full')
        c_max = np.max(c)
        if c_max == 0: return 0.0
        return (l2_sq / (l1 * c_max))

    best_score = calc_score(h)
    best_h = h.copy()

    # 2. Gradient Ascent Optimization
    # We maximize J(h) = ||h||_2^2 / (||h||_1 * ||c||_inf)
    # Since we project to simplex, ||h||_1 = 1 always.
    # So we maximize ||h||_2^2 / ||c||_inf.
    
    steps = 200
    # Adaptive learning rate
    lr = 0.1
    
    for step in range(steps):
        c = np.convolve(h, h, mode='full')
        c_max = np.max(c)
        if c_max == 0:
            break
        
        # Find indices of max c
        max_indices = np.where(c == c_max)[0]
        
        # Compute gradient of M = max(c_k)
        # We use the subgradient corresponding to the max indices.
        # To be stable, we can average gradients over all max indices or just pick one.
        # Picking one is a valid subgradient. Let's pick the first one for simplicity, 
        # or average if multiple. Averaging might be smoother.
        
        grad_M = np.zeros(n)
        for k in max_indices:
            # c_k = sum_{i+j=k} h_i h_j
            # d c_k / d h_i = sum_{j: i+j=k} h_j
            # j = k - i
            # i ranges from max(0, k-(n-1)) to min(n-1, k)
            # For each i, contribution is h[k-i] if k-i is in [0, n-1]
            
            # Vectorized:
            # i from 0 to n-1
            # j = k - i
            # valid if 0 <= j < n
            # grad_M[i] += h[k-i]
            
            # We can do this via convolution or direct loop. 
            # n=50 is small, loop is fine.
            for i in range(n):
                j = k - i
                if 0 <= j < n:
                    grad_M[i] += h[j]
        
        # Gradient of J = S / M where S = ||h||_2^2
        # dJ/dh = (M * dS/dh - S * dM/dh) / M^2
        # dS/dh = 2h
        S = np.sum(h**2)
        grad_J = (c_max * 2 * h - S * grad_M) / (c_max ** 2)
        
        # Update
        h_new = h + lr * grad_J
        
        # Project to simplex
        h_new = project_simplex(h_new)
        
        # Check score
        score = calc_score(h_new)
        if score > best_score:
            best_score = score
            best_h = h_new.copy()
            
        # Simple convergence check or step decay
        h = h_new
        lr *= 0.99 # Decay learning rate slightly

    return list(best_h)
