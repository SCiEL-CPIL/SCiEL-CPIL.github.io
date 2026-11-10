import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    total_points = q * q
    num_dirs = q + 1
    
    start_time = time.time()
    deadline = start_time + 570
    
    def point_idx(xa, xb, ya, yb):
        return xa + xb * p + ya * q + yb * q * p
    
    # Precompute mul table
    mul = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ai, bi = i % p, i // p
        for j in range(q):
            aj, bj = j % p, j // p
            ra = (ai * aj + bi * bj * 3) % p
            rb = (ai * bj + bi * aj) % p
            mul[i, j] = ra + rb * p
    
    # Precompute other_points
    other_points = np.zeros((total_points, num_dirs, q - 1), dtype=np.int32)
    for pi in range(total_points):
        xa = pi % p
        xb = (pi // p) % p
        ya = (pi // q) % p
        yb = (pi // (q * p)) % p
        x_idx = xa + xb * p
        y_idx = ya + yb * p
        for d in range(num_dirs):
            if d < q:
                m = d
                for t in range(1, q):
                    t_a, t_b = t % p, t // p
                    nx_a = (xa + t_a) % p
                    nx_b = (xb + t_b) % p
                    mt = mul[m, t]
                    mt_a, mt_b = mt % p, mt // p
                    ny_a = (ya + mt_a) % p
                    ny_b = (yb + mt_b) % p
                    other_points[pi, d, t-1] = point_idx(nx_a, nx_b, ny_a, ny_b)
            else:
                for t in range(1, q):
                    t_a, t_b = t % p, t // p
                    ny_a = (ya + t_a) % p
                    ny_b = (yb + t_b) % p
                    other_points[pi, d, t-1] = point_idx(xa, xb, ny_a, ny_b)
    
    # Also precompute the "reverse" mapping: for each point p and direction d,
    # which points x have p in other_points[x, d]? By symmetry, this is other_points[p, d].
    
    # Vectorized greedy
    best_S_size = 0
    best_S = np.zeros(total_points, dtype=bool)
    
    num_trials = 30
    for trial in range(num_trials):
        if time.time() > deadline - 15:
            break
        
        order = np.random.permutation(total_points)
        S = np.zeros(total_points, dtype=bool)
        s_count = np.zeros((total_points, num_dirs), dtype=np.int32)
        valid_count = np.full(total_points, num_dirs, dtype=np.int32)
        
        for p in order:
            if time.time() > deadline - 5:
                break
            
            # Vectorized check: for all x in other_points[p, :], check if any has s_count[x,d]==0 and valid_count[x]==1
            # other_points[p, :] has shape (num_dirs, q-1)
            # We need to check: exists d, i such that s_count[other_points[p,d,i], d] == 0 and valid_count[other_points[p,d,i]] == 1
            
            # Reshape for broadcasting
            x_arr = other_points[p]  # shape (num_dirs, q-1)
            d_arr = np.arange(num_dirs)[:, None]  # shape (num_dirs, 1)
            
            # s_count[x_arr, d_arr] - but x_arr is 2D and d_arr is 2D, need to align
            # s_count[x_arr, d_arr] doesn't work directly. Let's use advanced indexing.
            
            # Flatten
            x_flat = x_arr.ravel()  # shape (num_dirs*(q-1),)
            d_flat = np.repeat(np.arange(num_dirs), q-1)  # shape (num_dirs*(q-1),)
            
            sc_vals = s_count[x_flat, d_flat]  # shape (num_dirs*(q-1),)
            vc_vals = valid_count[x_flat]  # shape (num_dirs*(q-1),)
            
            # Check if any has sc==0 and vc==1
            if np.any((sc_vals == 0) & (vc_vals == 1)):
                continue
            
            # Add p to S
            S[p] = True
            # Update s_count and valid_count
            for d in range(num_dirs):
                xs = other_points[p, d]
                was_zero = (s_count[xs, d] == 0)
                valid_count[xs[was_zero]] -= 1
                s_count[xs, d] += 1
        
        S_size = S.sum()
        if S_size > best_S_size:
            best_S_size = S_size
            best_S = S.copy()
    
    N_mask = ~best_S
    points = []
    for pi in range(total_points):
        if N_mask[pi]:
            xa = pi % p
            xb = (pi // p) % p
            ya = (pi // q) % p
            yb = (pi // (q * p)) % p
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)
