import numpy as np
import time

def construct():
    # Try N=51 first, then fall back to N=50
    for N in [51, 50]:
        result = _solve(N)
        if result:
            return result
    # Fallback to 32 if all else fails (should not happen)
    return _solve_32()

def _solve(N):
    E = N * (N - 1) // 2
    # Map (i, j) to edge index
    edge_idx = np.full((N, N), -1, dtype=np.int32)
    us = np.zeros(E, dtype=np.int32)
    vs = np.zeros(E, dtype=np.int32)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_idx[i, j] = idx
            edge_idx[j, i] = idx
            us[idx] = i
            vs[idx] = j
            idx += 1
            
    rng = np.random.default_rng(42)
    
    # Time limit per N
    start_time = time.time()
    time_limit = 20.0 if N == 51 else 5.0
    
    max_iters = 200000
    
    for restart in range(5):
        colors = rng.integers(0, 4, size=E, dtype=np.int8)
        
        # Initialize conflict counts
        conflict = np.zeros(E, dtype=np.int32)
        
        # Initial conflict calculation
        for i in range(N):
            for j in range(i + 1, N):
                e1 = edge_idx[i, j]
                c1 = colors[e1]
                w_list = np.arange(N)
                w_list = w_list[(w_list != i) & (w_list != j)]
                e2s = edge_idx[i, w_list]
                e3s = edge_idx[j, w_list]
                c2s = colors[e2s]
                c3s = colors[e3s]
                mono = (c2s == c1) & (c3s == c1)
                conflict[e1] = np.sum(mono)
                
        if np.sum(conflict) == 0:
            return {'n': N, 'colors': colors.tolist()}
            
        for step in range(max_iters):
            if time.time() - start_time > time_limit:
                break
                
            e = rng.integers(E)
            if conflict[e] == 0:
                if np.sum(conflict) == 0:
                    return {'n': N, 'colors': colors.tolist()}
                continue
                
            u = us[e]
            v = vs[e]
            c_old = colors[e]
            
            w_list = np.arange(N)
            w_list = w_list[(w_list != u) & (w_list != v)]
            
            e1s = edge_idx[u, w_list]
            e2s = edge_idx[v, w_list]
            c1s = colors[e1s]
            c2s = colors[e2s]
            
            was_mono = (c1s == c_old) & (c2s == c_old)
            
            best_delta = np.inf
            best_color = c_old
            
            for c_new in range(4):
                if c_new == c_old:
                    continue
                is_mono = (c1s == c_new) & (c2s == c_new)
                delta = -3 * np.sum(was_mono & ~is_mono) + 3 * np.sum(~was_mono & is_mono)
                
                if delta < best_delta:
                    best_delta = delta
                    best_color = c_new
                    
            if best_color == c_old:
                best_color = rng.integers(0, 4)
                if best_color == c_old:
                    best_color = (best_color + 1) % 4
                    
            c_new = best_color
            if c_new == c_old:
                continue
                
            colors[e] = c_new
            
            is_mono = (c1s == c_new) & (c2s == c_new)
            
            breaks = was_mono & ~is_mono
            if np.any(breaks):
                conflict[e] -= np.sum(breaks)
                np.add.at(conflict, e1s[breaks], -1)
                np.add.at(conflict, e2s[breaks], -1)
                
            makes = ~was_mono & is_mono
            if np.any(makes):
                conflict[e] += np.sum(makes)
                np.add.at(conflict, e1s[makes], 1)
                np.add.at(conflict, e2s[makes], 1)
                
            if np.sum(conflict) == 0:
                return {'n': N, 'colors': colors.tolist()}
                
    return None

def _solve_32():
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': colors}
