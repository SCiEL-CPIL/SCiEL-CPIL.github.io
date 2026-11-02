import numpy as np
import time

N = 400

def _max_a3(f):
    f64 = f.astype(np.float64)
    mx = 0.0
    for t in range(N):
        n = N - 2*t
        if n <= 0:
            break
        v = np.sum(f64[:n] * f64[t:t+n] * f64[2*t:2*t+n])
        av = abs(v)
        if av > mx:
            mx = av
    return mx

def construct():
    t0 = time.time()
    dl = t0 + 130
    
    best_f = np.ones(N, dtype=np.int8)
    best_mx = _max_a3(best_f)
    
    rng = np.random.default_rng(777)
    
    # Generate candidates
    cands = []
    for _ in range(200):
        cands.append(rng.choice([-1,1], size=N).astype(np.int8))
    
    # CA
    for rule in [30, 45, 73, 86, 105, 110, 150, 184, 225, 254, 15, 54, 102, 110, 126, 150, 184, 225]:
        for _ in range(10):
            L = N + 200
            row = rng.integers(0, 2, size=L)
            for _ in range(40):
                pat = (np.roll(row,1) << 2) | (row << 1) | np.roll(row,-1)
                row = ((rule >> pat) & 1).astype(np.int8)
            cands.append((row[100:100+N]*2-1).astype(np.int8))
    
    for f in cands:
        if time.time() > dl - 60:
            break
        mx = _max_a3(f)
        if mx < best_mx:
            best_mx = mx
            best_f = f.copy()
    
    # Local search
    if time.time() < dl - 5:
        f = best_f.copy()
        cmx = best_mx
        for pass_num in range(50):
            if time.time() > dl - 5:
                break
            improved = False
            for i in range(N):
                if time.time() > dl - 5:
                    break
                f[i] = -f[i]
                nm = _max_a3(f)
                if nm < cmx:
                    cmx = nm
                    best_f = f.copy()
                    best_mx = cmx
                    improved = True
                else:
                    f[i] = -f[i]
            if not improved:
                break
    
    return best_f
