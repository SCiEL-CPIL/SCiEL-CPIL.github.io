"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _score(centers):
    """Compute the packing radius."""
    wall = 0.5 - np.abs(centers)
    min_wall = np.min(wall)
    d = centers[:, None, :] - centers[None, :, :]
    dsq = np.einsum('ijk,ijk->ij', d, d)
    np.fill_diagonal(dsq, np.inf)
    min_pair = np.sqrt(np.min(dsq))
    return min(min_pair * 0.5, min_wall)


def _sa(centers, rng, budget):
    """Simulated annealing to maximize packing radius."""
    t0 = time.time()
    c = centers.copy()
    w = 0.5 - np.abs(c)
    d = c[:, None, :] - c[None, :, :]
    dsq = np.einsum('ijk,ijk->ij', d, d)
    np.fill_diagonal(dsq, np.inf)
    
    best_c = c.copy()
    best_s = _score(c)
    cur_s = best_s
    
    T = 0.02
    T_min = 1e-10
    step = 0.015
    no_imp = 0
    
    while time.time() - t0 < budget:
        T *= 0.9998
        if T < T_min:
            T = T_min
        
        i = rng.integers(N)
        old = c[i].copy()
        new = old + rng.normal(0, step, 2)
        new = np.clip(new, -0.4999, 0.4999)
        
        # Compute new distances from i
        nd = c - new
        ndsq = np.einsum('ij,ij->i', nd, nd)
        ndsq[i] = np.inf
        
        nw = 0.5 - np.abs(new)
        nwi = np.min(nw)
        
        # New min wall
        old_wi = np.min(0.5 - np.abs(old))
        if i == 0:
            wexcl = np.min(w[1:])
        elif i == N - 1:
            wexcl = np.min(w[:-1])
        else:
            wexcl = np.min(np.concatenate([w[:i], w[i+1:]]))
        new_min_wall = min(nwi, wexcl)
        
        # New min pair: min of pairs not involving i, and new distances from i
        # Pairs not involving i: need min of dsq excluding row/col i
        if i == 0:
            pexcl = np.min(dsq[1:, 1:])
        elif i == N - 1:
            pexcl = np.min(dsq[:-1, :-1])
        else:
            tmp = dsq.copy()
            tmp[i, :] = np.inf
            tmp[:, i] = np.inf
            pexcl = np.min(tmp)
        
        new_min_pair_sq = min(np.min(ndsq), pexcl)
        new_s = min(np.sqrt(new_min_pair_sq) * 0.5, new_min_wall)
        
        delta = new_s - cur_s
        
        accept = False
        if delta > 0:
            accept = True
        else:
            if delta / T > -50:
                if rng.random() < np.exp(delta / T):
                    accept = True
        
        if accept:
            c[i] = new
            w[i] = nw
            c[i] = new
            # Update dsq row/col i
            dd = c - new
            dd_sq = np.einsum('ij,ij->i', dd, dd)
            dsq[i, :] = dd_sq
            dsq[:, i] = dd_sq
            dsq[i, i] = np.inf
            
            cur_s = new_s
            no_imp = 0
            if cur_s > best_s:
                best_s = cur_s
                best_c = c.copy()
        else:
            no_imp += 1
        
        if no_imp > 3000:
            step *= 0.5
            no_imp = 0
        elif it % 5000 == 0 if 'it' in dir() else False:
            pass
        
        # Periodic step reset
        if no_imp > 10000:
            step = 0.015 * (0.5 + rng.random())
            no_imp = 0
    
    # Final polish
    remaining = budget - (time.time() - t0)
    if remaining > 2:
        c = best_c.copy()
        cs = _score(c)
        rs = 0.003
        while time.time() - t0 < budget - 0.5:
            imp = False
            for _ in range(300):
                i = rng.integers(N)
                old = c[i].copy()
                new = old + rng.normal(0, rs, 2)
                new = np.clip(new, -0.4999, 0.4999)
                c[i] = new
                ns = _score(c)
                if ns > cs:
                    cs = ns
                    imp = True
                    if cs > best_s:
                        best_s = cs
                        best_c = c.copy()
                else:
                    c[i] = old
            if not imp:
                rs *= 0.5
                if rs < 1e-7:
                    break
    
    return best_c, best_s


def construct():
    """Multi-start SA for 50-circle packing in unit square."""
    rng = np.random.default_rng(2024)
    t0 = time.time()
    T_max = 110.0
    
    best_c = None
    best_s = -1
    
    # Initial configs
    inits = []
    
    # Grid 7x8
    for rows, cols in [(7, 8), (8, 7), (5, 10), (10, 5), (6, 9)]:
        xs = np.linspace(-0.48, 0.48, cols)
        ys = np.linspace(-0.48, 0.48, rows)
        c = np.zeros((N, 2))
        k = 0
        for r in range(rows):
            for cc in range(cols):
                if k < N:
                    c[k] = [xs[cc], ys[r]]
                    k += 1
                else:
                    break
            if k >= N:
                break
        c += rng.normal(0, 0.002, (N, 2))
        c = np.clip(c, -0.4999, 0.4999)
        inits.append(c)
    
    # Hexagonal
    c = np.zeros((N, 2))
    sp = 0.19
    rh = sp * np.sqrt(3) / 2
    k = 0
    row = 0
    while k < N and row < 12:
        off = 0 if row % 2 == 0 else sp / 2
        for col in range(10):
            if k >= N:
                break
            x = -0.48 + off + col * sp
            y = -0.48 + row * rh
            if -0.4999 <= x <= 0.4999 and -0.4999 <= y <= 0.4999:
                c[k] = [x, y]
                k += 1
        row += 1
    if k == N:
        c += rng.normal(0, 0.002, (N, 2))
        c = np.clip(c, -0.4999, 0.4999)
        inits.append(c)
    
    # Random
    inits.append(rng.uniform(-0.45, 0.45, (N, 2)))
    
    for ii, ic in enumerate(inits):
        rem = T_max - (time.time() - t0)
        if rem < 3:
            break
        tb = min(rem, max(8.0, rem / (len(inits) - ii)))
        lc, ls = _sa(ic, rng, tb)
        if ls > best_s:
            best_s = ls
            best_c = lc
    
    if best_c is None:
        xs = np.linspace(-0.48, 0.48, 8)
        ys = np.linspace(-0.48, 0.48, 7)
        best_c = np.zeros((N, 2))
        k = 0
        for r in range(7):
            for cc in range(8):
                if k < N:
                    best_c[k] = [xs[cc], ys[r]]
                    k += 1
                else:
                    break
            if k >= N:
                break
    
    return best_c
# EVOLVE-BLOCK-END
