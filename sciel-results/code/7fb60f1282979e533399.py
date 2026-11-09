"""Rewritten design for TM band-gap photonic crystal unit cell."""
import numpy as np
import time
import random

_START_TIME = time.time()
_WALL_LIMIT = 215.0

def _check_deadline():
    return (time.time() - _START_TIME) < _WALL_LIMIT

def design(problem):
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    N_PW = 8
    G_range = np.arange(-N_PW, N_PW + 1)
    Gx, Gy = np.meshgrid(G_range, G_range, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    N_G = len(G_vectors)
    
    # Build lookup: index in G_vectors for each (gx, gy)
    g_to_idx = {}
    for idx, (gx, gy) in enumerate(G_vectors):
        g_to_idx[(int(gx), int(gy))] = idx
    
    # K-points
    k_points = []
    for t in np.linspace(0, 0.5, 8):
        k_points.append((t, t))
    for t in np.linspace(0.01, 0.5, 8):
        k_points.append((0.5, t))
    for t in np.linspace(0.01, 0.5, 8):
        k_points.append((t, 0.01))
    k_points = list(set([tuple(p) for p in k_points]))
    
    def compute_inv_eps(cell_np):
        inv_eps = np.where(cell_np == 1, 1.0/eps_high, 1.0/eps_low)
        fft2 = np.fft.fftshift(np.fft.fft2(inv_eps)) / (size*size)
        coeffs = np.zeros(N_G, dtype=complex)
        for idx in range(N_G):
            gx, gy = G_vectors[idx]
            igx, igy = int(gx) + N_PW, int(gy) + N_PW
            if 0 <= igx < size and 0 <= igy < size:
                coeffs[idx] = fft2[igy, igx]
        return coeffs
    
    def eval_cell(cell_np):
        inv_coeffs = compute_inv_eps(cell_np)
        max_w4, min_w5 = 0.0, np.inf
        for kx, ky in k_points:
            kg2 = np.array([(kx+gx)**2 + (ky+gy)**2 for gx,gy in G_vectors])
            conv = np.zeros((N_G, N_G), dtype=complex)
            for i in range(N_G):
                for j in range(N_G):
                    dgx = int(G_vectors[i,0] - G_vectors[j,0])
                    dgy = int(G_vectors[i,1] - G_vectors[j,1])
                    if (dgx, dgy) in g_to_idx:
                        conv[i,j] = inv_coeffs[g_to_idx[(dgx,dgy)]]
            M = np.diag(kg2) @ conv
            try:
                ev = np.sort(np.real(np.linalg.eigvalsh(M)))
                om = np.sqrt(np.maximum(ev, 0))
                if len(om) >= 5:
                    max_w4 = max(max_w4, om[3])
                    min_w5 = min(min_w5, om[4])
            except:
                pass
        if min_w5 == np.inf:
            return 0.0, 0.0
        return max_w4, min_w5
    
    def score(mw4, mn5):
        s = mw4 + mn5
        if s <= 0: return 0.0
        return 2.0*(mn5 - mw4)/s
    
    def sym(cell):
        c = np.array(cell)
        for i in range(size):
            for j in range(size//2):
                c[i,j] = max(c[i,j], c[i,size-1-j])
                c[i,size-1-j] = c[i,j]
        for i in range(size//2):
            for j in range(size):
                c[i,j] = max(c[i,j], c[size-1-i,j])
                c[size-1-i,j] = c[i,j]
        for i in range(size):
            for j in range(size):
                if i<j:
                    v = max(c[i,j], c[j,i])
                    c[i,j] = v
                    c[j,i] = v
        return c.tolist()
    
    def rand_cell():
        c = np.zeros((size,size), dtype=int)
        for i in range(size//2+1):
            for j in range(size//2+1):
                v = random.randint(0,1)
                c[i,j]=v
                c[i,size-1-j]=v
                c[size-1-i,j]=v
                c[size-1-i,size-1-j]=v
                if i!=j:
                    c[j,i]=v
                    c[j,size-1-i]=v
                    c[size-1-j,i]=v
                    c[size-1-j,size-1-i]=v
        return c.tolist()
    
    def mut(cell):
        c = np.array(cell)
        i,j = random.randint(0,size-1), random.randint(0,size-1)
        c[i,j] = 1-c[i,j]
        return sym(c)
    
    # Seed
    sc = np.zeros((size,size), dtype=int)
    for i in range(size):
        for j in range(size):
            x=(i+0.5)/size-0.5
            y=(j+0.5)/size-0.5
            sc[i,j] = 1 if x*x+y*y < 0.04 else 0
    sc = sym(sc)
    
    best = sc.tolist()
    mw4, mn5 = eval_cell(sc)
    bs = score(mw4, mn5)
    
    # Local search
    cur = sc.copy()
    cs = bs
    noimp = 0
    while noimp < 25 and _check_deadline():
        m = mut(cur.tolist())
        mnp = np.array(m)
        mw, mn = eval_cell(mnp)
        ms = score(mw, mn)
        if ms > cs:
            cur = mnp
            cs = ms
            noimp = 0
            if cs > bs:
                bs = cs
                best = m
        else:
            noimp += 1
    
    return best
