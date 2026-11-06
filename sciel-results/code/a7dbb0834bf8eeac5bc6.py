import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    # Build edge index mapping
    ei = [[-1]*N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            ei[i][j] = idx
            ei[j][i] = idx
            idx += 1
    
    # Precompute all cycles as flat array
    cycles_e = []  # list of (e1,e2,e3,e4)
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                for d in range(c+1, N):
                    e_ab = ei[a][b]
                    e_bc = ei[b][c]
                    e_cd = ei[c][d]
                    e_da = ei[d][a]
                    e_bd = ei[b][d]
                    e_ca = ei[c][a]
                    cycles_e.append((e_ab, e_bc, e_cd, e_da))
                    cycles_e.append((e_ab, e_bd, e_cd, e_ca))
                    cycles_e.append((e_ca, e_bc, e_bd, e_da))
    
    ncyc = len(cycles_e)
    
    # edge -> list of cycle indices
    ecyc = [[] for _ in range(NUM_EDGES)]
    for ci in range(ncyc):
        for e in cycles_e[ci]:
            ecyc[e].append(ci)
    
    # Initial coloring: star from 0
    colors = [0] * NUM_EDGES
    for j in range(1, N):
        colors[ei[0][j]] = j
    
    # Count violations
    viol = [0] * ncyc
    tv = 0
    for ci in range(ncyc):
        e1,e2,e3,e4 = cycles_e[ci]
        c1,c2,c3,c4 = colors[e1],colors[e2],colors[e3],colors[e4]
        if len({c1,c2,c3,c4}) == 4:
            viol[ci] = 1
            tv += 1
    
    best = colors[:]
    bc = len(set(best))
    cc = bc
    
    temp = 2.0
    cool = 0.9995
    rng = random.Random(123)
    
    while time.time() - start_time < DEADLINE:
        e = rng.randrange(NUM_EDGES)
        oc = colors[e]
        nc = rng.randrange(120)
        if nc == oc:
            continue
        
        colors[e] = nc
        aff = ecyc[e]
        dv = 0
        for ci in aff:
            e1,e2,e3,e4 = cycles_e[ci]
            old_v = viol[ci]
            c1,c2,c3,c4 = colors[e1],colors[e2],colors[e3],colors[e4]
            nv = 1 if len({c1,c2,c3,c4}) == 4 else 0
            viol[ci] = nv
            dv += nv - old_v
        tv += dv
        
        if tv == 0:
            ncc = len(set(colors))
            d = ncc - cc
            if d >= 0 or rng.random() < temp ** (-d):
                cc = ncc
                if ncc > bc:
                    bc = ncc
                    best = colors[:]
            else:
                colors[e] = oc
                for ci in aff:
                    e1,e2,e3,e4 = cycles_e[ci]
                    c1,c2,c3,c4 = colors[e1],colors[e2],colors[e3],colors[e4]
                    viol[ci] = 1 if len({c1,c2,c3,c4}) == 4 else 0
                tv -= dv
        else:
            if rng.random() < temp ** (-tv * 2):
                pass
            else:
                colors[e] = oc
                for ci in aff:
                    e1,e2,e3,e4 = cycles_e[ci]
                    c1,c2,c3,c4 = colors[e1],colors[e2],colors[e3],colors[e4]
                    viol[ci] = 1 if len({c1,c2,c3,c4}) == 4 else 0
                tv -= dv
        
        temp *= cool
    
    if time.time() - start_time > 88:
        return [0] * NUM_EDGES
    
    return best
