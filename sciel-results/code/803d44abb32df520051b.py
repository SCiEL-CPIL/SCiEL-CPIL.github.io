import time
import random
import math

# EVOLVE-BLOCK-START

_DEADLINE = 315.0
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

_N = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

def _energy(seq, chain):
    n = len(chain)
    if n == 0: return 0
    pos_map = {p: i for i, p in enumerate(chain)}
    is_h = [c == 'H' for c in seq]
    contacts = 0
    for i, p in enumerate(chain):
        if is_h[i]:
            x, y, z = p
            for dx, dy, dz in _N:
                j = pos_map.get((x+dx, y+dy, z+dz), -1)
                if j > i + 1 and is_h[j]:
                    contacts += 1
    return -contacts

def _kink_move(chain, i):
    if i <= 0 or i >= len(chain) - 1:
        return None
    occ = set(chain)
    occ.discard(chain[i])
    prev_p, next_p, curr = chain[i-1], chain[i+1], chain[i]
    cx, cy, cz = curr
    for dx, dy, dz in _N:
        site = (cx+dx, cy+dy, cz+dz)
        if site in occ: continue
        if abs(site[0]-prev_p[0])+abs(site[1]-prev_p[1])+abs(site[2]-prev_p[2]) != 1: continue
        if abs(site[0]-next_p[0])+abs(site[1]-next_p[1])+abs(site[2]-next_p[2]) != 1: continue
        new_chain = list(chain)
        new_chain[i] = site
        return new_chain
    return None

def _pivot_move(chain, i, axis, sign):
    n = len(chain)
    if i <= 1 or i >= n - 1: return None
    head = chain[:i+1]
    tail = chain[i+1:]
    pivot = chain[i]
    head_set = set(head)
    new_tail = []
    seen = set()
    px, py, pz = pivot
    for pos in tail:
        vx, vy, vz = pos[0]-px, pos[1]-py, pos[2]-pz
        if axis == 0:
            nv = (vx, -vz, vy) if sign == 1 else (vx, vz, -vy)
        elif axis == 1:
            nv = (vz, vy, -vx) if sign == 1 else (-vz, vy, vx)
        else:
            nv = (-vy, vx, vz) if sign == 1 else (vy, -vx, vz)
        np_ = (px+nv[0], py+nv[1], pz+nv[2])
        if np_ in seen or np_ in head_set: return None
        seen.add(np_)
        new_tail.append(np_)
    return head + new_tail

def _grow_greedy(seq, dim, rng, time_budget):
    n = len(seq)
    is_h = [c == 'H' for c in seq]
    chain = [(0, 0, 0)]
    occ = {(0, 0, 0): 0}
    contacts = 0
    
    for i in range(1, n):
        if _time_left() < 3.0:
            break
        last = chain[-1]
        x, y, z = last
        candidates = []
        for dx, dy, dz in _N:
            site = (x+dx, y+dy, z+dz)
            if site in occ: continue
            # Check if this site is adjacent to any previously placed H (not adjacent in chain)
            gain = 0
            if is_h[i]:
                for dx2, dy2, dz2 in _N:
                    j = occ.get((site[0]+dx2, site[1]+dy2, site[2]+dz2), -1)
                    if 0 <= j < i and j < i - 1 and is_h[j]:
                        gain += 1
            # Small random perturbation to avoid greedy traps
            candidates.append((gain, rng.random(), site))
        
        if not candidates:
            break
        
        # Sort by gain (descending), then random
        candidates.sort(key=lambda c: (-c[0], c[1]))
        # Pick from top 3 with some probability
        k = min(3, len(candidates))
        if rng.random() < 0.7:
            site = candidates[0][2]
        elif rng.random() < 0.4:
            site = candidates[1][2]
        else:
            site = candidates[rng.randint(0, k-1)][2]
        
        chain.append(site)
        occ[site] = i
        contacts += 0  # Already accounted for in gain logic above conceptually, but let's track properly
        
    # Compute actual energy
    energy = _energy(seq, chain)
    return chain, energy

def _local_search(seq, chain, rng, time_budget):
    n = len(chain)
    is_h = [c == 'H' for c in seq]
    cur_chain = list(chain)
    cur_energy = _energy(seq, cur_chain)
    best_chain = list(cur_chain)
    best_energy = cur_energy
    
    T = 2.0
    alpha = 0.998
    min_T = 0.05
    
    # Phase 1: Greedy kink descent
    max_greedy = 2000 if n < 60 else 500
    for _ in range(max_greedy):
        if _time_left() < 3.0: break
        i = rng.randint(1, n-2)
        new_chain = _kink_move(cur_chain, i)
        if new_chain:
            new_e = _energy(seq, new_chain)
            if new_e < cur_energy:
                cur_chain = new_chain
                cur_energy = new_e
                if cur_energy < best_energy:
                    best_energy = cur_energy
                    best_chain = list(cur_chain)
    
    # Phase 2: SA with kinks and pivots
    while _time_left() > 3.0 and T > min_T:
        r = rng.random()
        new_chain = None
        delta = 0
        
        if r < 0.6:
            i = rng.randint(1, n-2)
            new_chain = _kink_move(cur_chain, i)
            if new_chain:
                new_e = _energy(seq, new_chain)
                delta = new_e - cur_energy
        elif r < 0.9 and n > 15:
            i = rng.randint(2, n-2)
            axis = rng.randint(0, 2)
            sign = rng.choice([-1, 1])
            new_chain = _pivot_move(cur_chain, i, axis, sign)
            if new_chain:
                new_e = _energy(seq, new_chain)
                delta = new_e - cur_energy
        else:
            # Reversal
            i = rng.randint(1, n-3)
            max_len = min(5, n-i-1)
            j = rng.randint(i+1, i+max_len)
            new_chain = None
            if i > 0:
                d1 = sum(abs(cur_chain[i-1][k]-cur_chain[j][k]) for k in range(3))
                if d1 == 1:
                    if j < n-1:
                        d2 = sum(abs(cur_chain[j+1][k]-cur_chain[i][k]) for k in range(3))
                        if d2 == 1:
                            nc = cur_chain[:i] + list(reversed(cur_chain[i:j+1])) + cur_chain[j+1:]
                            if len(set(nc)) == len(nc): new_chain = nc
                    else:
                        nc = cur_chain[:i] + list(reversed(cur_chain[i:j+1])) + cur_chain[j+1:]
                        if len(set(nc)) == len(nc): new_chain = nc
            else:
                if j < n-1:
                    d2 = sum(abs(cur_chain[j+1][k]-cur_chain[i][k]) for k in range(3))
                    if d2 == 1:
                        nc = cur_chain[:i] + list(reversed(cur_chain[i:j+1])) + cur_chain[j+1:]
                        if len(set(nc)) == len(nc): new_chain = nc
                else:
                    nc = cur_chain[:i] + list(reversed(cur_chain[i:j+1])) + cur_chain[j+1:]
                    if len(set(nc)) == len(nc): new_chain = nc
            if new_chain:
                new_e = _energy(seq, new_chain)
                delta = new_e - cur_energy
        
        if new_chain:
            if delta < 0 or rng.random() < math.exp(-delta/T):
                cur_chain = new_chain
                cur_energy += delta
                if cur_energy < best_energy:
                    best_energy = cur_energy
                    best_chain = list(cur_chain)
        T *= alpha
    
    return best_chain, best_energy

def _multi_start_fold(seq, dim):
    n = len(seq)
    best_chain = None
    best_energy = float('inf')
    
    # Number of restarts based on length
    if n <= 30:
        n_restarts = 8
    elif n <= 50:
        n_restarts = 6
    elif n <= 80:
        n_restarts = 4
    else:
        n_restarts = 3
    
    # Reduce if time is tight
    if _time_left() < 60:
        n_restarts = min(n_restarts, 3)
    if _time_left() < 20:
        n_restarts = 1
    
    for r in range(n_restarts):
        if _time_left() < 5.0: break
        
        seed = 12345 + r * 7919
        rng = random.Random(seed)
        
        # Grow a candidate
        chain, energy = _grow_greedy(seq, dim, rng, _time_left())
        
        if len(chain) < n:
            # Failed to complete growth, try a simpler fallback growth
            # For now, skip this restart
            continue
        
        # Local search
        opt_chain, opt_energy = _local_search(seq, chain, rng, _time_left())
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_chain = opt_chain
    
    if best_chain is None:
        # Fallback: straight line
        return [[i, 0, 0] for i in range(n)]
    
    return [list(p) for p in best_chain]

def fold(sequence, dimension):
    n = len(sequence)
    return _multi_start_fold(sequence, dimension)

# EVOLVE-BLOCK-END
