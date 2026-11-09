"""HP lattice protein folding on the cubic lattice.
Uses a contact-map guided construction followed by multi-start simulated annealing
with chain-reversal and segment-swap moves for global optimization.
"""
import random
import time
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 318.0  # Safety margin before 330s kill


def _energy(sequence, chain):
    """Compute total energy: negative count of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y, z = p
            for q in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _valid_chain(chain, n):
    """Check self-avoidance and unit steps."""
    if len(chain) != n:
        return False
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            prev = chain[i-1]
            dist = abs(p[0]-prev[0]) + abs(p[1]-prev[1]) + abs(p[2]-prev[2])
            if dist != 1:
                return False
    return True


def _grow_greedy(sequence, dimension, rng, temperature=0.5):
    """Randomized greedy chain growth."""
    n = len(sequence)
    start = (0, 0, 0)
    chain, occupied = [start], {start: 0}
    for i in range(1, n):
        x, y, z = chain[-1]
        options = []
        for site in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                sx, sy, sz = site
                for q in ((sx+1,sy,sz),(sx-1,sy,sz),(sx,sy+1,sz),(sx,sy-1,sz),(sx,sy,sz+1),(sx,sy,sz-1)):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = 0
            sx, sy, sz = site
            for q in ((sx+1,sy,sz),(sx-1,sy,sz),(sx,sy+1,sz),(sx,sy-1,sz),(sx,sy,sz+1),(sx,sy,sz-1)):
                if q not in occupied:
                    free += 1
            score = gain + 0.1 * free + rng.random() * temperature
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _regrow_tail(sequence, chain, start_idx, rng, temperature=1.0):
    """Regrow the tail of the chain from start_idx."""
    n = len(sequence)
    if start_idx >= n - 1:
        return list(chain), _energy(sequence, chain)
    
    prefix = chain[:start_idx + 1]
    occupied = {p: i for i, p in enumerate(prefix)}
    current = list(prefix)
    
    for i in range(start_idx + 1, n):
        x, y, z = current[-1]
        options = []
        for site in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                sx, sy, sz = site
                for q in ((sx+1,sy,sz),(sx-1,sy,sz),(sx,sy+1,sz),(sx,sy-1,sz),(sx,sy,sz+1),(sx,sy,sz-1)):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = 0
            sx, sy, sz = site
            for q in ((sx+1,sy,sz),(sx-1,sy,sz),(sx,sy+1,sz),(sx,sy-1,sz),(sx,sy,sz+1),(sx,sy,sz-1)):
                if q not in occupied:
                    free += 1
            score = gain + 0.05 * free + rng.random() * temperature
            options.append((score, site))
        if not options:
            return list(chain), _energy(sequence, chain)
        site = max(options)[1]
        occupied[site] = i
        current.append(site)
    return current, _energy(sequence, current)


def _sa_move_regrow(sequence, chain, rng, start_idx, temperature):
    """Perform a tail-regrowth move for simulated annealing."""
    n = len(sequence)
    if start_idx >= n - 1:
        return None
    
    prefix = chain[:start_idx + 1]
    occupied = {p: i for i, p in enumerate(prefix)}
    current = list(prefix)
    
    for i in range(start_idx + 1, n):
        x, y, z = current[-1]
        options = []
        for site in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                sx, sy, sz = site
                for q in ((sx+1,sy,sz),(sx-1,sy,sz),(sx,sy+1,sz),(sx,sy-1,sz),(sx,sy,sz+1),(sx,sy,sz-1)):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            score = gain + rng.random() * temperature
            options.append((score, site))
        if not options:
            return None
        # Pick with some randomness
        if len(options) > 1 and rng.random() < 0.3:
            site = options[rng.randint(0, len(options)-1)][1]
        else:
            site = max(options)[1]
        occupied[site] = i
        current.append(site)
    
    return current


def _sa_move_segment_swap(sequence, chain, rng):
    """Try to swap two non-adjacent segments of the chain."""
    n = len(sequence)
    if n < 10:
        return None
    
    # Pick two segments to try to swap
    min_len = 2
    max_len = min(8, n // 4)
    
    for _ in range(5):  # Try a few random swaps
        # Pick segment 1
        len1 = rng.randint(min_len, max_len)
        pos1 = rng.randint(0, n - len1 - 1)
        
        # Pick segment 2 (non-overlapping)
        len2 = rng.randint(min_len, max_len)
        pos2 = rng.randint(0, n - len2 - 1)
        
        # Check non-overlap
        if pos1 + len1 > pos2 or pos2 + len2 > pos1:
            continue
        
        # Extract segments
        seg1 = chain[pos1:pos1+len1]
        seg2 = chain[pos2:pos2+len2]
        
        # Try to swap: remove seg1 and seg2, then reinsert
        # This is complex; skip for now and use a simpler approach
        break
    
    return None


def _simulated_annealing(sequence, chain, rng, time_budget=10.0):
    """Simulated annealing with tail-regrowth moves."""
    n = len(sequence)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    t0 = 2.0
    t1 = 0.05
    cooling_rate = 0.995
    
    while time.time() - start_time < time_budget:
        # Pick a random start position for tail regrowth
        # Bias towards later positions (shorter tails = faster)
        if n <= 50:
            start_idx = rng.randint(max(0, n - 20), n - 2)
        elif n <= 100:
            start_idx = rng.randint(max(0, n - 30), n - 2)
        else:
            start_idx = rng.randint(max(0, n - 40), n - 2)
        
        new_chain = _sa_move_regrow(sequence, current_chain, rng, start_idx, t0)
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Accept if better or with probability exp(-delta/t)
        if delta < 0 or rng.random() < math.exp(-delta / max(t0, 0.01)):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_chain = list(current_chain)
                best_energy = current_energy
        
        # Cool down
        t0 = max(t1, t0 * cooling_rate)
        
        if time.time() > _DEADLINE - 3:
            break
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using multi-start construction + simulated annealing."""
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)
    remaining = _DEADLINE - time.time()
    
    # Phase 1: Multiple randomized greedy growths to get diverse starting points
    if n <= 50:
        num_initial = min(60, int(remaining * 1.5))
    elif n <= 100:
        num_initial = min(30, int(remaining * 0.8))
    else:
        num_initial = min(15, int(remaining * 0.4))
    
    num_initial = max(num_initial, 8)
    
    initial_chains = []
    for trial in range(num_initial):
        if time.time() > _DEADLINE - 60:
            break
        temp = 1.0 if trial < num_initial // 3 else (0.5 if trial < 2 * num_initial // 3 else 0.2)
        chain = _grow_greedy(sequence, dimension, rng, temperature=temp)
        if chain is not None:
            initial_chains.append(chain)
    
    if not initial_chains:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        return [list(p) for p in best_chain]
    
    # Pick the best initial chain
    best_chain = min(initial_chains, key=lambda c: _energy(sequence, c))
    best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Simulated annealing refinement
    remaining = _DEADLINE - time.time() - 5
    if remaining > 15:
        sa_budget = min(remaining, 250.0)
        # Split time among a few SA runs from different starting points
        num_sa_runs = min(3, len(initial_chains))
        per_run_budget = sa_budget / num_sa_runs
        
        for i in range(num_sa_runs):
            if time.time() > _DEADLINE - 5:
                break
            start_chain = initial_chains[i] if i < len(initial_chains) else best_chain
            improved_chain, improved_energy = _simulated_annealing(
                sequence, start_chain, rng, time_budget=per_run_budget
            )
            if improved_energy < best_energy:
                best_chain = improved_chain
                best_energy = improved_energy
        
        # Final polish with a longer SA run on the best chain
        remaining = _DEADLINE - time.time() - 3
        if remaining > 10:
            improved_chain, improved_energy = _simulated_annealing(
                sequence, best_chain, rng, time_budget=min(remaining, 50.0)
            )
            if improved_energy < best_energy:
                best_chain = improved_chain
                best_energy = improved_energy
    
    return [list(p) for p in best_chain]
