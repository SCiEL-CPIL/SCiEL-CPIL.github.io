"""HP lattice protein folding on the cubic lattice.
Move-based Monte Carlo / simulated annealing with tail moves, reversals,
and tail regrowth, using a spatial hash and incremental energy updates.
"""
import random
import time
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # Safety margin before 330s kill


def _energy(sequence, chain):
    """Compute total energy: negative count of non-adjacent H-H contacts."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i in range(n):
        if sequence[i] != "H":
            continue
        x, y, z = chain[i]
        for dx, dy, dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
            q = (x+dx, y+dy, z+dz)
            j = where.get(q)
            if j is not None and j > i + 1:
                if sequence[j] == "H":
                    contacts += 1
    return -contacts


def _contact_set(sequence, chain):
    """Return a set of (i, j) pairs (i<j, j>i+1) representing H-H contacts."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = set()
    for i in range(n):
        if sequence[i] != "H":
            continue
        x, y, z = chain[i]
        for dx, dy, dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
            q = (x+dx, y+dy, z+dz)
            j = where.get(q)
            if j is not None and j > i + 1 and sequence[j] == "H":
                contacts.add((i, j))
    return contacts


def _initial_chain(sequence, rng):
    """Generate a self-avoiding random walk (greedy with randomization)."""
    n = len(sequence)
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    for i in range(1, n):
        x, y, z = chain[-1]
        candidates = []
        for dx, dy, dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
            site = (x+dx, y+dy, z+dz)
            if site not in occupied:
                gain = 0
                if sequence[i] == "H":
                    for ddx, ddy, ddz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                        q = (site[0]+ddx, site[1]+ddy, site[2]+ddz)
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == "H":
                            gain += 1
                candidates.append((gain + rng.random() * 0.5, site))
        if not candidates:
            x, y, z = chain[-1]
            for k in range(i, n):
                site = (x + (k - i + 1), y, z)
                chain.append(site)
            break
        candidates.sort(key=lambda t: -t[0])
        site = candidates[0][1]
        occupied[site] = i
        chain.append(site)
    return chain


def _try_move_tail(sequence, chain, where, rng, contacts):
    """Try moving the C-terminus to a neighboring free site."""
    n = len(chain)
    i = n - 1
    x, y, z = chain[i]
    candidates = []
    for dx, dy, dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
        site = (x+dx, y+dy, z+dz)
        if site not in where:
            candidates.append(site)
    if not candidates:
        return None
    site = rng.choice(candidates)
    old_contacts = set()
    for (a, b) in contacts:
        if a == i or b == i:
            old_contacts.add((a, b))
    new_contacts = set()
    for dx, dy, dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
        q = (site[0]+dx, site[1]+dy, site[2]+dz)
        j = where.get(q)
        if j is not None and abs(j - i) > 1:
            if sequence[i] == "H" and sequence[j] == "H":
                if j < i:
                    new_contacts.add((j, i))
                else:
                    new_contacts.add((i, j))
    delta = len(new_contacts) - len(old_contacts)
    new_chain = list(chain)
    new_chain[i] = site
    new_where = dict(where)
    del new_where[chain[i]]
    new_where[site] = i
    new_contact_set = (contacts - old_contacts) | new_contacts
    return new_chain, new_where, new_contact_set, delta


def _try_move_reverse(sequence, chain, where, rng, contacts, min_len=2, max_len=12):
    """Try reversing a sub-chain of length L."""
    n = len(chain)
    if n < 3:
        return None
    max_len = min(max_len, n - 1)
    L = rng.randint(min_len, max_len)
    start = rng.randint(0, n - L - 1)
    end = start + L
    if start > 0:
        if abs(chain[start-1][0]-chain[end][0]) + abs(chain[start-1][1]-chain[end][1]) + abs(chain[start-1][2]-chain[end][2]) != 1:
            return None
    if end < n - 1:
        if abs(chain[start][0]-chain[end+1][0]) + abs(chain[start][1]-chain[end+1][1]) + abs(chain[start][2]-chain[end+1][2]) != 1:
            return None
    new_chain = list(chain)
    new_chain[start:end+1] = new_chain[start:end+1][::-1]
    new_where = {}
    for idx, p in enumerate(new_chain):
        new_where[p] = idx
    new_contacts = _contact_set(sequence, new_chain)
    delta = len(new_contacts) - len(contacts)
    return new_chain, new_where, new_contacts, delta


def _try_tail_regrow(sequence, chain, where, rng, contacts, regrow_len=10):
    """Regrow the last regrow_len residues greedily with randomization."""
    n = len(chain)
    if n <= regrow_len:
        return None
    cut = n - regrow_len
    prefix = chain[:cut]
    prefix_where = {p: i for i, p in enumerate(prefix)}
    # Remove contacts involving indices >= cut
    old_contacts = set()
    for (a, b) in contacts:
        if a >= cut or b >= cut:
            old_contacts.add((a, b))
    # Regrow
    current = list(prefix)
    occupied = dict(prefix_where)
    for i in range(cut, n):
        x, y, z = current[-1]
        candidates = []
        for dx, dy, dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
            site = (x+dx, y+dy, z+dz)
            if site not in occupied:
                gain = 0
                if sequence[i] == "H":
                    for ddx, ddy, ddz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                        q = (site[0]+ddx, site[1]+ddy, site[2]+ddz)
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == "H":
                            gain += 1
                candidates.append((gain + rng.random() * 1.0, site))
        if not candidates:
            return None
        candidates.sort(key=lambda t: -t[0])
        site = candidates[0][1]
        occupied[site] = i
        current.append(site)
    new_contacts = _contact_set(sequence, current)
    delta = len(new_contacts) - len(contacts)
    return current, occupied, new_contacts, delta


def _sa_search(sequence, chain, rng, time_budget=60.0):
    """Simulated annealing search using tail moves, reversals, and tail regrowth."""
    n = len(sequence)
    current_chain = list(chain)
    current_where = {p: i for i, p in enumerate(current_chain)}
    current_contacts = _contact_set(sequence, current_chain)
    current_energy = -len(current_contacts)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    best_contacts = set(current_contacts)
    
    start_time = time.time()
    T = 2.0
    cooling = 0.9998
    min_T = 0.05
    no_improve = 0
    
    while time.time() - start_time < time_budget:
        if time.time() > _DEADLINE - 3:
            break
        
        move_type = rng.random()
        result = None
        if move_type < 0.5:
            result = _try_move_tail(sequence, current_chain, current_where, rng, current_contacts)
        elif move_type < 0.8:
            result = _try_move_reverse(sequence, current_chain, current_where, rng, current_contacts)
        else:
            result = _try_tail_regrow(sequence, current_chain, current_where, rng, current_contacts, regrow_len=rng.randint(5, 15))
        
        if result is None:
            continue
        
        new_chain, new_where, new_contacts, delta = result
        new_energy = -len(new_contacts)
        
        if new_energy < best_energy:
            best_chain = list(new_chain)
            best_energy = new_energy
            best_contacts = set(new_contacts)
            no_improve = 0
        else:
            no_improve += 1
        
        if new_energy <= current_energy:
            accept = True
        else:
            delta_e = new_energy - current_energy
            if T > min_T:
                accept = rng.random() < math.exp(-delta_e / T)
            else:
                accept = False
        
        if accept:
            current_chain = new_chain
            current_where = new_where
            current_contacts = new_contacts
            current_energy = new_energy
            T = max(min_T, T * cooling)
        else:
            T = max(min_T, T * cooling)
        
        if no_improve > 10000:
            current_chain = list(best_chain)
            current_where = {p: i for i, p in enumerate(current_chain)}
            current_contacts = set(best_contacts)
            current_energy = best_energy
            T = 1.5
            no_improve = 0
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using move-based SA with deadline guard."""
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)
    
    best_chain = None
    best_energy = 0
    num_initial = min(15, max(5, int((_DEADLINE - time.time()) * 0.1)))
    
    for trial in range(num_initial):
        if time.time() > _DEADLINE - 60:
            break
        chain = _initial_chain(sequence, rng)
        if len(chain) != n:
            continue
        e = _energy(sequence, chain)
        if e < best_energy:
            best_chain = chain
            best_energy = e
    
    if best_chain is None:
        best_chain = _initial_chain(sequence, rng)
        best_energy = _energy(sequence, best_chain)
    
    remaining = _DEADLINE - time.time() - 5
    if remaining > 10:
        best_chain, best_energy = _sa_search(sequence, best_chain, rng, time_budget=remaining)
    
    return [list(p) for p in best_chain]
