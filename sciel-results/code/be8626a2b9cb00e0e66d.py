"""HP lattice protein folding with multi-start growth and segment-reversal annealing."""
# EVOLVE-BLOCK-START
import random
import time
from math import exp


def _neigh(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _grow_from(seq, dim, rng, start):
    """Greedy chain growth from a given start, biased toward H-H contacts."""
    chain = [start]
    occ = {start: 0}
    for i in range(1, len(seq)):
        cands = []
        for s in _neigh(chain[-1]):
            if s in occ:
                continue
            g = 0
            if seq[i] == 'H':
                for q in _neigh(s):
                    if q in occ:
                        j = occ[q]
                        if j < i - 1 and seq[j] == 'H':
                            g += 1
            free = sum(1 for q in _neigh(s) if q not in occ)
            cands.append((g * 2.0 + 0.15 * free + rng.random() * 0.8, s))
        if not cands:
            return None
        cands.sort(key=lambda t: -t[0])
        # Pick from top with slight randomness
        k = min(len(cands), 3)
        chosen = rng.choice(cands[:k])[1]
        occ[chosen] = i
        chain.append(chosen)
    return chain


def _energy_of(seq, chain):
    """Compute HP contact energy (negative number of H-H non-adjacent contacts)."""
    pos = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if seq[i] == 'H':
            for q in _neigh(p):
                j = pos.get(q)
                if j is not None and j > i + 1 and seq[j] == 'H':
                    contacts += 1
    return -contacts


def _delta_rev(seq, chain, a, b):
    """Compute energy delta for reversing segment [a..b] (inclusive).
    Only need to check contacts involving positions at the boundaries."""
    # After reversal, position a gets old position b's point, etc.
    # We need to check:
    # 1. Contacts lost: H-H pairs where one is in [a..b] and the other outside,
    #    AND the pair was a contact (neighboring lattice sites, non-adjacent in chain)
    # 2. Contacts gained: new H-H pairs formed after reversal
    #
    # Simpler approach: compute energy of affected region before and after.
    
    # Positions in the reversed segment
    seg = list(range(a, b + 1))
    # The segment [a..b] will be reversed, so new chain has:
    # chain[:a] + chain[b:a-1:-1] + chain[b+1:]
    
    # We need to check all contacts involving any position in [a-1..b+1]
    # that are H-H and non-adjacent
    
    # Build a set of occupied positions for quick lookup
    pos_map = {p: i for i, p in enumerate(chain)}
    
    # Old contacts involving segment
    old_contacts = 0
    for i in range(max(0, a - 1), min(len(chain), b + 2)):
        if seq[i] != 'H':
            continue
        for q in _neigh(chain[i]):
            j = pos_map.get(q)
            if j is not None and j > i + 1 and seq[j] == 'H':
                # This is a contact. Count it if at least one index is in [a..b]
                if a <= i <= b or a <= j <= b:
                    old_contacts += 1
    
    # New chain after reversal
    new_chain = chain[:a] + chain[b:a-1:-1] + chain[b+1:]
    new_pos = {p: i for i, p in enumerate(new_chain)}
    
    # New contacts involving the reversed region
    new_contacts = 0
    for i in range(max(0, a - 1), min(len(new_chain), b + 2)):
        if seq[i] != 'H':
            continue
        for q in _neigh(new_chain[i]):
            j = new_pos.get(q)
            if j is not None and j > i + 1 and seq[j] == 'H':
                if a <= i <= b or a <= j <= b:
                    new_contacts += 1
    
    return new_contacts - old_contacts  # delta in contacts; energy delta is -delta


def _try_reverse(seq, chain, a, b):
    """Try reversing segment [a..b]. Return new chain if valid, else None."""
    if b - a < 2:  # need at least 3 residues for a meaningful reversal
        return None
    new_chain = chain[:a] + chain[b:a-1:-1] + chain[b+1:]
    # Check self-avoidance
    seen = set()
    for p in new_chain:
        if p in seen:
            return None
        seen.add(p)
    return new_chain


def _anneal(seq, chain, time_budget, rng):
    """Simulated annealing with segment reversal moves."""
    n = len(chain)
    current = list(chain)
    current_e = _energy_of(seq, current)
    best = list(current)
    best_e = current_e
    
    t_start = time.time()
    T = 1.0
    cooling = 0.995
    
    while time.time() - t_start < time_budget:
        # Choose random segment to reverse
        if n <= 4:
            a, b = 0, n - 1
        else:
            # Prefer shorter segments early, longer later
            max_len = min(n - 1, 2 + int((1.0 - T) * (n - 2)))
            a = rng.randint(0, n - 2)
            b = rng.randint(a + 2, min(n - 1, a + max_len))
        
        new_chain = _try_reverse(seq, current, a, b)
        if new_chain is None:
            T *= cooling
            continue
        
        delta_e = _energy_of(seq, new_chain) - current_e
        
        # Acceptance criterion
        if delta_e < 0 or (T > 0.01 and rng.random() < exp(-delta_e / T)):
            current = new_chain
            current_e = current_e + delta_e
            if current_e < best_e:
                best_e = current_e
                best = list(current)
        
        T *= cooling
        if T < 0.001:
            T = 0.001
    
    return best, best_e


def fold(sequence, dimension):
    """Fold HP sequence into low-energy self-avoiding conformation on square lattice."""
    t0 = time.time()
    deadline = t0 + 195.0  # Safety margin before 210s kill
    n = len(sequence)
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0  # 0 means no contacts (worst case for negative energy)
    
    # Phase 1: Generate diverse initial conformations via multi-start growth
    initial_chains = []
    max_growth = 80
    for trial in range(max_growth):
        if time.time() > deadline - 15:
            break
        # Random start position
        sx = rng.randint(-5, 5)
        sy = rng.randint(-5, 5)
        start = (sx, sy)
        chain = _grow_from(sequence, dimension, rng, start)
        if chain is not None:
            e = _energy_of(sequence, chain)
            initial_chains.append((e, chain))
            if e < best_energy:
                best_energy = e
                best_chain = list(chain)
    
    # Sort by energy, keep best ones for annealing
    initial_chains.sort(key=lambda x: x[0])
    top_chains = [c for _, c in initial_chains[:20]]
    
    # Phase 2: Anneal each initial conformation with allocated time
    remaining = deadline - time.time() - 5  # 5s safety buffer
    if remaining > 2 and top_chains:
        time_per = max(0.5, remaining / len(top_chains))
        for chain in top_chains:
            if time.time() > deadline - 3:
                break
            alloc = min(time_per, deadline - time.time() - 2)
            if alloc < 0.2:
                break
            annealed, ae = _anneal(sequence, chain, alloc, rng)
            if ae < best_energy:
                best_energy = ae
                best_chain = list(annealed)
    
    # Fallback: straight line if nothing worked
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
