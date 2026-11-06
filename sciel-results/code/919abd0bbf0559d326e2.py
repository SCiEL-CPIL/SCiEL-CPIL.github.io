"""HP lattice protein folding: beam search growth + local refinement with deadline guard."""
# EVOLVE-BLOCK-START
import time
import random

_T0 = time.time()
_DEADLINE = 195.0  # 210s wall-clock limit, 15s safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _neighbours2d(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _energy(seq, chain):
    """Compute energy: -number of H-H non-bonded contacts."""
    n = len(chain)
    if n < 3:
        return 0
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    h_indices = set(i for i, c in enumerate(seq) if c == "H")
    contacts = 0
    for i in h_indices:
        x, y = chain[i]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = pos_to_idx.get((x + dx, y + dy))
            if j is not None and j > i + 1 and seq[j] == "H":
                contacts += 1
    return -contacts


def _beam_grow(seq, dimension, beam_width, rng):
    """Beam search chain growth with efficient index tracking."""
    n = len(seq)
    if n <= 1:
        return [(0, 0)] if n == 1 else []
    
    # Each beam: (energy, chain, pos_to_idx)
    # We'll track pos_to_idx as a dict for O(1) lookups
    start = (0, 0)
    beams = [(0, [start], {start: 0})]
    
    for i in range(1, n):
        new_beams = []
        for energy, chain, p2i in beams:
            last = chain[-1]
            for site in _neighbours2d(last):
                if site in p2i:
                    continue
                # Incremental energy: only check if new residue is H
                delta_e = 0
                if seq[i] == "H":
                    for nb in _neighbours2d(site):
                        j = p2i.get(nb)
                        if j is not None and j < i - 1 and seq[j] == "H":
                            delta_e -= 1
                new_energy = energy + delta_e
                new_chain = chain + [site]
                new_p2i = dict(p2i)
                new_p2i[site] = i
                new_beams.append((new_energy, new_chain, new_p2i))
        
        if not new_beams:
            return None
        
        # Sort by energy (most negative first), keep top beam_width
        new_beams.sort(key=lambda x: x[0])
        beams = new_beams[:beam_width]
    
    if not beams:
        return None
    
    return beams[0][1]


def _local_refine(seq, chain, rng, max_steps=300):
    """Local refinement: try relocating individual residues to reduce energy."""
    n = len(seq)
    if n < 3:
        return chain
    
    current_energy = _energy(seq, chain)
    best_chain = list(chain)
    best_energy = current_energy
    occupied = {p: i for i, p in enumerate(chain)}
    
    steps = 0
    no_improve = 0
    while steps < max_steps and _remaining() > 5.0 and no_improve < 100:
        steps += 1
        # Pick a random interior residue
        i = rng.randint(1, n - 2)
        old_pos = chain[i]
        
        improved = False
        for new_pos in _neighbours2d(old_pos):
            if new_pos in occupied:
                continue
            
            test_chain = list(chain)
            test_chain[i] = new_pos
            test_energy = _energy(seq, test_chain)
            
            if test_energy < best_energy:
                best_energy = test_energy
                best_chain = test_chain
                chain = test_chain
                occupied[new_pos] = i
                del occupied[old_pos]
                current_energy = test_energy
                improved = True
                no_improve = 0
                break
            elif test_energy <= current_energy and rng.random() < 0.05:
                # Accept non-worsening move occasionally
                chain = test_chain
                occupied[new_pos] = i
                del occupied[old_pos]
                current_energy = test_energy
                improved = True
                no_improve = 0
                break
        
        if not improved:
            no_improve += 1
    
    return best_chain


def _random_walk(seq, dimension, rng):
    """Generate a random self-avoiding walk."""
    n = len(seq)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = [p for p in _neighbours2d(last) if p not in occupied]
        if not candidates:
            return None
        site = rng.choice(candidates)
        chain.append(site)
        occupied[site] = i
    
    return chain


def fold(sequence, dimension):
    """Best conformation using beam search + local refinement with deadline guard."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    rng = random.Random(42 + n)
    
    # Baseline: straight line
    best_chain = [(i, 0) for i in range(n)]
    best_energy = _energy(sequence, best_chain)
    
    # Phase 1: Beam search with various widths
    for beam_width in [30, 50, 100]:
        if _remaining() < 15.0:
            break
        
        for restart in range(5):
            if _remaining() < 15.0:
                break
            
            chain = _beam_grow(sequence, dimension, beam_width, rng)
            if chain is None or len(chain) != n:
                continue
            
            energy = _energy(sequence, chain)
            
            # Local refinement
            if _remaining() > 20.0:
                refined = _local_refine(sequence, chain, rng, max_steps=200)
                refined_energy = _energy(sequence, refined)
                if refined_energy < energy:
                    chain = refined
                    energy = refined_energy
            
            if energy < best_energy:
                best_energy = energy
                best_chain = chain
    
    # Phase 2: Random walks + refinement if time remains
    while _remaining() > 25.0:
        chain = _random_walk(sequence, dimension, rng)
        if chain is None or len(chain) != n:
            continue
        
        energy = _energy(sequence, chain)
        
        if _remaining() > 15.0:
            refined = _local_refine(sequence, chain, rng, max_steps=150)
            refined_energy = _energy(sequence, refined)
            if refined_energy < energy:
                chain = refined
                energy = refined_energy
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
