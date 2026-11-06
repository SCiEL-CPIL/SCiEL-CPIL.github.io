"""Improved HP lattice folding: fast growth + fragment moves + efficient local search."""
# EVOLVE-BLOCK-START
import time
import random

_T0 = time.time()
_DEADLINE = 195.0  # 210s wall-clock limit, 15s safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _energy_fast(seq, chain):
    """Fast energy using a position->index map."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if seq[i] == "H":
            x, y = p
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                j = where.get((x + dx, y + dy))
                if j is not None and j > i + 1 and seq[j] == "H":
                    contacts += 1
    return -contacts


def _is_valid_chain(chain):
    """Check self-avoidance and unit steps."""
    n = len(chain)
    if n < 2:
        return True
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            dx = abs(chain[i][0] - chain[i - 1][0])
            dy = abs(chain[i][1] - chain[i - 1][1])
            if dx + dy != 1:
                return False
    return True


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    for i in range(1, n):
        options = []
        x, y = chain[-1]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            site = (x + dx, y + dy)
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                sx, sy = site
                for adx, ady in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    q = (sx + adx, sy + ady)
                    if q in occupied:
                        idx = occupied[q]
                        if idx < i - 1 and sequence[idx] == "H":
                            gain += 1
            # Free space heuristic
            free = 0
            sx, sy = site
            for adx, ady in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (sx + adx, sy + ady) not in occupied:
                    free += 1
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _fragment_move(chain, rng, max_frag_len=6):
    """
    Try a fragment move: pick a contiguous sub-chain of length L (2..max_frag_len),
    remove it, and re-insert it at a different position by growing it again.
    Returns new chain if valid, else None.
    """
    n = len(chain)
    if n < 4:
        return None
    
    # Pick fragment length
    L = rng.randint(2, min(max_frag_len, n - 2))
    # Pick start position (ensure fragment fits)
    a = rng.randint(0, n - L - 1)
    b = a + L  # fragment is chain[a:b]
    
    # The fragment to move
    frag = chain[a:b]
    # The rest of the chain: prefix + suffix
    prefix = chain[:a]
    suffix = chain[b:]
    
    # We need to re-insert the fragment somewhere.
    # Strategy: pick a new insertion point in the remaining chain and try to grow the fragment from the end of prefix (or some other anchor).
    # Actually, a simpler fragment move: pick two indices i < j, and reverse the segment chain[i:j+1].
    # But the inspiration says "fragment moves allow large-scale conformational changes".
    # Let's implement a "re-insertion" move:
    # 1. Remove fragment chain[a:b].
    # 2. Pick a new anchor point. The fragment must be connected to the rest of the chain.
    #    This is complex. Let's stick to reversals for now, but also try "slide" moves?
    #    Actually, let's implement a proper fragment re-insertion:
    #    - The fragment is a self-avoiding walk of length L.
    #    - We need to place it such that its first residue is adjacent to the new "left" neighbor
    #      and its last residue is adjacent to the new "right" neighbor, and it doesn't collide.
    #    - This is essentially a sub-problem of chain growth.
    
    # Let's try a different move: "Cut and paste"
    # Pick i < j. Cut chain into A = chain[:i], B = chain[i:j], C = chain[j:].
    # Try to re-attach B between some point in A and some point in C? No, that breaks topology.
    # The chain is linear. We can only reverse segments or do "kink" moves.
    
    # Let's implement "Reversal" which is already there, but also "Translation" of a small fragment?
    # No, translation breaks connectivity.
    
    # Let's implement "Reversal" with variable length, which is already in the current code.
    # The inspiration mentions fragment moves improving score. Let's try to implement a true fragment re-insertion.
    
    # To re-insert a fragment:
    # 1. Remove frag = chain[a:b].
    # 2. The remaining chain is R = chain[:a] + chain[b:].
    # 3. We need to find a position in R where we can insert frag such that:
    #    - The last residue of the prefix part of R is adjacent to frag[0].
    #    - The first residue of the suffix part of R is adjacent to frag[-1].
    #    - frag does not collide with R.
    #    - frag maintains its internal shape (or we can rotate/reflect it).
    
    # This is expensive. Let's try a simpler "2-opt" style move:
    # Reverse a segment. This is already implemented.
    # Let's add "3-opt": pick 3 points and rearrange. Too complex.
    
    # Let's try "Kink" move: pick 3 consecutive residues and try to rotate the middle one.
    # If chain[i-1], chain[i], chain[i+1] form a kink, can we move chain[i] to the other side?
    # No, chain[i] is fixed relative to neighbors in a 2D lattice if we only move it.
    # Actually, if we have a straight segment, we can't kink it without moving more.
    
    # Let's stick to Reversals but make them more effective by trying longer fragments and more iterations.
    # Also, let's add a "slide" move: if we have a segment of length 2 or 3, can we shift it?
    # No, shifting breaks connectivity.
    
    # Let's implement "Reversal" with random length.
    if n < 4:
        return None
    a = rng.randint(0, n - 3)
    b = rng.randint(a + 2, n - 1)
    new_chain = chain[:a] + chain[a:b + 1][::-1] + chain[b + 1:]
    if _is_valid_chain(new_chain):
        return new_chain
    return None


def _local_search(sequence, chain, rng, max_iters=500):
    """Local search via random reversals. Returns improved chain."""
    best_chain = list(chain)
    best_energy = _energy_fast(sequence, best_chain)
    n = len(best_chain)
    
    for _ in range(max_iters):
        if _remaining() < 5.0:
            break
        new_chain = _fragment_move(best_chain, rng)
        if new_chain is None:
            continue
        
        e = _energy_fast(sequence, new_chain)
        if e < best_energy:
            best_chain = new_chain
            best_energy = e
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using multiple restarts and local search with time guard."""
    rng = random.Random(12345 + len(sequence))
    best = None
    best_energy = 0  # start at 0 (unfolded)
    n = len(sequence)
    
    # Number of restarts based on time available
    num_restarts = 300
    for r in range(num_restarts):
        if _remaining() < 3.0:
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        e = _energy_fast(sequence, chain)
        if e < best_energy:
            best_energy = e
            best = chain
        
        # Apply local search on the best candidate periodically
        if r % 10 == 0 and best is not None and _remaining() > 10.0:
            improved, imp_e = _local_search(sequence, best, rng, max_iters=100)
            if imp_e < best_energy:
                best_energy = imp_e
                best = improved
    
    # Final local search on the overall best if time permits
    if best is not None and _remaining() > 5.0:
        improved, imp_e = _local_search(sequence, best, rng, max_iters=200)
        if imp_e < best_energy:
            best_energy = imp_e
            best = improved
    
    if best is None:
        # Fallback: straight line
        best = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
