"""Combined HP lattice folding: bond-vector growth + SA with segment moves."""
# EVOLVE-BLOCK-START
import random
import time
import math

_TOTAL_START = time.time()
_HARD_LIMIT = 300.0
_SAFETY = 15.0


def _remaining():
    return _TOTAL_START + _HARD_LIMIT - _SAFETY - time.time()


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _neigh_set(point):
    return set(_neighbours(point))


def _grow(sequence, dimension, rng, bias=1.0):
    """Randomized greedy growth with hydrophobic contact bias."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = bias * gain + 0.1 * free + rng.random() * 0.8
            options.append((score, site))
        if not options:
            return None
        options.sort(reverse=True)
        k = min(3, len(options))
        site = options[rng.randint(0, k - 1)][1]
        occupied[site] = i
        chain.append(site)
    return chain


def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _local_search(sequence, chain, rng, deadline):
    """Simulated annealing with single-residue and segment-reversal moves."""
    n = len(chain)
    if n < 3:
        return chain, _energy(sequence, chain)

    best = list(chain)
    current = list(chain)
    best_e = _energy(sequence, best)
    current_e = best_e
    occupied = {p: i for i, p in enumerate(current)}
    h_mask = [1 if c == "H" else 0 for c in sequence]

    temp = 4.0
    min_temp = 0.02
    cooling = 0.997
    no_improve = 0
    best_no_improve = 0

    def delta_single(i, old, new):
        d = 0
        if h_mask[i]:
            for q in _neighbours(old):
                j = occupied.get(q)
                if j is not None and j != i and h_mask[j]:
                    d -= 1
            for q in _neighbours(new):
                j = occupied.get(q)
                if j is not None and j != i and h_mask[j]:
                    d += 1
        return d

    def try_single(i):
        old = current[i]
        prev = current[i - 1]
        nxt = current[i + 1]
        cands = set()
        for s in _neighbours(prev):
            if s in _neigh_set(nxt):
                if (s not in occupied or occupied[s] == i) and s != old:
                    cands.add(s)
        if not cands:
            return None
        for site in cands:
            d = delta_single(i, old, site)
            if d > 0 or rng.random() < math.exp(-d / max(temp, 1e-9)):
                occupied[old] = -1
                occupied[site] = i
                current[i] = site
                return d
        return None

    def try_reversal(i, length):
        if i < 1 or i + length >= n:
            return None
        end = i + length - 1
        a = current[i - 1]
        b = current[i]
        d_end = current[end]
        e = current[end + 1] if end + 1 < n else None
        # Check connectivity after reversal
        if sum(abs(x - y) for x, y in zip(a, d_end)) != 1:
            return None
        if e is not None and sum(abs(x - y) for x, y in zip(b, e)) != 1:
            return None
        seg = current[i:end + 1]
        rev = list(reversed(seg))
        # Remove from occupied
        for idx in range(i, end + 1):
            occupied[current[idx]] = -1
        # Compute delta
        d = 0
        for k in range(i, end + 1):
            old_site = seg[k - i]
            new_site = rev[k - i]
            if old_site == new_site:
                continue
            if h_mask[k]:
                for q in _neighbours(old_site):
                    j = occupied.get(q)
                    if j is not None and j != k and h_mask[j]:
                        d -= 1
                for q in _neighbours(new_site):
                    j = occupied.get(q)
                    if j is not None and j != k and h_mask[j]:
                        d += 1
        # Apply
        for k in range(i, end + 1):
            occupied[rev[k - i]] = k
            current[k] = rev[k - i]
        return d

    while time.time() < deadline and temp > min_temp:
        moved = False
        # Single moves
        for i in range(1, n - 1):
            if time.time() >= deadline:
                break
            d = try_single(i)
            if d is not None:
                current_e -= d
                if current_e < best_e:
                    best_e = current_e
                    best = list(current)
                    no_improve = 0
                else:
                    no_improve += 1
                moved = True
                break
        # Segment reversals
        if not moved:
            max_len = min(5, n // 3)
            for length in range(2, max_len + 1):
                if time.time() >= deadline:
                    break
                for i in range(1, n - length):
                    d = try_reversal(i, length)
                    if d is not None:
                        current_e -= d
                        if current_e < best_e:
                            best_e = current_e
                            best = list(current)
                            no_improve = 0
                        else:
                            no_improve += 1
                        moved = True
                        break
                if moved:
                    break

        temp *= cooling

        if no_improve > 150:
            current = list(best)
            current_e = best_e
            occupied = {p: i for i, p in enumerate(current)}
            no_improve = 0
            temp = max(temp * 2.5, 0.5)
            best_no_improve += 1
            if best_no_improve > 5:
                break

    return best, best_e


def fold(sequence, dimension):
    """Fold a sequence on the cubic lattice using growth + SA."""
    seq_len = len(sequence)
    # Per-sequence budget: aim for ~10s average over 26 sequences
    per_seq = min(10.0, max(2.0, _remaining() / max(1, 26)))
    deadline = time.time() + per_seq

    rng = random.Random(12345 + seq_len)
    best_chain = None
    best_energy = 0

    # Phase 1: Multiple growths with varying bias
    chains = []
    for _ in range(8):
        if time.time() > deadline - 3.0:
            break
        bias = 0.5 + rng.random() * 1.5
        chain = _grow(sequence, dimension, rng, bias)
        if chain is not None:
            e = _energy(sequence, chain)
            chains.append((e, chain))
            if best_chain is None or e < best_energy:
                best_chain = chain
                best_energy = e

    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(seq_len)]
        best_energy = _energy(sequence, best_chain)

    # Phase 2: SA on best initial
    if time.time() < deadline - 1.0:
        best_chain, best_energy = _local_search(
            sequence, best_chain, rng, deadline - 0.5
        )

    # Phase 3: SA on second-best if time remains
    if len(chains) > 1:
        chains.sort(key=lambda x: x[0])
        if time.time() < deadline - 2.0:
            e2, ch2 = chains[1]
            if ch2 != best_chain:
                ch2, e2 = _local_search(sequence, ch2, rng, deadline - 0.5)
                if e2 < best_energy:
                    best_chain, best_energy = ch2, e2

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
