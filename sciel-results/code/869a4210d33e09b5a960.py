"""HP lattice protein folding: diverse init + fast incremental local search
with 2-opt segment reversal and serpentine template seeding."""
# EVOLVE-BLOCK-START
import random
import time

def _neighbours(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))

def _energy(sequence, chain):
    """Full recompute of HP contact energy (used for final verification)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _energy_fast(sequence, chain):
    """Energy with a precomputed position->index map (avoids repeated rebuilds)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _greedy_grow(sequence, dimension, rng, temperature=1.0):
    """Randomized greedy chain growth. Prefers sites creating H-H contacts,
    then free space, then randomness."""
    n = len(sequence)
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}

    for i in range(1, n):
        candidates = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            gain += 1

            # look-ahead: can the next residue also make a contact here?
            future_gain = 0
            if i + 1 < n and sequence[i + 1] == 'H':
                for q in _neighbours(site):
                    if q not in occupied:
                        for r in _neighbours(q):
                            if r in occupied:
                                j = occupied[r]
                                if j < i - 1 and sequence[j] == 'H':
                                    future_gain += 1
                        break

            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1

            score = gain * 10.0 + future_gain * 3.0 + free * 0.3 + rng.random() * temperature
            candidates.append((score, site))

        if not candidates:
            return None

        candidates.sort(key=lambda x: -x[0])
        if rng.random() < 0.8:
            best_site = candidates[0][1]
        else:
            best_site = rng.choice(candidates[:max(1, len(candidates) // 3)])[1]

        chain.append(best_site)
        occupied[best_site] = i

    return chain

def _serpentine_template(n):
    """Compact self-avoiding serpentine walk of length n (a dense starting point)."""
    pts = []
    # grow a 2-row serpentine: (0,0)->(1,0)->(1,1)->(0,1)->(0,2)->(1,2)->...
    x, y = 0, 0
    pts.append((x, y))
    direction = (1, 0)  # start moving +x
    for _ in range(1, n):
        x += direction[0]
        y += direction[1]
        pts.append((x, y))
        # turn: alternate between +x/-x runs of length 1, stepping +y each pair
        # simpler: snake by columns of height 2
        if (len(pts) % 2) == 0:
            direction = (0, 1) if direction[0] == 1 else (-1, 0)
        else:
            direction = (1, 0) if direction[0] == -1 else (0, 1)
    # verify self-avoidance; if failed, fall back to straight line
    if len(set(pts)) != n:
        pts = [(i, 0) for i in range(n)]
    return pts

def _diverse_init(sequence, n, rng):
    chains = []
    for temp in [0.1, 0.5, 1.0, 2.0]:
        c = _greedy_grow(sequence, 2, rng, temperature=temp)
        if c is not None and len(c) == n:
            chains.append(c)
    for _ in range(3):
        chain = [(0, 0)]
        occupied = {(0, 0)}
        ok = True
        for i in range(1, n):
            last = chain[-1]
            opts = [s for s in _neighbours(last) if s not in occupied]
            if not opts:
                ok = False
                break
            site = rng.choice(opts)
            chain.append(site)
            occupied.add(site)
        if ok and len(chain) == n:
            chains.append(chain)
    serp = _serpentine_template(n)
    chains.append(serp)
    return chains

def _try_move(chain, i, new_pos, where):
    """Can residue i move to new_pos? (i is interior, new_pos free, still adjacent)."""
    if new_pos in where:
        return False
    prev = chain[i - 1]
    nxt = chain[i + 1]
    if abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) != 1:
        return False
    if abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) != 1:
        return False
    return True

def _local_search(sequence, chain, max_iters=200, rng=None, time_limit=None):
    """Local search with single-residue moves and 2-opt segment reversal,
    using an incrementally-maintained position map for fast energy checks."""
    if rng is None:
        rng = random.Random()
    n = len(sequence)
    if n < 3:
        return chain, _energy_fast(sequence, chain)

    chain = list(chain)
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy_fast(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    start_time = time.time()
    iters = 0
    no_improve = 0

    while iters < max_iters:
        if time_limit is not None and time.time() > time_limit:
            break
        iters += 1
        improved = False

        # --- single residue moves ---
        num_tries = min(8, n - 2)
        for _ in range(num_tries):
            i = rng.randint(1, n - 2)
            old_pos = chain[i]
            for new_pos in _neighbours(old_pos):
                if not _try_move(chain, i, new_pos, where):
                    continue
                # apply move, recompute energy via a fresh map (cheap enough)
                del where[old_pos]
                where[new_pos] = i
                chain[i] = new_pos
                new_energy = _energy_fast(sequence, chain)
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
                    current_energy = new_energy
                    improved = True
                    break
                else:
                    # revert
                    del where[new_pos]
                    where[old_pos] = i
                    chain[i] = old_pos
            if improved:
                break

        # --- 2-opt segment reversal (richer move) ---
        if n >= 6 and rng.random() < 0.4:
            seg_len = rng.randint(3, min(9, n // 2))
            a = rng.randint(1, n - seg_len - 1)
            b = a + seg_len - 1
            # reversing chain[a..b] keeps the overall path connected iff
            # chain[a-1] is adjacent to chain[b] and chain[a] is adjacent to chain[b+1]
            if b + 1 < n:
                left = chain[a - 1]
                right = chain[b + 1]
                if (abs(left[0] - chain[b][0]) + abs(left[1] - chain[b][1]) == 1 and
                        abs(chain[a][0] - right[0]) + abs(chain[a][1] - right[1]) == 1):
                    # check no self-overlap within the reversed block (positions unchanged,
                    # so always fine) -> just apply and recompute
                    seg = chain[a:b + 1]
                    seg.reverse()
                    chain[a:b + 1] = seg
                    where = {p: i for i, p in enumerate(chain)}
                    new_energy = _energy_fast(sequence, chain)
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                        current_energy = new_energy
                        improved = True
                    else:
                        seg.reverse()
                        chain[a:b + 1] = seg
                        where = {p: i for i, p in enumerate(chain)}
                        current_energy = _energy_fast(sequence, chain)

        if not improved:
            no_improve += 1
            if no_improve > 25:
                break
        else:
            no_improve = 0

    return best_chain, best_energy

def fold(sequence, dimension):
    start_time = time.time()
    deadline = start_time + 170.0  # safety margin before 180s
    n = len(sequence)
    rng = random.Random(42 + n)

    best_chain = None
    best_energy = 0

    inits = _diverse_init(sequence, n, rng)

    for init_chain in inits:
        e = _energy_fast(sequence, init_chain)
        if e < best_energy:
            best_energy = e
            best_chain = list(init_chain)

    # local search on each diverse init
    for init_chain in inits:
        if time.time() > deadline - 25.0:
            break
        remaining = deadline - time.time()
        time_per_init = max(3.0, remaining / max(1, len(inits)))
        improved_chain, improved_energy = _local_search(
            sequence, init_chain, max_iters=400, rng=rng,
            time_limit=time.time() + time_per_init)
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = list(improved_chain)

    # fresh greedy grows + quick polish
    grows_done = 0
    while time.time() < deadline - 15.0:
        temperature = 0.3 + rng.random() * 2.5
        chain = _greedy_grow(sequence, dimension, rng, temperature=temperature)
        if chain is not None and len(chain) == n:
            grows_done += 1
            e = _energy_fast(sequence, chain)
            if e < best_energy or e < -3:
                time_left = deadline - time.time() - 10.0
                if time_left > 4.0:
                    improved_chain, improved_energy = _local_search(
                        sequence, chain, max_iters=120, rng=rng,
                        time_limit=time.time() + min(8.0, time_left * 0.3))
                    if improved_energy < best_energy:
                        best_energy = improved_energy
                        best_chain = list(improved_chain)
                elif e < best_energy:
                    best_energy = e
                    best_chain = list(chain)
        if grows_done > 1500:
            break

    # final polish on the best chain
    if best_chain is not None and time.time() < deadline - 4.0:
        final_chain, final_energy = _local_search(
            sequence, best_chain, max_iters=300, rng=rng, time_limit=deadline - 2.0)
        if final_energy < best_energy:
            best_chain = final_chain
            best_energy = final_energy

    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
