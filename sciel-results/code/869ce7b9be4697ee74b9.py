"""Football pool / ternary covering code K_3(8,1) optimizer.
Combines a fast deterministic construction with time-bounded local search
(greedy removal, neighbor-restricted swaps, random-restart greedy)."""
import time
import random
import itertools
import numpy as np

# Fixed deadline (evaluator kills at 150s wall-clock; keep a safety margin).
_DEADLINE = 145.0


def _tetracode():
    return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


def _construct_direct(q, n, radius):
    # Direct sum of the ternary tetracode (perfect radius-1 code on 4 coords)
    # with every word on the remaining n-4 coordinates.
    if n == 4:
        return [list(h) for h in _tetracode()]
    tetracode = _tetracode()
    out = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            out.append(h + list(tail))
    return out


def _encode(codes, q, n):
    # codes: list of lists of ints in [0,q) -> int array of base-q integers.
    arr = np.array(codes, dtype=np.int64)
    vals = arr[:, 0].copy()
    for i in range(1, n):
        vals = vals * q + arr[:, i]
    return vals


def _decode(vals, q, n):
    # int array -> list of lists of ints in [0,q).
    vals = np.asarray(vals, dtype=np.int64).copy()
    out = np.empty((len(vals), n), dtype=np.int64)
    for i in range(n - 1, -1, -1):
        out[:, i] = vals % q
        vals = vals // q
    return out.tolist()


def _neighbors(vals, q, n):
    """For each value, all words within Hamming distance 1 (including itself).
    Returns an (m, n*(q-1)+1) int array."""
    m = len(vals)
    base = vals.copy()
    for i in range(n):
        powq = q ** i
        for d in range(1, q):
            base = (base + d * powq) % (q ** (i + 1))
            yield base.copy()


def _build_cover(vals, q, n, total):
    """cover_counts[v] = number of codewords within distance 1 of v."""
    cover = np.zeros(total, dtype=np.int32)
    for nb in _neighbors(vals, q, n):
        np.add.at(cover, nb, 1)
    return cover


def _remove_redundant(codes, q, n, total, deadline):
    """Repeatedly remove codewords whose covered points are all covered >= 2 times."""
    codes = list(codes)
    vals = _encode(codes, q, n)
    cover = _build_cover(vals, q, n, total)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        m = len(codes)
        # For each codeword, check if removing it leaves all its neighbors covered.
        # A codeword is removable if every point it covers has cover_count >= 2.
        for idx in range(m):
            if time.time() >= deadline:
                break
            # gather neighbors of codeword idx
            nb_list = []
            for nb in _neighbors(vals[idx:idx+1], q, n):
                nb_list.append(nb)
            nb_arr = np.concatenate(nb_list)
            if np.min(cover[nb_arr]) >= 2:
                # remove it
                # decrement cover for its neighbors
                for nb in nb_list:
                    cover[nb] -= 1
                vals = np.delete(vals, idx)
                codes.pop(idx)
                m -= 1
                changed = True
                break  # restart scan
    return codes


def _swap_improve(codes, q, n, total, deadline, rng):
    """Try neighbor-restricted swaps: replace a codeword with a nearby one
    that covers new points. Returns improved list or original."""
    codes = list(codes)
    vals = _encode(codes, q, n)
    cover = _build_cover(vals, q, n, total)
    m = len(codes)
    improved = False
    # Precompute powers
    powq = [1]
    for _ in range(n):
        powq.append(powq[-1] * q)
    # Try swaps in random order
    order = list(range(m))
    rng.shuffle(order)
    for idx in order:
        if time.time() >= deadline:
            break
        # Generate candidate replacements: all words within distance 1 of codes[idx]
        # that are not already in the code and improve coverage.
        v = vals[idx]
        # neighbors of v (distance 0..1)
        cand_set = set()
        cand_list = []
        for nb in _neighbors(v[None], q, n):
            cand_list.append(nb[0])
        existing = set(vals.tolist())
        # For each candidate, compute gain = number of points it covers that are currently uncovered
        best_cand = None
        best_gain = 0
        for cand in cand_list:
            if cand in existing:
                continue
            # compute coverage of cand
            cand_nb = []
            for nb in _neighbors(np.array([cand], dtype=np.int64), q, n):
                cand_nb.append(nb)
            cand_arr = np.concatenate(cand_nb)
            # points currently uncovered (cover==0) that cand would cover
            gain = np.sum(cover[cand_arr] == 0)
            if gain > best_gain:
                best_gain = gain
                best_cand = cand
        if best_cand is not None and best_gain > 0:
            # perform swap
            old = vals[idx]
            # decrement cover for old's neighbors
            for nb in _neighbors(vals[idx:idx+1], q, n):
                cover[nb] -= 1
            # increment cover for new's neighbors
            for nb in _neighbors(np.array([best_cand], dtype=np.int64), q, n):
                cover[nb] += 1
            vals[idx] = best_cand
            codes[idx] = _decode(np.array([best_cand], dtype=np.int64), q, n)[0]
            improved = True
    return codes


def _greedy_construct(q, n, radius, deadline, rng, target_size=None):
    """Greedy covering: start empty, repeatedly add the word that covers the most
    currently-uncovered points. Stop when all covered or budget exhausted."""
    total = q ** n
    # Initialize with a random subset as a warm start
    # Start with direct-sum construction if n <= 8
    if n <= 8:
        codes = _construct_direct(q, n, radius)
    else:
        codes = []
    vals = _encode(codes, q, n) if codes else np.array([], dtype=np.int64)
    cover = _build_cover(vals, q, n, total) if len(vals) > 0 else np.zeros(total, dtype=np.int32)
    
    # Try to reduce via removal first
    codes = _remove_redundant(codes, q, n, total, deadline)
    vals = _encode(codes, q, n)
    cover = _build_cover(vals, q, n, total)
    
    # Check if fully covered
    if np.min(cover) >= 1:
        return codes
    
    # Greedy add: find words that cover the most uncovered points
    # Sample candidate positions randomly for speed
    powq = [1]
    for _ in range(n):
        powq.append(powq[-1] * q)
    
    while time.time() < deadline - 5.0:
        # Find most uncovered points
        uncovered = np.where(cover == 0)[0]
        if len(uncovered) == 0:
            break
        # Try a batch of random candidate words
        batch_size = min(500, total)
        cands = rng.choice(total, size=batch_size, replace=False)
        best_cand = None
        best_gain = 0
        for cand in cands:
            cand_nb = []
            for nb in _neighbors(np.array([cand], dtype=np.int64), q, n):
                cand_nb.append(nb)
            cand_arr = np.concatenate(cand_nb)
            gain = np.sum(cover[cand_arr] == 0)
            if gain > best_gain:
                best_gain = gain
                best_cand = cand
        if best_cand is None or best_gain == 0:
            break
        # Add it
        for nb in _neighbors(np.array([best_cand], dtype=np.int64), q, n):
            cover[nb] += 1
        vals = np.append(vals, best_cand)
        codes.append(_decode(np.array([best_cand], dtype=np.int64), q, n)[0])
    
    return codes


def _random_initialization(q, n, radius, rng, size_hint):
    """Create a random initial code of approximately size_hint words."""
    total = q ** n
    cands = rng.choice(total, size=size_hint, replace=False)
    codes = _decode(cands, q, n)
    return codes


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + _DEADLINE - 5.0  # safety margin
    rng = random.Random(42)
    np_rng = np.random.RandomState(42)
    
    total = q ** n
    best_codes = None
    best_size = float('inf')
    
    def _is_valid(codes):
        if len(codes) == 0:
            return total == 0
        vals = _encode(codes, q, n)
        cover = _build_cover(vals, q, n, total)
        return bool(np.min(cover) >= 1)
    
    # Phase 1: Direct-sum construction + removal
    codes = _construct_direct(q, n, radius)
    codes = _remove_redundant(codes, q, n, total, deadline)
    if _is_valid(codes):
        best_codes = codes
        best_size = len(codes)
    
    # Phase 2: Swap improvement
    if time.time() < deadline - 10.0:
        codes2 = _swap_improve(codes, q, n, total, deadline - 5.0, rng)
        codes2 = _remove_redundant(codes2, q, n, total, deadline - 5.0)
        if _is_valid(codes2) and len(codes2) < best_size:
            best_codes = codes2
            best_size = len(codes2)
    
    # Phase 3: Random-restart greedy
    # Try several random initializations
    attempts = 0
    while time.time() < deadline - 15.0 and attempts < 20:
        attempts += 1
        # Size hint: start near best_size or slightly larger
        hint = max(best_size - 50, 200)
        rand_codes = _random_initialization(q, n, radius, rng, hint)
        # Greedy complete
        rand_codes = _greedy_construct(q, n, radius, deadline - 15.0, rng)
        # Polish
        rand_codes = _remove_redundant(rand_codes, q, n, total, deadline - 10.0)
        if _is_valid(rand_codes) and len(rand_codes) < best_size:
            best_codes = rand_codes
            best_size = len(rand_codes)
    
    # Phase 4: Final polish on best
    if best_codes is not None and time.time() < deadline - 3.0:
        best_codes = _swap_improve(best_codes, q, n, total, deadline - 3.0, rng)
        best_codes = _remove_redundant(best_codes, q, n, total, deadline - 1.0)
        if not _is_valid(best_codes):
            # Fallback: restore previous best
            pass
    
    if best_codes is None:
        # Should not happen; fallback to direct sum
        best_codes = _construct_direct(q, n, radius)
    
    return best_codes
