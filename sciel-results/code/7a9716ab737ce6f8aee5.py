"""Improved covering-code search: vectorized greedy + fast local search with restarts."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

_DEADLINE_SECONDS = 138.0


def _elapsed(start):
    return time.monotonic() - start


def _idx_to_tuple(idx, n, q):
    t = []
    for i in range(n):
        t.append(idx % q)
        idx //= q
    return t[::-1]


def _tuple_to_idx(t, n, q):
    idx = 0
    for d in t:
        idx = idx * q + d
    return idx


def _build_neighbor_table(total, n, q):
    """Precompute for each codeword index, the list of indices it covers (radius 1)."""
    neighbors = []
    for c in range(total):
        c_t = _idx_to_tuple(c, n, q)
        covered_set = [c]
        for pos in range(n):
            orig = c_t[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_t = list(c_t)
                new_t[pos] = val
                covered_set.append(_tuple_to_idx(new_t, n, q))
        neighbors.append(covered_set)
    return neighbors


def _greedy_construct_fast(q, n, radius, rng, deadline):
    """Vectorized greedy set cover with efficient gain updates."""
    total = q ** n
    neighbors = _build_neighbor_table(total, n, q)
    
    covered = np.zeros(total, dtype=np.uint8)
    gain = np.full(total, 1 + n * (q - 1), dtype=np.int32)
    codewords = []
    
    # Precompute reverse: for each word, which codewords cover it
    # For radius 1, word w is covered by codeword c if dist(c,w) <= 1
    # Build this efficiently
    reverse_cover = [[] for _ in range(total)]
    for c in range(total):
        for w in neighbors[c]:
            reverse_cover[w].append(c)
    
    while True:
        if time.monotonic() > deadline:
            break
        
        max_gain = int(np.max(gain))
        if max_gain <= 0:
            break
        
        # Find codeword with max gain (break ties randomly for diversity)
        candidates = np.where(gain == max_gain)[0]
        if len(candidates) > 1:
            c = int(rng.integers(0, len(candidates)))
            c = int(candidates[c])
        else:
            c = int(candidates[0])
        
        codewords.append(c)
        
        # Update: mark neighbors as covered, decrement gains of other codewords
        for w in neighbors[c]:
            if covered[w] == 0:
                covered[w] = 1
                for cc in reverse_cover[w]:
                    if cc != c:  # don't decrement our own gain
                        gain[cc] -= 1
        
        if int(np.sum(covered)) == total:
            break
    
    return codewords


def _is_cover(codewords_idx, q, n, radius, neighbors):
    """Check if codewords (as indices) form a valid covering."""
    total = q ** n
    covered = np.zeros(total, dtype=np.uint8)
    for c in codewords_idx:
        for w in neighbors[c]:
            covered[w] = 1
    return bool(covered.all())


def _local_search_fast(codewords_idx, q, n, radius, rng, deadline, neighbors):
    """
    Fast local search using coverage counts.
    codewords_idx: list of codeword indices
    """
    total = q ** n
    codewords_idx = list(codewords_idx)
    
    # Build coverage count: how many codewords cover each word
    cover_count = np.zeros(total, dtype=np.int32)
    for c in codewords_idx:
        for w in neighbors[c]:
            cover_count[w] += 1
    
    def try_remove(i):
        """Try removing codeword i. Returns True if coverage maintained."""
        c = codewords_idx[i]
        # Check if any word would become uncovered
        for w in neighbors[c]:
            if cover_count[w] == 1:
                return False
        # Remove it
        for w in neighbors[c]:
            cover_count[w] -= 1
        codewords_idx.pop(i)
        return True
    
    def try_swap(i, new_c):
        """Try swapping codeword i with new_c. Returns True if coverage maintained and new_c not already present."""
        if new_c in codewords_idx:
            return False
        c = codewords_idx[i]
        # Check: after removing c and adding new_c, is everything still covered?
        # Words that would lose coverage: those covered only by c (not by new_c)
        # Words that gain coverage: those covered by new_c but not by c
        # We need: for all w, cover_count[w] - (1 if w in neighbors[c] else 0) + (1 if w in neighbors[new_c] else 0) >= 1
        
        # Quick check: for words covered by c but not new_c, cover_count must be > 1
        set_c = set(neighbors[c])
        set_new = set(neighbors[new_c])
        only_c = set_c - set_new
        for w in only_c:
            if cover_count[w] == 1:
                return False
        
        # Perform swap
        for w in neighbors[c]:
            cover_count[w] -= 1
        for w in neighbors[new_c]:
            cover_count[w] += 1
        codewords_idx[i] = new_c
        return True
    
    # Phase 1: Remove redundant codewords
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        for i in range(len(codewords_idx)):
            if time.monotonic() > deadline:
                break
            if try_remove(i):
                improved = True
                break
    
    # Phase 2: Try swaps to reduce count (remove one, we already tried that)
    # Actually swaps don't reduce count. We need to find ways to remove more.
    # Better: after removing, try to find if we can remove more by first swapping.
    
    # Phase 3: Iterative removal with swap assistance
    # Try: for each codeword, if it can't be removed, try swapping it with something
    # that allows removal of another codeword.
    
    for iteration in range(100):
        if time.monotonic() > deadline:
            break
        progress = False
        for i in range(len(codewords_idx)):
            if time.monotonic() > deadline:
                break
            # Try direct removal
            if try_remove(i):
                progress = True
                break
        if not progress:
            # Try swap + remove: swap codeword i with random new_c, then try to remove some j
            for _ in range(20):
                if time.monotonic() > deadline:
                    break
                i = int(rng.integers(0, len(codewords_idx)))
                new_c = int(rng.integers(0, total))
                if new_c in codewords_idx:
                    continue
                # Save state for rollback
                old_c = codewords_idx[i]
                if try_swap(i, new_c):
                    # Now try to remove some codeword
                    for j in range(len(codewords_idx)):
                        if j == i:
                            continue
                        if try_remove(j):
                            progress = True
                            break
                    if not progress:
                        # Rollback: swap back
                        # This is complex; just accept the swap if it doesn't hurt
                        # Actually we should rollback to keep count same
                        # For simplicity, keep the swap (count unchanged) and continue
                        pass
                if progress:
                    break
        if not progress:
            break
    
    return codewords_idx


def _greedy_with_diversity(q, n, radius, rng, deadline):
    """Greedy with random tie-breaking for diversity."""
    total = q ** n
    neighbors = _build_neighbor_table(total, n, q)
    
    covered = np.zeros(total, dtype=np.uint8)
    gain = np.full(total, 1 + n * (q - 1), dtype=np.int32)
    codewords = []
    
    reverse_cover = [[] for _ in range(total)]
    for c in range(total):
        for w in neighbors[c]:
            reverse_cover[w].append(c)
    
    while True:
        if time.monotonic() > deadline:
            break
        
        max_gain = int(np.max(gain))
        if max_gain <= 0:
            break
        
        candidates = np.where(gain == max_gain)[0]
        c = int(candidates[int(rng.integers(0, len(candidates)))])
        
        codewords.append(c)
        
        for w in neighbors[c]:
            if covered[w] == 0:
                covered[w] = 1
                for cc in reverse_cover[w]:
                    if cc != c:
                        gain[cc] -= 1
        
        if int(np.sum(covered)) == total:
            break
    
    return codewords


def construct(q, n, radius):
    start = time.monotonic()
    deadline = start + _DEADLINE_SECONDS
    total = q ** n
    
    # Precompute neighbor table once
    neighbors = _build_neighbor_table(total, n, q)
    
    best = None
    best_len = float('inf')
    
    # Multiple restarts with different random seeds
    num_restarts = 5
    time_per_restart = (_DEADLINE_SECONDS - 5.0) / num_restarts
    
    for restart in range(num_restarts):
        if time.monotonic() > deadline - 10.0:
            break
        
        rng = np.random.default_rng(42 + restart * 1000)
        
        # Greedy construction
        t_restart_start = time.monotonic()
        t_restart_deadline = min(deadline - 5.0, t_restart_start + time_per_restart * 0.6)
        
        cw_idx = _greedy_with_diversity(q, n, radius, rng, t_restart_deadline)
        
        if len(cw_idx) >= best_len:
            continue
        
        # Local search
        t_ls_start = time.monotonic()
        t_ls_deadline = min(deadline - 3.0, t_ls_start + time_per_restart * 0.4)
        
        cw_idx = _local_search_fast(cw_idx, q, n, radius, rng, t_ls_deadline, neighbors)
        
        # Verify
        if _is_cover(cw_idx, q, n, radius, neighbors):
            if len(cw_idx) < best_len:
                best_len = len(cw_idx)
                best = list(cw_idx)
    
    # If no valid solution found, use fallback
    if best is None:
        # Simple greedy without local search
        rng = np.random.default_rng(0)
        cw_idx = _greedy_construct_fast(q, n, radius, rng, deadline - 1.0)
        if not _is_cover(cw_idx, q, n, radius, neighbors):
            # Direct product construction
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            best = []
            for h in tetracode:
                for tail in itertools.product(range(3), repeat=n - 4):
                    best.append(h + list(tail))
            return best
        best = cw_idx
    
    return [_idx_to_tuple(c, n, q) for c in best]
# EVOLVE-BLOCK-END
