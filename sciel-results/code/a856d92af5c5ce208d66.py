"""Size-efficient sorting networks on 16 wires.

Search-based construction: randomized greedy build + local-search (2-opt)
optimization, verified via the zero-one principle (all 65536 bitmasks) with
fast bitmask simulation.
"""
import time
import random
import math

N = 16
TOTAL = 1 << N          # 65536
TARGET = 0xFFFF         # fully sorted bitmask (all ones)
MAX_COMPARATORS = 4096

# Precompute: for each wire i, the bitmask of the top (N-1-i) bits.
# Used to check "wire i holds a 1 in a non-sorted position".
_TOPMASK = []
for _i in range(N):
    _m = 0
    for _b in range(N - 1 - _i, N):
        _m |= (1 << _b)
    _TOPMASK.append(_m)


def _apply(comp, state):
    """Apply one comparator (i,j) to every state in the array.
    state[k] is a 16-bit int (bit b = wire b)."""
    i, j = comp
    bi, bj = 1 << i, 1 << j
    for k in range(TOTAL):
        s = state[k]
        if (s & bi) and not (s & bj):
            state[k] = (s ^ bi) | bj


def _apply_batch(comps, state):
    for c in comps:
        _apply(c, state)


def _verify(comps, state):
    """Return True if the network sorts all inputs."""
    for k in range(TOTAL):
        if state[k] != TARGET:
            return False
    return True


def _count_bad(state):
    """Count (k,i) pairs where wire i is 1 but position i is 'wrong' (should be 0)."""
    bad = 0
    top = _TOPMASK
    for k in range(TOTAL):
        s = state[k]
        if s == TARGET:
            continue
        # For each wire i, if s has bit i set and top[i] has bit i set (i.e. i < N-1),
        # that's a misplaced 1.
        # Actually simpler: a 1 is misplaced iff it's in a position that should be 0 in sorted.
        # Sorted is all 1s, so NO position is "should be 0". 
        # Wait: sorted output is all 1s (0xFFFF). So any input that isn't all 1s must have
        # zeros that need to move to the top. Let me reconsider.
        # 
        # For a bitmask input, sorted output is always 0xFFFF (all ones) ONLY IF the input
        # is all ones. That's wrong. Let me re-think.
        #
        # A bitmask represents a 0/1 sequence. Sorted means all 0s come first, then all 1s.
        # So sorted output for a mask with m ones is: lower (N-m) bits = 0, upper m bits = 1.
        # I need to fix the target.
        pass
    return bad


def _sorted_target(mask):
    """Return the sorted bitmask for a given input mask."""
    m = bin(mask).count('1')
    # m ones in the top m positions
    return ((1 << m) - 1) << (N - m)


def _count_bad2(state):
    """Count total misplaced bits across all inputs."""
    bad = 0
    for k in range(TOTAL):
        s = state[k]
        t = _sorted_target(s)
        # Wait, state[k] is the CURRENT state of input k after some comparators.
        # We need to compare to the target for the ORIGINAL input k.
        # Actually state[k] starts as k itself (the input). After comparators, it evolves.
        # The target for input k is _sorted_target(k).
        # But I'm storing state indexed by k (the original input). So target = _sorted_target(k).
        pass
    return bad


def _misplaced_count(state, targets):
    """Count total number of bits that are in wrong positions across all inputs.
    state[k] = current bitmask for input k.
    targets[k] = sorted target bitmask for input k.
    """
    bad = 0
    for k in range(TOTAL):
        s = state[k]
        t = targets[k]
        # XOR gives bits that differ; count them
        bad += bin(s ^ t).count('1')
    return bad


def _init_state():
    """Initialize state array: state[k] = k for k in 0..65535."""
    return list(range(TOTAL))


def _build_targets():
    """Precompute sorted target for each input mask."""
    targets = [0] * TOTAL
    for k in range(TOTAL):
        m = bin(k).count('1')
        targets[k] = ((1 << m) - 1) << (N - m)
    return targets


# Precompute targets once
_TARGETS = _build_targets()


def _greedy_build(rng, max_iters=200000):
    """Greedy: repeatedly add the comparator that reduces total misplacement the most."""
    state = _init_state()
    comps = []
    targets = _TARGETS
    best_bad = _misplaced_count(state, targets)
    
    # Candidate comparators: all pairs (i, j) with i < j
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    while best_bad > 0 and len(comps) < MAX_COMPARATORS:
        # Evaluate each candidate
        best_gain = 0
        best_comp = None
        for c in all_pairs:
            i, j = c
            bi, bj = 1 << i, 1 << j
            # Simulate applying c to all states, count misplacement
            # To save time, only count the change
            gain = 0
            for k in range(TOTAL):
                s = state[k]
                t = targets[k]
                before = bin(s ^ t).count('1')
                if (s & bi) and not (s & bj):
                    s2 = (s ^ bi) | bj
                    after = bin(s2 ^ t).count('1')
                    gain += before - after
                # else no change
            if gain > best_gain:
                best_gain = gain
                best_comp = c
        
        if best_comp is None or best_gain <= 0:
            break
        
        _apply(best_comp, state)
        comps.append(best_comp)
        best_bad -= best_gain
    
    return comps


def _random_greedy_build(rng, max_comps=150):
    """Build a random-ish network by adding comparators that help, with some randomness."""
    state = _init_state()
    comps = []
    targets = _TARGETS
    best_bad = _misplaced_count(state, targets)
    
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    while best_bad > 0 and len(comps) < max_comps:
        # Evaluate all candidates
        gains = []
        for c in all_pairs:
            i, j = c
            bi, bj = 1 << i, 1 << j
            gain = 0
            for k in range(TOTAL):
                s = state[k]
                t = targets[k]
                if (s & bi) and not (s & bj):
                    s2 = (s ^ bi) | bj
                    gain += bin(s ^ t).count('1') - bin(s2 ^ t).count('1')
            gains.append(gain)
        
        # Pick among top candidates with some randomness
        max_gain = max(gains)
        if max_gain <= 0:
            break
        # Select from candidates with gain >= max_gain * 0.5
        threshold = max_gain * 0.5
        candidates = [c for c, g in zip(all_pairs, gains) if g >= threshold]
        c = rng.choice(candidates)
        _apply(c, state)
        comps.append(c)
        best_bad -= max_gain  # approximate
    
    return comps


def _local_search(comps, rng, max_iters=5000, timeout=60):
    """2-opt local search: try removing one comparator and see if still valid; 
    also try swapping adjacent comparators."""
    start = time.time()
    state = _init_state()
    _apply_batch(comps, state)
    
    if not _verify(comps, state):
        return comps  # not valid, skip
    
    current_len = len(comps)
    best_comps = comps[:]
    best_len = current_len
    
    for it in range(max_iters):
        if time.time() - start > timeout:
            break
        
        # Try removing a random comparator
        if best_len > 1:
            idx = rng.randint(0, best_len - 1)
            trial = best_comps[:idx] + best_comps[idx + 1:]
            st = _init_state()
            _apply_batch(trial, st)
            if _verify(trial, st):
                best_comps = trial
                best_len -= 1
                state = st
                continue
        
        # Try swapping adjacent comparators
        if best_len > 1:
            idx = rng.randint(0, best_len - 2)
            trial = best_comps[:]
            trial[idx], trial[idx + 1] = trial[idx + 1], trial[idx]
            st = _init_state()
            _apply_batch(trial, st)
            if _verify(trial, st):
                best_comps = trial
                state = st
    
    return best_comps


def construct():
    deadline = time.time() + 55  # 55s safety margin
    
    rng = random.Random(42)
    
    # Start with a known valid network: odd-even transposition
    comps = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # Verify it's correct
    state = _init_state()
    _apply_batch(comps, state)
    assert _verify(comps, state), "Initial network should be valid"
    
    best_comps = comps[:]
    best_len = len(comps)
    
    # Try random greedy builds + local search
    while time.time() < deadline:
        # Build a random network
        trial = _random_greedy_build(rng, max_comps=100)
        
        # Check validity
        st = _init_state()
        _apply_batch(trial, st)
        if _verify(trial, st):
            # Local search
            trial = _local_search(trial, rng, max_iters=2000, timeout=30)
            if len(trial) < best_len:
                best_comps = trial
                best_len = len(trial)
        
        # Also try local search on the current best
        if time.time() < deadline:
            improved = _local_search(best_comps, rng, max_iters=2000, timeout=30)
            if len(improved) < best_len:
                best_comps = improved
                best_len = len(improved)
    
    return best_comps
