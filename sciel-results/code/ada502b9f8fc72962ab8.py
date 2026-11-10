import time
from math import gcd

N = 512
DEADLINE = 140.0  # safety margin below the 150 s kill

# Precompute prime factor bitmasks
# Primes < 512
def _get_primes(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

_PRIMES = _get_primes(N)
_PRIME_INDEX = {p: i for i, p in enumerate(_PRIMES)}

# Mask for each number
_MASKS = [0] * N
for a in range(1, N):
    m = a
    mask = 0
    for p in _PRIMES:
        if p * p > m:
            break
        if m % p == 0:
            mask |= (1 << _PRIME_INDEX[p])
            while m % p == 0:
                m //= p
    if m > 1:
        mask |= (1 << _PRIME_INDEX[m])
    _MASKS[a] = mask

def _is_coprime_mask(a, chosen_masks):
    """Check if a is coprime to all in chosen_masks.
    chosen_masks is a list of masks or a single accumulated mask?
    If we maintain a single accumulated mask of all prime factors used,
    we can just check (mask_a & used_mask) == 0.
    """
    return (_MASKS[a] & used_mask) == 0

# We need to maintain `used_mask` in the search state.
# The current code passes `chosen_set`. We should change it to pass `used_mask` and `chosen_list`.

def _score(chosen):
    return sum(1.0 / (N - a) for a in chosen)

def _greedy_from(chosen, blocked, deadline, used_mask):
    """Greedy fill from current state."""
    chosen = list(chosen)
    # used_mask is the mask of primes already used in 'chosen'
    for a in range(N - 1, 0, -1):
        if time.time() > deadline:
            break
        if a in blocked:
            continue
        if (_MASKS[a] & used_mask) != 0:
            continue
        chosen.append(a)
        used_mask |= _MASKS[a]
    return chosen, used_mask

def construct():
    deadline = time.time() + DEADLINE

    # --- Seed: Greedy ---
    chosen = []
    used_mask = 0
    for a in range(N - 1, 0, -1):
        if (_MASKS[a] & used_mask) == 0:
            chosen.append(a)
            used_mask |= _MASKS[a]
    
    best = chosen
    best_score = _score(best)
    best_mask = used_mask

    # --- Local Search: 1-move add ---
    # Note: Adding an element is only possible if it's coprime to all existing.
    # Since we started with greedy, we might have missed some if we didn't scan all?
    # Greedy from high to low adds if possible. It doesn't remove.
    # So any element not added was blocked by some higher element.
    # To add one, we must remove a blocker.
    # So "1-move add" is impossible without removing first, unless we missed something?
    # No, greedy is maximal. You cannot add any single element to a maximal independent set
    # without removing at least one element.
    # So the "1-move add" loop in the original code was likely redundant or only useful if
    # the initial seed wasn't maximal. But my seed is maximal.
    # Let's skip pure add moves.

    # --- 1-move remove + re-greedy ---
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i, c in enumerate(best):
            if time.time() > deadline:
                break
            # Remove c
            rest = [x for j, x in enumerate(best) if j != i]
            # Recalculate mask for rest
            rest_mask = 0
            for x in rest:
                rest_mask |= _MASKS[x]
            
            cand, cand_mask = _greedy_from(rest, {c}, deadline, rest_mask)
            s = _score(cand)
            if s > best_score + 1e-15:
                best = cand
                best_score = s
                best_mask = cand_mask
                improved = True
                break

    # --- 2-move swap: Remove 2, re-greedy ---
    # This is the main engine for improvement.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(best)
        for i in range(n):
            if time.time() > deadline:
                return best
            for j in range(i+1, n):
                if time.time() > deadline:
                    return best
                removed = {best[i], best[j]}
                rest = [x for k, x in enumerate(best) if k != i and k != j]
                rest_mask = 0
                for x in rest:
                    rest_mask |= _MASKS[x]
                
                cand, cand_mask = _greedy_from(rest, removed, deadline, rest_mask)
                s = _score(cand)
                if s > best_score + 1e-15:
                    best = cand
                    best_score = s
                    best_mask = cand_mask
                    improved = True
                    break
            if improved:
                break

    # --- 3-move swap? ---
    # Might be too slow. Let's see if 2-move is enough.
    # The original code had a "2-move swap" that tried adding a specific candidate.
    # My _greedy_from already tries to add any candidate.
    
    return best
