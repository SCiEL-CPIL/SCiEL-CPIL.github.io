"""Rewritten program: greedy + hill-climb with 2-swap moves for pairwise-coprime selection."""
# EVOLVE-BLOCK-START
import time
import math
import random

_DEADLINE = 150.0
_START = None

def _elapsed():
    return time.time() - _START if _START is not None else 0.0

def _deadline_guard(limit=_DEADLINE, margin=2.0):
    return _elapsed() > (limit - margin)

def _sieve_primes(n):
    """Return list of primes < n."""
    if n < 2:
        return []
    sieve = bytearray([1]) * n
    sieve[0] = sieve[1] = 0
    for i in range(2, int(math.isqrt(n - 1)) + 1):
        if sieve[i]:
            for j in range(i * i, n, i):
                sieve[j] = 0
    return [i for i in range(2, n) if sieve[i]]

def _build_prime_masks(primes):
    """For each number 1..511, compute bitmask of its prime factors."""
    masks = [0] * 512
    for a in range(2, 512):
        x = a
        mask = 0
        for idx, p in enumerate(primes):
            if p * p > x:
                break
            if x % p == 0:
                mask |= (1 << idx)
                while x % p == 0:
                    x //= p
        if x > 1:
            for idx, p in enumerate(primes):
                if p == x:
                    mask |= (1 << idx)
                    break
        masks[a] = mask
    return masks

def _score(chosen):
    return sum(1.0 / (512.0 - a) for a in chosen)

def _greedy_select(masks, order):
    """Greedy select in given order using bitmask coprimality."""
    chosen = []
    chosen_mask = 0
    for a in order:
        if masks[a] & chosen_mask:
            continue
        chosen.append(a)
        chosen_mask |= masks[a]
    return chosen

def _hill_climb(chosen, masks):
    """Hill-climbing with add and 2-swap moves. Repeat until no improvement."""
    chosen_set = set(chosen)
    cur_score = _score(chosen)
    chosen_mask = 0
    for a in chosen:
        chosen_mask |= masks[a]
    
    improved = True
    while improved:
        if _deadline_guard():
            break
        improved = False
        best_gain = 0.0
        best_action = None  # ('add', a) or ('swap', remove, add)
        
        # Try adding each candidate not in chosen
        for a in range(511, 0, -1):
            if a in chosen_set:
                continue
            if masks[a] & chosen_mask:
                continue
            gain = 1.0 / (512.0 - a)
            if gain > best_gain:
                best_gain = gain
                best_action = ('add', a)
        
        # Try 2-swap: remove one chosen element, add one new element
        for rem in chosen:
            if _deadline_guard():
                break
            rem_mask = masks[rem]
            # Mask of remaining chosen elements
            rem_chosen_mask = 0
            for a in chosen:
                if a != rem:
                    rem_chosen_mask |= masks[a]
            
            rem_loss = 1.0 / (512.0 - rem)
            
            for a in range(511, 0, -1):
                if a in chosen_set:
                    continue
                if masks[a] & rem_chosen_mask:
                    continue
                gain = 1.0 / (512.0 - a) - rem_loss
                if gain > best_gain:
                    best_gain = gain
                    best_action = ('swap', rem, a)
        
        if best_action is not None and best_gain > 1e-15:
            if best_action[0] == 'add':
                a = best_action[1]
                chosen.append(a)
                chosen_set.add(a)
                chosen_mask |= masks[a]
                cur_score += 1.0 / (512.0 - a)
            else:
                _, rem, a = best_action
                chosen.remove(rem)
                chosen_set.remove(rem)
                # Recompute chosen_mask
                chosen_mask = 0
                for x in chosen:
                    chosen_mask |= masks[x]
                chosen.append(a)
                chosen_set.add(a)
                chosen_mask |= masks[a]
                cur_score += 1.0 / (512.0 - a) - 1.0 / (512.0 - rem)
            improved = True
    
    return chosen, cur_score

def _polish(chosen, masks):
    """Final polish: try all 2-swaps one more time with tighter tolerance."""
    chosen_set = set(chosen)
    cur_score = _score(chosen)
    
    improved = True
    while improved:
        if _deadline_guard():
            break
        improved = False
        best_gain = 0.0
        best_action = None
        
        # Try adding
        chosen_mask = 0
        for a in chosen:
            chosen_mask |= masks[a]
        
        for a in range(511, 0, -1):
            if a in chosen_set:
                continue
            if masks[a] & chosen_mask:
                continue
            gain = 1.0 / (512.0 - a)
            if gain > best_gain:
                best_gain = gain
                best_action = ('add', a)
        
        # Try 2-swaps
        for rem in chosen:
            if _deadline_guard():
                break
            rem_chosen_mask = 0
            for a in chosen:
                if a != rem:
                    rem_chosen_mask |= masks[a]
            rem_loss = 1.0 / (512.0 - rem)
            
            for a in range(511, 0, -1):
                if a in chosen_set:
                    continue
                if masks[a] & rem_chosen_mask:
                    continue
                gain = 1.0 / (512.0 - a) - rem_loss
                if gain > best_gain:
                    best_gain = gain
                    best_action = ('swap', rem, a)
        
        if best_action is not None and best_gain > 1e-18:
            if best_action[0] == 'add':
                a = best_action[1]
                chosen.append(a)
                chosen_set.add(a)
                cur_score += 1.0 / (512.0 - a)
            else:
                _, rem, a = best_action
                chosen.remove(rem)
                chosen_set.remove(rem)
                chosen.append(a)
                chosen_set.add(a)
                cur_score += 1.0 / (512.0 - a) - 1.0 / (512.0 - rem)
            improved = True
    
    return chosen, cur_score

def construct():
    global _START
    _START = time.time()
    
    N = 512
    primes = _sieve_primes(N)
    masks = _build_prime_masks(primes)
    
    best_chosen = []
    best_score = 0.0
    rng = random.Random(42)
    
    # Multiple greedy initializations with different orderings
    n_restarts = 100
    for restart in range(n_restarts):
        if _deadline_guard():
            break
        
        if restart == 0:
            order = list(range(511, 0, -1))
        elif restart == 1:
            order = sorted(range(511, 0, -1), key=lambda a: (masks[a].bit_count(), -a))
        else:
            order = list(range(511, 0, -1))
            n_swaps = rng.randint(5, 150)
            for _ in range(n_swaps):
                i, j = rng.randint(0, 510), rng.randint(0, 510)
                order[i], order[j] = order[j], order[i]
        
        chosen = _greedy_select(masks, order)
        
        # Hill-climb to local optimum with add + swap moves
        chosen, score = _hill_climb(chosen, masks)
        
        # Final polish pass
        chosen, score = _polish(chosen, masks)
        
        if score > best_score:
            best_score = score
            best_chosen = chosen[:]
    
    # Verify
    assert len(set(best_chosen)) == len(best_chosen)
    assert all(1 <= x <= 511 for x in best_chosen)
    for i in range(len(best_chosen)):
        for j in range(i + 1, len(best_chosen)):
            assert math.gcd(best_chosen[i], best_chosen[j]) == 1
    
    return best_chosen
# EVOLVE-BLOCK-END
