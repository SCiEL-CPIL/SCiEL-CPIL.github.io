"""Optimized pairwise-coprime selector for N=512."""
import math
import time
import random

_N = 512
_DEADLINE_SECONDS = 140.0  # safety margin within 150s evaluator kill

# Precompute primes < 512
def _sieve(n):
    is_p = [True] * (n + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_p[i]:
            for j in range(i*i, n + 1, i):
                is_p[j] = False
    return is_p

_IS_PRIME = _sieve(_N)
_PRIMES = [p for p in range(2, _N) if _IS_PRIME[p]]
# Prime reciprocal sum (for reporting, not scoring)
_PRIME_RECIP_SUM = sum(1.0 / p for p in _PRIMES)

# Precompute: for each a in 1..511, its set of prime divisors (as frozenset of small ints)
# We use a bitmask over primes < 512 for fast gcd-free coprime check
_NUM_PRIMES = len(_PRIMES)
_PRIME_TO_IDX = {p: i for i, p in enumerate(_PRIMES)}

def _factor_mask(a):
    """Return a bitmask of primes dividing a, for a < 512."""
    mask = 0
    for i, p in enumerate(_PRIMES):
        if p * p > a:
            break
        if a % p == 0:
            mask |= (1 << i)
            while a % p == 0:
                a //= p
    if a > 1:
        mask |= (1 << _PRIME_TO_IDX[a])
    return mask

# Precompute masks for all a in 1..511
_MASKS = [0] + [_factor_mask(a) for a in range(1, _N)]

# Score for each a: 1/(512 - a)
_SCORES = [0.0] + [1.0 / (_N - a) for a in range(1, _N)]

def construct():
    """
    Return a list of distinct pairwise coprime integers a in 1..511
    maximizing sum 1/(512 - a).
    """
    start_time = time.time()
    
    # Greedy baseline: pick in descending order of 1/(512-a), i.e., a=511, 510, ...
    # but only if coprime to all already chosen.
    # This is the seed's approach. We'll improve with local search.
    
    def _greedy_order():
        """Return candidate order: highest score first, with some randomization."""
        order = list(range(_N - 1, 0, -1))
        # Slightly shuffle among close scores to escape local optima
        return order
    
    def _build_greedy():
        chosen = set()
        for a in range(_N - 1, 0, -1):
            m = _MASKS[a]
            ok = True
            for b in chosen:
                if m & _MASKS[b]:
                    ok = False
                    break
            if ok:
                chosen.add(a)
        return chosen
    
    def _score_of(chosen):
        return sum(_SCORES[a] for a in chosen)
    
    # Initial solution
    best_set = _build_greedy()
    best_score = _score_of(best_set)
    
    # Local search: try to replace one element with another to improve score
    # While respecting pairwise coprimality.
    # Also try adding if possible (greedy already maximal by construction, but check).
    
    def _local_search(chosen_set, budget_sec):
        """
        Hill-climb on the set. At each step, try:
          1. Add an unchosen a if coprime to all chosen.
          2. Replace a chosen a with an unchosen b if:
             - b is coprime to all other chosen
             - score(b) > score(a)
        """
        chosen = set(chosen_set)
        cur_score = _score_of(chosen)
        changed = True
        no_improve_rounds = 0
        
        while changed and (time.time() - start_time) < budget_sec:
            changed = False
            no_improve_rounds += 1
            
            # Try adding
            for a in range(_N - 1, 0, -1):
                if a in chosen:
                    continue
                m = _MASKS[a]
                ok = True
                for b in chosen:
                    if m & _MASKS[b]:
                        ok = False
                        break
                if ok:
                    chosen.add(a)
                    cur_score += _SCORES[a]
                    changed = True
                    no_improve_rounds = 0
                    break
            
            if changed:
                continue
            
            # Try replacing
            best_gain = 0.0
            best_swap = None
            for a in list(chosen):
                ma = _MASKS[a]
                others = [b for b in chosen if b != a]
                for b in range(_N - 1, 0, -1):
                    if b in chosen or b == a:
                        continue
                    mb = _MASKS[b]
                    # b must be coprime to all others
                    ok = True
                    for c in others:
                        if mb & _MASKS[c]:
                            ok = False
                            break
                    if not ok:
                        continue
                    gain = _SCORES[b] - _SCORES[a]
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (a, b)
            
            if best_swap is not None:
                a_out, b_in = best_swap
                chosen.remove(a_out)
                chosen.add(b_in)
                cur_score += best_gain
                changed = True
                no_improve_rounds = 0
        
        return chosen, cur_score
    
    # Run local search with time budget
    search_budget = _DEADLINE_SECONDS - (time.time() - start_time) - 15.0
    if search_budget > 5.0:
        improved_set, improved_score = _local_search(best_set, time.time() + search_budget)
        if improved_score > best_score:
            best_set = improved_set
            best_score = improved_score
    
    # Multi-start with randomization if time permits
    remaining = _DEADLINE_SECONDS - (time.time() - start_time) - 10.0
    if remaining > 30.0:
        random.seed(42)
        for trial in range(10):
            if (time.time() - start_time) > _DEADLINE_SECONDS - 10.0:
                break
            
            # Randomized greedy: shuffle with bias toward high scores
            candidates = list(range(_N - 1, 0, -1))
            # Perturb: swap some random pairs
            for _ in range(50):
                i, j = random.randint(0, len(candidates)-1), random.randint(0, len(candidates)-1)
                candidates[i], candidates[j] = candidates[j], candidates[i]
            
            # Build greedy with this order
            trial_set = set()
            for a in candidates:
                m = _MASKS[a]
                ok = True
                for b in trial_set:
                    if m & _MASKS[b]:
                        ok = False
                        break
                if ok:
                    trial_set.add(a)
            
            trial_score = _score_of(trial_set)
            if trial_score > best_score:
                # Polish
                trial_set, trial_score = _local_search(
                    trial_set, time.time() + max(0, _DEADLINE_SECONDS - (time.time() - start_time) - 10.0)
                )
                if trial_score > best_score:
                    best_set = trial_set
                    best_score = trial_score
    
    # Final safety check
    if (time.time() - start_time) > _DEADLINE_SECONDS - 5.0:
        pass  # Just return what we have
    
    return list(best_set)


# Keep the function name and signature exactly as required
# construct() -> list of distinct pairwise coprime integers in 1..511
