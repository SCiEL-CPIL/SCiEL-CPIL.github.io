"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

# Precompute smooth numbers and their divisibility relations
_smooth_numbers = []
for _x in range(1, 129):
    _y = _x
    for _p in (2, 3, 5):
        while _y % _p == 0:
            _y //= _p
    if _y == 1:
        _smooth_numbers.append(_x)

# Precompute which smooth numbers divide which
# For each smooth number s, find all smooth numbers that are multiples of s (and > s)
_divides_map = {}
for s in _smooth_numbers:
    _divides_map[s] = set()
    for t in _smooth_numbers:
        if t > s and t % s == 0:
            _divides_map[s].add(t)

# Precompute conflict sets: for each smooth number, which other smooth numbers conflict with it
# Two numbers conflict if one divides the other
_conflict = {}
for s in _smooth_numbers:
    _conflict[s] = set()
    for t in _smooth_numbers:
        if t != s:
            if t % s == 0 or s % t == 0:
                _conflict[s].add(t)

def _is_valid_antichain(terms):
    """Check if terms form a divisibility antichain."""
    ts = sorted(terms)
    for i in range(len(ts)):
        a = ts[i]
        for j in range(i + 1, len(ts)):
            b = ts[j]
            if b % a == 0:
                return False
    return True

def _find_min_antichain(target, max_terms=30, time_limit=None):
    """Find minimum-cardinality antichain of smooth numbers summing to target."""
    # Get smooth numbers <= target
    candidates = [s for s in _smooth_numbers if s <= target]
    candidates.sort()
    
    # Try increasing number of terms
    for k in range(1, min(max_terms, len(candidates)) + 1):
        if time_limit and time.time() > time_limit:
            break
        result = _find_antichain_of_size(target, candidates, k, time_limit)
        if result is not None:
            return result
    return None

def _find_antichain_of_size(target, candidates, k, time_limit=None):
    """Find an antichain of exactly k smooth numbers summing to target using DFS."""
    # Use DFS with pruning
    # Sort candidates to try larger values first (better for pruning)
    candidates_desc = sorted(candidates, reverse=True)
    
    # For efficiency, precompute suffix sums to prune
    # If we need k more terms and the sum of the largest k remaining is < remaining target, prune
    # If the sum of the smallest k remaining is > remaining target, prune
    
    def dfs(start_idx, remaining_sum, chosen, k_remaining):
        if time_limit and time.time() > time_limit:
            return None
        if k_remaining == 0:
            if remaining_sum == 0:
                return list(chosen)
            return None
        if remaining_sum < 0:
            return None
        
        # Pruning: if remaining_sum is less than the sum of the k_remaining smallest available, fail
        # if remaining_sum is greater than the sum of the k_remaining largest available, fail
        # But we need to respect the start_idx constraint
        
        # Simple bound: each term must be positive, so remaining_sum >= k_remaining
        if remaining_sum < k_remaining:
            return None
        
        # We need to pick k_remaining terms from candidates_desc[start_idx:]
        # The largest possible sum is sum of the first k_remaining from candidates_desc[start_idx:]
        # The smallest possible sum is sum of the last k_remaining from candidates_desc[start_idx:]
        available = candidates_desc[start_idx:]
        if len(available) < k_remaining:
            return None
        
        max_possible = sum(available[:k_remaining])
        if remaining_sum > max_possible:
            return None
        
        # Minimum possible: sum of k_remaining smallest from available
        # Since available is sorted descending, smallest are at the end
        min_possible = sum(available[-k_remaining:])
        if remaining_sum < min_possible:
            return None
        
        for i in range(start_idx, len(candidates_desc)):
            if time_limit and time.time() > time_limit:
                return None
            val = candidates_desc[i]
            if val > remaining_sum:
                continue
            
            # Check if val conflicts with any already chosen
            conflict = False
            for c in chosen:
                if val in _conflict[c] or c in _conflict[val]:
                    conflict = True
                    break
            if conflict:
                continue
            
            chosen.append(val)
            result = dfs(i + 1, remaining_sum - val, chosen, k_remaining - 1)
            if result is not None:
                return result
            chosen.pop()
        
        return None
    
    return dfs(0, target, [], k)

# Precompute all answers at module load time
_start_time = time.time()
_deadline = _start_time + 80.0  # Leave 10s margin

_precomputed_answers = []
for _n in range(1, 129):
    if time.time() > _deadline:
        # Fallback for remaining targets
        _precomputed_answers.append(None)
        continue
    
    _answer = _find_min_antichain(_n, max_terms=30, time_limit=_deadline)
    if _answer is None:
        # Should not happen for valid targets, but fallback
        _precomputed_answers.append(None)
    else:
        _precomputed_answers.append(_answer)

# Verify all are found; if any are None, we have a problem
# For safety, ensure we have 128 entries
while len(_precomputed_answers) < 128:
    _precomputed_answers.append(None)

def construct():
    # Return the precomputed answers
    # Ensure all entries are valid lists
    results = []
    for i in range(128):
        if _precomputed_answers[i] is not None:
            results.append(_precomputed_answers[i])
        else:
            # This should not happen; return empty as fallback
            results.append([])
    return results
# EVOLVE-BLOCK-END
