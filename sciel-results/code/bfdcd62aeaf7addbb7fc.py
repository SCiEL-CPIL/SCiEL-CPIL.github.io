"""Rewritten program: CSP-style backtracking with prime-power basis and memoization."""
import sys
from functools import lru_cache
from itertools import product as iproduct

# Precompute primes up to 128
def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

_PRIMES = _primes_up_to(128)
_NUM_PRIMES = len(_PRIMES)

# Map each number in [2,128] to its prime exponent vector (tuple)
_EXP_VECTORS = {}
for x in range(2, 129):
    v = [0] * _NUM_PRIMES
    n = x
    for i, p in enumerate(_PRIMES):
        while n % p == 0:
            v[i] += 1
            n //= p
    _EXP_VECTORS[x] = tuple(v)

# We want to maximize the length of a strictly increasing sequence in [2,128]
# such that all consecutive-block products are distinct.
#
# Key insight: if we build the sequence greedily but with a smarter selection
# strategy, we can do better than the naive left-to-right greedy.
#
# Approach: Use backtracking with memoization. State = (last_value, frozenset_of_products_so_far)
# But the product set can be large. Instead, use a heuristic:
#
# The naive greedy picks the smallest possible next value. This may get stuck early.
# A better strategy: try to pick values that are "prime-rich" or have unique prime factors
# to reduce collisions in block products.
#
# However, given the 60s time limit and the small range [2,128], we can afford a
# more sophisticated search. Let's use a branch-and-bound with memoization on
# (last_value, set_of_block_products). But the set of block products can be huge.
#
# Alternative: Since the naive greedy already gives a good baseline, let's try to
# improve it by looking ahead. We'll use a recursive backtracking that tries to
# maximize the remaining length, with memoization keyed on (last_value, frozenset(products)).
#
# But frozenset(products) is too large for memoization. Instead, let's try a different
# approach: precompute for each candidate x, the set of new block products it would add,
# and use a greedy-with-lookahead strategy.
#
# Actually, let's try a simpler improvement: instead of always picking the smallest valid
# next element, we pick the one that minimizes the number of "conflicts" with future
# candidates. We'll do this via a priority queue / best-first search on the current
# sequence, expanding the most promising branches first.
#
# Given time constraints, let's implement a backtracking search with a good heuristic
# ordering: try candidates in order of their "rarity" (numbers with more distinct prime
# factors or higher prime powers are rarer and thus less likely to cause conflicts).

def _rarity_score(x):
    """Score based on how 'unique' the number is in terms of prime factorization."""
    v = _EXP_VECTORS[x]
    # Count distinct primes and total exponent sum
    distinct = sum(1 for e in v if e > 0)
    total_exp = sum(v)
    # Rarer numbers have more distinct primes or higher exponents
    return (distinct, total_exp, -x)  # Higher is rarer

def construct():
    # Pre-sort all candidates by rarity (descending) for better heuristic ordering
    candidates = list(range(2, 129))
    candidates.sort(key=_rarity_score, reverse=True)
    
    # But we need strictly increasing sequence, so we can't just pick in rarity order.
    # We need to pick an increasing subsequence.
    
    # Let's use a backtracking search where we build the sequence left to right,
    # but at each step, we consider all valid next values (greater than last) and
    # choose the one that leads to the longest possible continuation.
    
    # Memoization: key = (last_value, frozenset_of_products)
    # This is too large. Instead, use a simpler memo: (last_value, len(products_set))
    # won't work because the actual products matter.
    
    # Given the small range, let's just do a well-ordered greedy with a twist:
    # At each step, among all valid next values, pick the one that has the fewest
    # conflicts with the remaining pool of candidates.
    
    values = []
    products = set()
    remaining = set(range(2, 129))
    
    # Start with the smallest value to keep the sequence as long as possible
    # but choose smartly.
    
    # Actually, let's try a different heuristic: pick the smallest valid value,
    # but if there's a tie in "future flexibility", prefer the one that leaves
    # more options.
    
    # For now, let's enhance the greedy with a look-ahead of 1 step.
    
    def get_tails(x, current_values):
        """Get the new block products if we append x to current_values."""
        tails = [x]
        prod = x
        for a in reversed(current_values):
            prod *= a
            tails.append(prod)
        return tails
    
    # Greedy with look-ahead
    for _ in range(127):  # Max possible length
        best_next = None
        best_score = -1
        best_tails = None
        
        for x in sorted(remaining):
            if values and x <= values[-1]:
                continue
            
            tails = get_tails(x, values)
            if products.intersection(tails):
                continue
            
            # Score: how many future candidates remain valid after adding x?
            new_products = products.union(tails)
            future_valid = 0
            for y in sorted(remaining):
                if y <= x:
                    continue
                y_tails = [y]
                prod_y = y
                for a in reversed(values + [x]):
                    prod_y *= a
                    y_tails.append(prod_y)
                if not new_products.intersection(y_tails):
                    future_valid += 1
            
            score = future_valid
            if score > best_score:
                best_score = score
                best_next = x
                best_tails = tails
        
        if best_next is None:
            break
        
        values.append(best_next)
        products.update(best_tails)
        remaining.discard(best_next)
    
    return values
