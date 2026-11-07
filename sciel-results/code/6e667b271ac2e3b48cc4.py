"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import gcd, isqrt
from functools import lru_cache

# We need to find a subset S of {1, 2, ..., 256} such that:
# 1. All elements are distinct.
# 2. Every pair (a, b) in S has gcd(a, b) == 1.
# 3. 1 is allowed (and gcd(1, x) = 1 for all x, so 1 is always compatible).
# 4. We want to maximize the sum of elements in S.

# Strategy:
# This is a Maximum Weight Independent Set problem in a graph where:
# - Vertices are integers 1..256.
# - An edge exists between u and v if gcd(u, v) > 1.
# - Weight of vertex v is v.
# We want the maximum weight independent set.

# Note: 1 is coprime with everything, so it's always beneficial to include 1.
# So we can fix 1 in the solution and solve for 2..256.

# The graph has 255 vertices (2..256).
# Maximum Weight Independent Set is NP-hard, but for n=255 with a specific structure
# (coprime graph), we might be able to use branch and bound or other heuristics.

# However, 255 is too large for a naive exponential branch and bound.
# Let's analyze the structure.
# Two numbers conflict if they share a prime factor.
# Primes up to 256: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251.

# If we pick a prime p, we cannot pick any multiple of p.
# To maximize the sum, we likely want to pick large numbers.
# Large numbers are often prime or have large prime factors.

# Let's try a greedy approach first to get a baseline, then maybe improve.
# Or, since the constraint is "pairwise coprime", we can think in terms of prime factors.
# Each chosen number must have a unique set of prime factors that doesn't overlap with others.
# Actually, it's simpler: no prime can divide two different numbers in the set.
# So, for each prime p, we can pick at most one number from the set of multiples of p.
# Wait, that's not quite right. If we pick 6 (2*3) and 35 (5*7), they are coprime.
# The constraint is that the sets of prime factors of any two chosen numbers must be disjoint.

# This is equivalent to finding a set of numbers with disjoint prime factor sets.
# This is a "packing" problem.

# Let's use a recursive search with pruning.
# Sort numbers in descending order of value.
# At each step, decide whether to include the current number or not.
# If we include it, we must exclude all numbers that share a prime factor with it.
# If we exclude it, we move to the next.

# To make this efficient:
# 1. Precompute prime factors for each number.
# 2. Sort numbers 2..256 in descending order.
# 3. Use a bitmask or a set to track which primes are "used".
#    If a number has any prime factor already used, it cannot be chosen.
# 4. Branch and bound:
#    - Upper bound: sum of all remaining numbers that are compatible with the current set of used primes.
#    - If current_sum + upper_bound <= best_sum, prune.

# Let's implement this.

# Precomputation
def get_prime_factors(n):
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

# Numbers 2..256
numbers = list(range(2, 257))
factors = {n: get_prime_factors(n) for n in numbers}

# Sort in descending order
numbers.sort(reverse=True)

# Map number to its index for faster lookup? Not strictly needed if we iterate.
# We need to check compatibility quickly.

# State:
# - current_sum
# - used_primes: set of primes already used
# - index: current index in the sorted list
# - best_sum: global best found

# Optimization:
# The set of used primes can be large.
# Instead of a set, we can use a bitmask if the number of primes is small enough.
# Primes up to 256: there are 54 primes.
# 54 bits fits in a 64-bit integer.
# Let's map each prime to a bit position.

primes = sorted(set(p for fs in factors.values() for p in fs))
prime_to_bit = {p: 1 << i for i, p in enumerate(primes)}

# Precompute bitmask for each number
masks = {}
for n in numbers:
    mask = 0
    for p in factors[n]:
        mask |= prime_to_bit[p]
    masks[n] = mask

# We also need to know which numbers are compatible with a given mask.
# But checking compatibility is just (masks[n] & used_mask) == 0.

# Branch and Bound
# We need a good upper bound.
# A simple upper bound is the sum of all numbers from index to end that are compatible.
# But calculating this every time might be expensive.
# A tighter bound: sum of the maximum compatible number for each "conflict class"?
# No, the conflict is based on primes.
# A better bound: For each prime p not in used_mask, we can pick at most one multiple of p.
# But a number can have multiple primes.
# This is a Weighted Set Packing problem.
# The LP relaxation is a valid upper bound.
# But solving LP in Python without scipy is hard.
# Let's stick to a greedy upper bound:
# Greedily pick the largest compatible number, then the next largest compatible with that, etc.
# This gives a valid solution, but as an upper bound, it's just a lower bound on the optimum?
# No, the greedy solution is a feasible solution, so it's a lower bound.
# We need an upper bound to prune.
# Upper bound: Sum of all remaining numbers that are compatible with used_mask.
# This is optimistic because it assumes we can pick all of them, which is impossible if they conflict.
# But it's fast to compute if we precompute suffix sums?
# No, compatibility depends on used_mask which changes.

# Let's try a simpler approach first: Greedy.
# Sort by value descending.
# Pick the largest number.
# Remove all numbers that share a prime factor with it.
# Repeat.
# This will give a valid set.
# Is it optimal? Probably not, but let's see.

def greedy_solution():
    used_mask = 0
    selected = []
    remaining = list(numbers)
    # We process in descending order
    for n in numbers:
        if (masks[n] & used_mask) == 0:
            selected.append(n)
            used_mask |= masks[n]
    return selected

# Let's run greedy and see the sum.
# Then we can try to improve it with local search or a limited branch and bound.

# Given the time limit (90s), and n=255, a full B&B might be too slow if the bound is weak.
# However, the structure of coprime numbers is sparse.
# Most large numbers are prime or have large prime factors, so they don't conflict with many others.
# For example, 256 = 2^8. It conflicts with all even numbers.
# 255 = 3*5*17. It conflicts with multiples of 3, 5, 17.
# The density of conflicts decreases as numbers get larger.

# Let's try a heuristic B&B.
# We will use a global variable for the best solution.
# We will sort numbers in descending order.
# We will use a recursive function.

# To speed up the upper bound, we can precompute for each index i and each possible used_mask?
# No, used_mask has 2^54 states.
# We can just calculate the sum of compatible remaining numbers.
# This is O(N) per node.
# If the tree is small, it's fine.

# Let's implement a time-limited search.
# Start with the greedy solution.
# Then try to improve it.

# Actually, let's just run the B&B with a time limit.
# If it finishes, great. If not, return the best found.

import sys
sys.setrecursionlimit(10000)

best_sum = 0
best_set = []
start_time = time.time()
time_limit = 85.0  # Leave 5s margin

def search(idx, current_sum, used_mask, current_set):
    global best_sum, best_set
    
    if time.time() - start_time > time_limit:
        return
    
    if idx == len(numbers):
        if current_sum > best_sum:
            best_sum = current_sum
            best_set = current_set[:]
        return
    
    n = numbers[idx]
    mask_n = masks[n]
    
    # Option 1: Exclude n
    search(idx + 1, current_sum, used_mask, current_set)
    
    # Option 2: Include n (if compatible)
    if (mask_n & used_mask) == 0:
        current_set.append(n)
        search(idx + 1, current_sum + n, used_mask | mask_n, current_set)
        current_set.pop()

# Initialize with greedy
greedy_set = greedy_solution()
best_sum = sum(greedy_set)
best_set = greedy_set[:]

# Now run B&B to try to beat greedy
# We need to be careful with the order.
# The B&B above explores "exclude" first.
# This might find a good solution quickly if the greedy was close.
# But we want to find the maximum.
# The bound is weak.
# Let's add a simple bound:
# Upper bound = current_sum + sum of all remaining numbers that are compatible with used_mask.
# We can precompute suffix sums? No, compatibility changes.
# Let's just compute it on the fly.

def upper_bound(idx, used_mask):
    ub = 0
    for i in range(idx, len(numbers)):
        if (masks[numbers[i]] & used_mask) == 0:
            ub += numbers[i]
    return ub

def search_bounded(idx, current_sum, used_mask, current_set):
    global best_sum, best_set
    
    if time.time() - start_time > time_limit:
        return
    
    # Prune
    if current_sum + upper_bound(idx, used_mask) <= best_sum:
        return
    
    if idx == len(numbers):
        if current_sum > best_sum:
            best_sum = current_sum
            best_set = current_set[:]
        return
    
    n = numbers[idx]
    mask_n = masks[n]
    
    # Option 2: Include n (if compatible) - Try include first to find good solutions quickly
    if (mask_n & used_mask) == 0:
        current_set.append(n)
        search_bounded(idx + 1, current_sum + n, used_mask | mask_n, current_set)
        current_set.pop()
        
    # Option 1: Exclude n
    search_bounded(idx + 1, current_sum, used_mask, current_set)

# Run search
search_bounded(0, 0, 0, [])

# Add 1 to the set
# 1 is coprime with everything, so we always include it.
final_set = [1] + best_set

# Verify
def is_valid(s):
    if not s:
        return False
    if len(set(s)) != len(s):
        return False
    for i in range(len(s)):
        for j in range(i+1, len(s)):
            if gcd(s[i], s[j]) != 1:
                return False
    return True

# assert is_valid(final_set)
# The problem states 1 is allowed once.
# The set contains distinct integers from 1..256.
# We constructed it from 2..256 and added 1.
# So it is valid.

def construct():
    # Return the list of integers
    return final_set
# EVOLVE-BLOCK-END
