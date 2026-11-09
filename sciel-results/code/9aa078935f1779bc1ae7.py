"""Program for the suite task; finds largest pairwise-coprime subset of 2..256."""
# EVOLVE-BLOCK-START
from math import isqrt
from functools import lru_cache

def _prime_factors(n):
    """Return the set of distinct prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

# Precompute prime factors for each number in 2..256
_N = 256
_NUMS = list(range(2, _N + 1))
_FACTORS = {x: _prime_factors(x) for x in _NUMS}
_IDX = {x: i for i, x in enumerate(_NUMS)}

# Build conflict graph: conflict[i][j] = True if _NUMS[i] and _NUMS[j] share a prime factor
_NN = len(_NUMS)
conflict = [[False] * _NN for _ in range(_NN)]
for i in range(_NN):
    fi = _FACTORS[_NUMS[i]]
    for j in range(i + 1, _NN):
        fj = _FACTORS[_NUMS[j]]
        if fi & fj:
            conflict[i][j] = True
            conflict[j][i] = True

# Degree of each vertex
_degree = [sum(conflict[i]) for i in range(_NN)]

# Greedy upper bound: for a set of candidate vertices, upper bound on independent set size
# We use a simple bound: number of candidates, but tighter via greedy coloring-like bound.
# A practical bound: sort candidates by degree, use a simple greedy matching bound.

def _greedy_upper_bound(candidates):
    """Upper bound on the size of an independent set within candidates.
    Uses a simple greedy coloring bound (like Tomita's bound)."""
    if not candidates:
        return 0
    # Greedy coloring: assign colors so that adjacent vertices get different colors.
    # The number of colors used is an upper bound on the independence number.
    color = {}
    max_color = 0
    # Sort by degree descending for better bounds
    for v in sorted(candidates, key=lambda x: -_degree[x]):
        # Find smallest color not used by neighbors
        used = set()
        for u in candidates:
            if u != v and conflict[v][u] and u in color:
                used.add(color[u])
        c = 0
        while c in used:
            c += 1
        color[v] = c
        if c > max_color:
            max_color = c
    return max_color

_best = []
_best_size = 0

def _branch_and_bound(candidates, current):
    """Branch and bound for maximum independent set."""
    global _best, _best_size
    if not candidates:
        if len(current) > _best_size:
            _best_size = len(current)
            _best = list(current)
        return
    # Upper bound
    ub = len(current) + _greedy_upper_bound(candidates)
    if ub <= _best_size:
        return
    # Pick a vertex to branch on (highest degree among candidates for better pruning)
    # Choose the vertex with highest degree in the candidate subgraph
    v = max(candidates, key=lambda x: sum(1 for u in candidates if u != x and conflict[x][u]))
    # Branch 1: include v
    current.append(v)
    # New candidates: those not conflicting with v
    new_candidates = [u for u in candidates if u != v and not conflict[v][u]]
    _branch_and_bound(new_candidates, current)
    current.pop()
    # Branch 2: exclude v
    new_candidates2 = [u for u in candidates if u != v]
    _branch_and_bound(new_candidates2, current)

def construct():
    global _best, _best_size
    # Initialize with a greedy solution to have a good lower bound
    greedy = []
    remaining = list(range(_NN))
    while remaining:
        # Pick vertex with smallest degree in remaining (to leave more options)
        v = min(remaining, key=lambda x: sum(1 for u in remaining if u != x and conflict[x][u]))
        greedy.append(v)
        remaining = [u for u in remaining if u != v and not conflict[v][u]]
    _best = list(greedy)
    _best_size = len(greedy)
    
    # Now run branch and bound to find the optimal
    # Order candidates by degree for better performance
    all_candidates = list(range(_NN))
    _branch_and_bound(all_candidates, [])
    
    # Convert indices back to values
    result = [_NUMS[i] for i in _best]
    return result

# EVOLVE-BLOCK-END
