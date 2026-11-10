"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 40, 'maximum_size': 18}

# Precompute prime factorizations for all numbers in [2, 40]
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def _factorize(num):
    """Return dict of prime->exponent for num."""
    factors = {}
    x = num
    for p in _PRIMES:
        if x % p == 0:
            cnt = 0
            while x % p == 0:
                x //= p
                cnt += 1
            factors[p] = cnt
    return factors

# Precompute factorizations
_FACTS = {n: _factorize(n) for n in range(2, 41)}

def _subset_products_distinct(nums):
    """Check if all subset products of nums are distinct."""
    k = len(nums)
    if k == 0:
        return True
    products = set()
    for mask in range(1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= nums[i]
        if prod in products:
            return False
        products.add(prod)
    return True

def _build_conflict_graph():
    """
    Build conflict graph where edge (i,j) means nums[i] and nums[j]
    cannot both be in a valid set (i.e., adding one breaks the other's
    subset product distinctness).
    
    Two numbers conflict if their prime exponent vectors are linearly
    dependent over the rationals in a way that causes subset product
    collisions. More precisely, a and b conflict if there exist non-zero
    integer coefficients such that a^x * b^y = 1 in the multiplicative
    group, which happens when log(a)/log(b) is rational, i.e., when
    a and b are powers of a common base.
    
    Actually, the correct condition: a set S has all distinct subset
    products iff the prime exponent vectors of elements in S are linearly
    independent over the integers (i.e., no non-trivial integer combination
    gives zero vector). This is equivalent to: for any two distinct subsets
    T1, T2 of S, product(T1) != product(T2), which means product(T1 \\ T2) != 
    product(T2 \\ T1), i.e., no non-empty subset has product 1 (impossible
    for positive integers > 1) ... wait, that's not right either.
    
    The correct condition: all subset products are distinct iff the vectors
    are linearly independent over the integers. That is, there is no non-trivial
    integer linear combination of the exponent vectors that equals zero.
    
    For two numbers a and b with exponent vectors v_a and v_b, they conflict
    if v_a and v_b are linearly dependent over Q, i.e., v_a = c * v_b for
    some rational c > 0. This means a = b^c, i.e., a and b are both powers
    of a common integer.
    """
    numbers = list(range(2, 41))
    n = len(numbers)
    # conflict[i][j] = True if numbers[i] and numbers[j] conflict
    conflict = [[False] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            ai = numbers[i]
            aj = numbers[j]
            fi = _FACTS[ai]
            fj = _FACTS[aj]
            
            # Check if fi and fj are linearly dependent over Q
            # i.e., fi = c * fj for some rational c
            # This means for all primes p: fi[p] / fj[p] is the same constant
            # (treating missing primes as 0)
            
            all_primes = set(fi.keys()) | set(fj.keys())
            ratio = None
            dependent = True
            
            for p in all_primes:
                ei = fi.get(p, 0)
                ej = fj.get(p, 0)
                
                if ei == 0 and ej == 0:
                    continue
                elif ei == 0 or ej == 0:
                    # One has this prime, the other doesn't -> not dependent
                    dependent = False
                    break
                else:
                    # Compute ratio ei/ej as fraction
                    from math import gcd
                    g = gcd(ei, ej)
                    num_r, den_r = ei // g, ej // g
                    
                    if ratio is None:
                        ratio = (num_r, den_r)
                    else:
                        if ratio != (num_r, den_r):
                            dependent = False
                            break
            
            if dependent:
                conflict[i][j] = True
                conflict[j][i] = True
    
    return numbers, conflict

def _mis_branch_and_bound(numbers, conflict, deadline):
    """
    Find Maximum Independent Set using branch and bound with bitmask.
    Independent set in conflict graph = set of non-conflicting numbers
    = set with all distinct subset products.
    """
    n = len(numbers)
    # Adjacency bitmasks for fast operations
    adj = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if conflict[i][j]:
                mask |= (1 << j)
        adj[i] = mask
    
    # Greedy upper bound: color the graph and use max color
    def _greedy_coloring(vertices):
        """Assign colors to vertices in the subgraph induced by vertices."""
        if not vertices:
            return 0
        colors = {}
        uncolored = set(vertices)
        color_idx = 0
        while uncolored:
            color_idx += 1
            # Pick a vertex with max degree in remaining uncolored
            best_v = max(uncolored, key=lambda v: sum(1 for u in uncolored if u != v and conflict[v][u]))
            # Assign this color to all non-adjacent vertices
            color_set = set()
            for v in list(uncolored):
                if v in color_set:
                    continue
                # Check if v can be added to this color class
                can_add = True
                for u in color_set:
                    if conflict[v][u]:
                        can_add = False
                        break
                if can_add:
                    color_set.add(v)
                    uncolored.discard(v)
                    colors[v] = color_idx
        return color_idx
    
    best = [0]
    best_set = [set()]
    
    def _search(candidates, current_set, depth):
        """Branch and bound search."""
        if time.time() > deadline:
            return
        
        # Prune: if current + remaining can't beat best, return
        if len(current_set) + len(candidates) <= best[0]:
            return
        
        if not candidates:
            if len(current_set) > best[0]:
                best[0] = len(current_set)
                best_set[0] = set(current_set)
            return
        
        # Upper bound: greedy coloring of candidates
        # For simplicity, use |candidates| as upper bound (weak but fast)
        # Better: use greedy coloring
        upper_bound = len(candidates)
        if len(current_set) + upper_bound <= best[0]:
            return
        
        # Pick a vertex to branch on
        # Heuristic: pick vertex with highest degree in candidates
        cand_list = list(candidates)
        # Sort by degree in candidates
        cand_list.sort(key=lambda v: sum(1 for u in candidates if u != v and conflict[v][u]), reverse=True)
        v = cand_list[0]
        
        # Branch 1: include v
        new_candidates = set()
        for u in candidates:
            if u != v and not conflict[v][u]:
                new_candidates.add(u)
        _search(new_candidates, current_set | {v}, depth + 1)
        
        # Branch 2: exclude v
        new_candidates2 = set(candidates)
        new_candidates2.discard(v)
        _search(new_candidates2, current_set, depth + 1)
    
    all_vertices = set(range(n))
    _search(all_vertices, set(), 0)
    
    return best_set[0]

def construct():
    deadline = time.time() + 50
    
    numbers, conflict = _build_conflict_graph()
    
    result_indices = _mis_branch_and_bound(numbers, conflict, deadline)
    
    result = [numbers[i] for i in sorted(result_indices)]
    
    # Verify the result
    if not _subset_products_distinct(result):
        # Fallback: just return primes
        result = list(_PRIMES)
    
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
