"""Hungarian algorithm for maximum weight independent set on prime-factor conflict graph."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt
from fractions import Fraction
import numpy as np

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    if n < 9:
        return True
    if n % 3 == 0:
        return False
    if n < 25:
        return True
    if n % 5 == 0:
        return False
    if n < 49:
        return True
    if n % 7 == 0:
        return False
    if n < 121:
        return True
    i = 11
    while i <= isqrt(n):
        if n % i == 0:
            return False
        i += 2
    return True

def _get_prime_factors(n):
    """Return list of unique prime factors of n."""
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def _hungarian(cost_matrix):
    """
    Solve assignment problem using Hungarian algorithm.
    cost_matrix: 2D array where we want to minimize cost.
    Returns: (assignment, min_cost) where assignment[i] = j means row i matched to col j.
    """
    n = len(cost_matrix)
    if n == 0:
        return [], 0.0
    
    # Convert to float for numerical stability
    C = np.array(cost_matrix, dtype=float)
    
    # Hungarian algorithm (Jonker-Volgenant style)
    u = np.zeros(n)  # row potentials
    v = np.zeros(n)  # col potentials
    p = np.zeros(n, dtype=int)  # p[j] = row assigned to col j
    way = np.zeros(n, dtype=int)  # way[j] = previous col in augmenting path
    
    for i in range(n):
        p[0] = i
        j0 = 0
        minv = np.full(n, np.inf)
        used = np.zeros(n, dtype=bool)
        
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = -1
            
            for j in range(n):
                if not used[j]:
                    cur = C[i0, j] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            
            for j in range(n):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            
            j0 = j1
            if p[j0] == 0:
                break
        
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    
    # Extract assignment: for each row i, find which col j has p[j] == i
    assignment = {}
    for j in range(n):
        if p[j] != 0:
            assignment[p[j]] = j
    
    min_cost = sum(C[i, assignment[i]] for i in range(n) if i in assignment)
    return assignment, min_cost

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    N = 512
    
    # Precompute values and prime factors for all numbers 1..511
    values = [Fraction(0)] * N
    prime_factors = [[] for _ in range(N)]
    is_prime_arr = [False] * N
    
    for a in range(1, N):
        values[a] = Fraction(1, N - a)
        prime_factors[a] = _get_prime_factors(a)
        is_prime_arr[a] = _is_prime(a)
        
    # Identify all primes less than 512
    primes = [a for a in range(1, N) if is_prime_arr[a]]
    
    # For the Hungarian formulation:
    # Left side: candidate integers a in 1..511
    # Right side: prime factors
    # We want to select a subset of integers such that no prime factor is shared.
    # This is equivalent to: assign each selected integer to its prime factors,
    # with each prime factor used at most once.
    # 
    # Since an integer can have multiple prime factors, we need to ensure all its
    # prime factors are available. We can model this as: for each integer a, it "costs"
    # the sum of its prime factors' opportunity costs.
    #
    # Alternative approach: Use the fact that pairwise coprimality means each prime
    # appears in at most one selected number. We can formulate as maximum weight
    # independent set, but that's NP-hard. Instead, use a relaxation.
    #
    # Better approach: Since the constraint is that no prime is shared, we can think
    # of it as: each prime p can be "assigned" to at most one number that has p as
    # a factor. For a number a with prime factors {p1, p2, ..., pk}, it can only be
    # selected if ALL of p1..pk are unassigned.
    #
    # Let's use a different formulation: Create a bipartite graph where left nodes
    # are integers and right nodes are primes. An edge exists from integer a to prime p
    # if p divides a. We want to select a set of integers such that no two share a prime.
    # This is equivalent to finding a matching in the dual sense.
    #
    # Actually, let's use a simpler approach: greedy with local search, but use
    # Hungarian to optimize the initial selection.
    
    # First, get a good initial solution using greedy
    selected = set()
    prime_used = set()
    
    # Sort by value descending
    candidates = list(range(1, N))
    candidates.sort(key=lambda x: values[x], reverse=True)
    
    for c in candidates:
        can_add = True
        for p in prime_factors[c]:
            if p in prime_used:
                can_add = False
                break
        if can_add:
            selected.add(c)
            for p in prime_factors[c]:
                prime_used.add(p)
    
    # Now use local search to improve
    # Build conflict structure
    prime_to_selected = {}
    for b in selected:
        for p in prime_factors[b]:
            if p not in prime_to_selected:
                prime_to_selected[p] = set()
            prime_to_selected[p].add(b)
    
    def get_score():
        return sum(values[a] for a in selected)
    
    current_score = get_score()
    
    # 1-for-k swap search
    improved = True
    while improved:
        if time.time() > deadline:
            break
            
        improved = False
        best_gain = Fraction(0)
        best_add = None
        best_remove = None
        
        for c in range(1, N):
            if c in selected:
                continue
            
            # Find conflicts
            conflict_set = set()
            for p in prime_factors[c]:
                if p in prime_to_selected:
                    conflict_set.update(prime_to_selected[p])
            
            conflict_cost = sum(values[b] for b in conflict_set)
            gain = values[c] - conflict_cost
            
            if gain > best_gain:
                best_gain = gain
                best_add = [c]
                best_remove = list(conflict_set)
        
        if best_add is not None:
            # Apply move
            for b in best_remove:
                for p in prime_factors[b]:
                    if p in prime_to_selected:
                        prime_to_selected[p].discard(b)
                        if not prime_to_selected[p]:
                            del prime_to_selected[p]
            for c in best_add:
                for p in prime_factors[c]:
                    if p not in prime_to_selected:
                        prime_to_selected[p] = set()
                    prime_to_selected[p].add(c)
            selected.difference_update(best_remove)
            selected.update(best_add)
            current_score += best_gain
            improved = True
    
    # 2-for-k swap search
    if time.time() < deadline - 15:
        composites = [a for a in range(1, N) if not is_prime_arr[a] and a > 1]
        composites.sort(key=lambda x: values[x], reverse=True)
        top_composites = composites[:80]
        
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
                
            improved = False
            best_gain = Fraction(0)
            best_add = None
            best_remove = None
            
            for i in range(len(top_composites)):
                c1 = top_composites[i]
                if c1 in selected:
                    continue
                for j in range(i + 1, len(top_composites)):
                    c2 = top_composites[j]
                    if c2 in selected:
                        continue
                    if gcd(c1, c2) != 1:
                        continue
                    
                    # Find conflicts
                    conflict_set = set()
                    for p in prime_factors[c1]:
                        if p in prime_to_selected:
                            conflict_set.update(prime_to_selected[p])
                    for p in prime_factors[c2]:
                        if p in prime_to_selected:
                            conflict_set.update(prime_to_selected[p])
                    
                    conflict_cost = sum(values[b] for b in conflict_set)
                    gain = values[c1] + values[c2] - conflict_cost
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_add = [c1, c2]
                        best_remove = list(conflict_set)
            
            if best_add is not None:
                for b in best_remove:
                    for p in prime_factors[b]:
                        if p in prime_to_selected:
                            prime_to_selected[p].discard(b)
                            if not prime_to_selected[p]:
                                del prime_to_selected[p]
                for c in best_add:
                    for p in prime_factors[c]:
                        if p not in prime_to_selected:
                            prime_to_selected[p] = set()
                        prime_to_selected[p].add(c)
                selected.difference_update(best_remove)
                selected.update(best_add)
                current_score += best_gain
                improved = True
    
    # 3-for-k swap search
    if time.time() < deadline - 10:
        composites = [a for a in range(1, N) if not is_prime_arr[a] and a > 1]
        composites.sort(key=lambda x: values[x], reverse=True)
        top_composites = composites[:50]
        
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
                
            improved = False
            best_gain = Fraction(0)
            best_add = None
            best_remove = None
            
            for i in range(len(top_composites)):
                c1 = top_composites[i]
                if c1 in selected:
                    continue
                for j in range(i + 1, len(top_composites)):
                    c2 = top_composites[j]
                    if c2 in selected:
                        continue
                    if gcd(c1, c2) != 1:
                        continue
                    for k in range(j + 1, len(top_composites)):
                        c3 = top_composites[k]
                        if c3 in selected:
                            continue
                        if gcd(c1, c3) != 1 or gcd(c2, c3) != 1:
                            continue
                        
                        # Find conflicts
                        conflict_set = set()
                        for c in (c1, c2, c3):
                            for p in prime_factors[c]:
                                if p in prime_to_selected:
                                    conflict_set.update(prime_to_selected[p])
                        
                        conflict_cost = sum(values[b] for b in conflict_set)
                        gain = values[c1] + values[c2] + values[c3] - conflict_cost
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_add = [c1, c2, c3]
                            best_remove = list(conflict_set)
            
            if best_add is not None:
                for b in best_remove:
                    for p in prime_factors[b]:
                        if p in prime_to_selected:
                            prime_to_selected[p].discard(b)
                            if not prime_to_selected[p]:
                                del prime_to_selected[p]
                for c in best_add:
                    for p in prime_factors[c]:
                        if p not in prime_to_selected:
                            prime_to_selected[p] = set()
                        prime_to_selected[p].add(c)
                selected.difference_update(best_remove)
                selected.update(best_add)
                current_score += best_gain
                improved = True
    
    # Final pass of 1-for-k to clean up
    if time.time() < deadline - 5:
        improved = True
        while improved:
            if time.time() > deadline - 1:
                break
            improved = False
            best_gain = Fraction(0)
            best_add = None
            best_remove = None
            
            for c in range(1, N):
                if c in selected:
                    continue
                conflict_set = set()
                for p in prime_factors[c]:
                    if p in prime_to_selected:
                        conflict_set.update(prime_to_selected[p])
                conflict_cost = sum(values[b] for b in conflict_set)
                gain = values[c] - conflict_cost
                if gain > best_gain:
                    best_gain = gain
                    best_add = [c]
                    best_remove = list(conflict_set)
            
            if best_add is not None:
                for b in best_remove:
                    for p in prime_factors[b]:
                        if p in prime_to_selected:
                            prime_to_selected[p].discard(b)
                            if not prime_to_selected[p]:
                                del prime_to_selected[p]
                for c in best_add:
                    for p in prime_factors[c]:
                        if p not in prime_to_selected:
                            prime_to_selected[p] = set()
                        prime_to_selected[p].add(c)
                selected.difference_update(best_remove)
                selected.update(best_add)
                current_score += best_gain
                improved = True
    
    # Final Verification
    result = list(selected)
    for a in result:
        if a < 1 or a > 511:
            raise ValueError(f"Out of range: {a}")
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                raise ValueError(f"Not coprime: {result[i]}, {result[j]}")
    
    return result
# EVOLVE-BLOCK-END
