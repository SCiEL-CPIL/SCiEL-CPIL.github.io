"""Combined program: MIS branch-and-bound on conflict graph + greedy fallback."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 40, 'maximum_size': 18}

def _is_prime(a):
    if a < 2:
        return False
    if a < 4:
        return True
    if a % 2 == 0 or a % 3 == 0:
        return False
    d = 5
    while d * d <= a:
        if a % d == 0 or a % (d + 2) == 0:
            return False
        d += 6
    return True

def _check_valid(S):
    """Validate that all subset products of S are distinct."""
    current = {1}
    for x in S:
        new_vals = {p * x for p in current}
        if current & new_vals:
            return False
        current |= new_vals
    return True

def _build_conflict_graph():
    """
    Build conflict graph: edge between i,j if including both creates a collision.
    Two numbers a,b conflict if there exist non-empty subsets A of {a} and B of {b}
    such that product(A) = product(B). For single elements, this means a=b (excluded)
    or one divides the other in a way that creates dependency.
    
    Actually, the conflict is: a and b conflict if a|b or b|a is NOT the issue.
    The real issue: including a and b means we need a, b, a*b all distinct from 1 and each other,
    AND for larger sets, any subset product must be unique.
    
    For two elements {a,b}: products are 1, a, b, ab. These are distinct iff a!=b, a!=1, b!=1, ab!=a (b!=1), ab!=b (a!=1), ab!=1.
    So for pairs, any two distinct integers >= 2 are compatible!
    
    The conflicts arise with 3+ elements. Let me think differently.
    
    Actually, the constraint is that ALL 2^k subset products must be distinct. This is equivalent to
    the numbers being "multiplicatively independent" in the sense that no non-trivial product
    of a subset equals a product of another disjoint subset.
    
    For the conflict graph approach, we need: two numbers conflict if they cannot both be in a
    valid set. But actually, any pair is valid. The issue is with triples and larger.
    
    Let me use a different approach: build the graph where an edge means "these two cannot coexist
    in ANY valid set of size >= 3". But that's complex.
    
    Alternative: use the incremental check as the validity test in branch and bound.
    """
    n = P['n']
    candidates = list(range(2, n + 1))
    m = len(candidates)
    
    # For now, use pairwise compatibility: all pairs are compatible (as shown above)
    # The real conflicts are higher-order. We'll handle this in the B&B with incremental checking.
    
    # Actually, let's identify pairwise conflicts more carefully.
    # a and b conflict if there exist subsets A,B of {a,b} (non-empty, disjoint) with prod(A)=prod(B)
    # Subsets of {a,b}: {a}, {b}, {a,b}
    # Pairs of disjoint non-empty subsets: ({a},{b}), ({a},{a,b} - not disjoint), ({b},{a,b} - not disjoint)
    # So only ({a},{b}) matters: a == b, which is excluded.
    # So NO pairwise conflicts!
    
    # The conflicts are higher-order. For B&B, we'll use incremental product-set checking.
    return candidates, m

def _mis_branch_and_bound(candidates, deadline):
    """
    Maximum Independent Set via branch and bound.
    Since there are no pairwise conflicts, we need a different formulation.
    
    Let's reformulate: we want the largest subset S of candidates such that all 2^|S|
    subset products are distinct.
    
    We'll use a recursive search with pruning:
    - Maintain current set and current product set
    - Try adding each remaining candidate
    - Prune if current_size + remaining_count <= best_size
    
    This is essentially a backtracking search with good pruning.
    """
    n = P['n']
    best = []
    best_size = 0
    
    def backtrack(remaining, current_set, product_set):
        nonlocal best, best_size
        
        # Pruning: even if we add all remaining, can we beat best?
        if len(current_set) + len(remaining) <= best_size:
            return
        
        # Update best if current is larger
        if len(current_set) > best_size:
            best = list(current_set)
            best_size = len(current_set)
            if best_size >= 18:
                return
        
        # Try adding each candidate from remaining
        for i in range(len(remaining)):
            x = remaining[i]
            # Check if x can be added
            new_prods = {p * x for p in product_set}
            if product_set & new_prods:
                continue  # Conflict
            
            # Recurse with x added
            new_remaining = remaining[:i] + remaining[i+1:]
            backtrack(new_remaining, current_set + [x], product_set | new_prods)
            
            # Check deadline
            if time.time() > deadline:
                return
    
    # Start with empty set
    start_time = time.time()
    backtrack(list(candidates), [], {1})
    
    return best

def _greedy_with_incremental(candidates, order, deadline):
    """Greedily add elements in given order, maintaining product set incrementally."""
    current_set = []
    product_set = {1}
    
    for x in order:
        if time.time() > deadline:
            break
        new_prods = {p * x for p in product_set}
        if product_set & new_prods:
            continue
        current_set.append(x)
        product_set |= new_prods
    
    return current_set

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill
    
    candidates = list(range(2, P['n'] + 1))
    primes = [a for a in candidates if _is_prime(a)]
    composites = [a for a in candidates if not _is_prime(a)]
    
    # Start with primes as baseline
    best_set = primes[:]
    best_size = len(best_set)
    
    # Try MIS-style backtracking with pruning
    # Since there are no pairwise conflicts, use backtracking with product-set pruning
    def backtrack_search(remaining, current_set, product_set, depth=0):
        nonlocal best_set, best_size
        
        # Deadline check
        if time.time() > deadline:
            return
        
        # Pruning
        if len(current_set) + len(remaining) <= best_size:
            return
        
        # Update best
        if len(current_set) > best_size:
            best_set = list(current_set)
            best_size = len(current_set)
            if best_size >= 18:
                return
        
        # Try adding candidates
        # Order: try composites first (more likely to cause conflicts, so prune early)
        # Actually, try in a smart order: larger numbers first (less likely to divide others)
        for i in range(len(remaining)):
            if time.time() > deadline:
                return
            
            x = remaining[i]
            new_prods = {p * x for p in product_set}
            if product_set & new_prods:
                continue
            
            new_remaining = remaining[:i] + remaining[i+1:]
            backtrack_search(new_remaining, current_set + [x], product_set | new_prods, depth + 1)
    
    # Run backtracking search
    # Start with primes, then try adding composites
    initial_product_set = {1}
    for p in primes:
        initial_product_set |= {pp * p for pp in initial_product_set}
    
    backtrack_search(composites, primes[:], initial_product_set)
    
    # If we haven't found a good solution, try randomized greedy restarts
    remaining_time = deadline - time.time()
    if remaining_time > 1.0 and best_size < 18:
        rng = random.Random(42)
        while time.time() < deadline - 0.5:
            # Random order: primes first, then shuffled composites
            c_shuffled = composites[:]
            rng.shuffle(c_shuffled)
            order = primes + c_shuffled
            sol = _greedy_with_incremental(candidates, order, deadline)
            if len(sol) > best_size:
                best_size = len(sol)
                best_set = sol[:]
            
            # Try a few more random orders
            for _ in range(3):
                if time.time() > deadline - 0.5:
                    break
                rng.shuffle(order)
                sol = _greedy_with_incremental(candidates, order, deadline)
                if len(sol) > best_size:
                    best_size = len(sol)
                    best_set = sol[:]
    
    # Final validation
    if not _check_valid(best_set):
        best_set = primes[:]
        best_size = len(best_set)
    
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
