"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 85.0  # safety margin before 90s wall-clock limit

def _bad_triple(a, b, c):
    """Return True if the triple (a,b,c) has all three pairwise gcds equal."""
    g_ab = math.gcd(a, b)
    g_ac = math.gcd(a, c)
    g_bc = math.gcd(b, c)
    return g_ab == g_ac == g_bc

def _is_valid_set(S):
    """Check if set S has no bad triples."""
    S_list = list(S)
    for a, b, c in combinations(S_list, 3):
        if _bad_triple(a, b, c):
            return False
    return True

def _greedy_add_order(order, S_init=None):
    """Greedy construction: add elements in given order if valid."""
    S = list(S_init) if S_init else []
    remaining = list(order)
    for x in remaining:
        if (time.time() - _START_TIME) > _DEADLINE:
            break
        if all(not _bad_triple(a, b, x) for a, b in combinations(S, 2)):
            S.append(x)
    return S

def _local_improve(S, max_iterations=500):
    """Try to improve set S by removing one element and re-adding others."""
    S = list(S)
    best_size = len(S)
    full = list(range(1, 129))
    
    for _ in range(max_iterations):
        if (time.time() - _START_TIME) > _DEADLINE:
            break
        if len(S) >= 128:
            break
        
        # Try removing each element and see if we can add more
        improved = False
        for i in range(len(S)):
            candidate = S[:i] + S[i+1:]
            # Try to add elements not in candidate
            added_any = False
            for x in full:
                if x in candidate:
                    continue
                if all(not _bad_triple(a, b, x) for a, b in combinations(candidate, 2)):
                    candidate.append(x)
                    added_any = True
            
            if len(candidate) > len(S):
                S = candidate
                best_size = len(S)
                improved = True
                break
        
        if not improved:
            break
    
    return S

def construct():
    """Construct a large subset of [1,128] with no three distinct elements
    whose three pairwise gcds are all equal."""
    random.seed(42)
    full = list(range(1, 129))
    best_set = None
    best_size = 0
    
    def within_deadline():
        return (time.time() - _START_TIME) < _DEADLINE
    
    # Strategy 1: Structured construction via prime powers
    def structured_prime_powers():
        """Take all prime powers up to 128 plus 1."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
        prime_powers = set()
        for p in primes:
            val = p
            while val <= 128:
                prime_powers.add(val)
                val *= p
        prime_powers.add(1)
        
        S = list(prime_powers)
        if _is_valid_set(S):
            return S
        return None
    
    # Strategy 2: Greedy max conflict removal
    def greedy_max_conflict():
        S = set(full)
        # Precompute all bad triples
        bad_triples = []
        for a, b, c in combinations(full, 3):
            if _bad_triple(a, b, c):
                bad_triples.append((a, b, c))
        
        # For each node, count how many bad triples it participates in
        node_conflict = {x: 0 for x in full}
        for a, b, c in bad_triples:
            node_conflict[a] += 1
            node_conflict[b] += 1
            node_conflict[c] += 1
        
        active_triples = set(bad_triples)
        
        # Remove nodes in order of highest conflict count
        order = sorted(full, key=lambda x: node_conflict[x], reverse=True)
        for v in order:
            if not within_deadline():
                break
            if node_conflict[v] == 0:
                continue
            S.discard(v)
            # Update: invalidate all active triples containing v
            to_remove = []
            for triple in active_triples:
                if v in triple:
                    to_remove.append(triple)
                    other_nodes = [x for x in triple if x != v]
                    for u in other_nodes:
                        node_conflict[u] = max(0, node_conflict[u] - 1)
            for triple in to_remove:
                active_triples.discard(triple)
        
        return list(S)
    
    # Strategy 3: Randomized greedy construction
    def random_greedy_add():
        order = list(full)
        random.shuffle(order)
        return _greedy_add_order(order)
    
    # Strategy 4: Greedy with local improvement
    def greedy_with_local_improve():
        # Start with a greedy construction
        order = list(full)
        random.shuffle(order)
        S = _greedy_add_order(order)
        # Local improvement
        S = _local_improve(S)
        return S
    
    # Run strategies
    # Strategy 1 first (fast, structured)
    if within_deadline():
        S_struct = structured_prime_powers()
        if S_struct is not None and len(S_struct) > best_size:
            best_size = len(S_struct)
            best_set = S_struct
    
    # Strategy 2 (greedy max conflict)
    if within_deadline():
        S_greedy = greedy_max_conflict()
        if len(S_greedy) > best_size:
            best_size = len(S_greedy)
            best_set = S_greedy
    
    # Strategy 3 (random greedy add) - try several times
    if within_deadline():
        for _ in range(30):
            if not within_deadline():
                break
            S_rand = random_greedy_add()
            if len(S_rand) > best_size:
                best_size = len(S_rand)
                best_set = S_rand
    
    # Strategy 4 (greedy with local improvement)
    if within_deadline():
        for _ in range(10):
            if not within_deadline():
                break
            S_improved = greedy_with_local_improve()
            if len(S_improved) > best_size:
                best_size = len(S_improved)
                best_set = S_improved
    
    # Final: ensure best_set is nonempty and valid
    if best_set is None or len(best_set) == 0:
        return [1]
    
    # Verify the best set is valid
    if not _is_valid_set(best_set):
        # Find a smaller valid subset
        S_list = list(best_set)
        for i in range(len(S_list)):
            candidate = S_list[:i] + S_list[i+1:]
            if _is_valid_set(candidate):
                best_set = candidate
                break
        else:
            return [1]
    
    return best_set
# EVOLVE-BLOCK-END
