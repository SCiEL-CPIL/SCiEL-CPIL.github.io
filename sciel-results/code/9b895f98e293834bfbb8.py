"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

_START_TIME = time.time()
_DEADLINE = _START_TIME + 55.0  # Safety margin for 60s limit

def construct():
    """
    Returns a list of distinct integers in 2..40 such that all 2^k subset products are distinct.
    Maximizes the number of selected integers.
    
    Strategy:
    1. Model the problem using prime exponent vectors. Two subset products are distinct iff their exponent vectors are distinct.
    2. The condition is that all 2^k subset sums of the selected exponent vectors are distinct.
    3. Use a greedy heuristic to find a large initial set.
    4. Use a branch-and-bound search to try to improve upon the initial set.
    5. The search space is reduced by sorting candidates by a heuristic (e.g., norm of the vector) and using symmetry breaking.
    """
    
    candidates = list(range(2, 41))
    
    # 1. Precompute prime exponent vectors for each candidate
    # Primes up to 40: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    def get_vector(n):
        v = [0] * num_primes
        for i, p in enumerate(primes):
            while n % p == 0:
                v[i] += 1
                n //= p
        return tuple(v)
        
    vectors = {x: get_vector(x) for x in candidates}
    
    # 2. Greedy construction to get a good initial lower bound
    # We want to add vectors that are "far" from existing subset sums?
    # Or simply try to add as many as possible.
    # A simple greedy: sort by norm (sum of exponents) descending? Or ascending?
    # Let's try sorting by the number of distinct primes (support size) ascending, then by norm.
    # Actually, numbers with fewer prime factors are "sparser" and less likely to cause collisions?
    # Let's try a few heuristics or just a simple one.
    
    def heuristic_score(v):
        # Smaller norm is generally better?
        # Or maybe we want to prioritize primes?
        # Let's try sorting by the value itself, or by norm.
        # Let's try sorting by norm ascending.
        return sum(v)
        
    sorted_candidates = sorted(candidates, key=lambda x: (heuristic_score(vectors[x]), x))
    
    current_set = []
    current_sums = {tuple([0]*num_primes)}  # Set of tuple exponent vectors
    
    def can_add(v, sums):
        # Check if any sum + v is already in sums
        # Since sums is a set of tuples, we can iterate
        new_sums = set()
        for s in sums:
            nv = tuple(s[i] + v[i] for i in range(num_primes))
            if nv in sums:
                return False
            new_sums.add(nv)
        return True
        
    for x in sorted_candidates:
        if time.time() > _DEADLINE:
            break
        v = vectors[x]
        if can_add(v, current_sums):
            current_set.append(x)
            # Update sums
            new_sums = set(current_sums)
            for s in current_sums:
                nv = tuple(s[i] + v[i] for i in range(num_primes))
                new_sums.add(nv)
            current_sums = new_sums
            
    best_size = len(current_set)
    best_set = list(current_set)
    
    # 3. Branch and Bound to improve
    # We try to swap out elements or add more?
    # Actually, the greedy might be suboptimal.
    # Let's do a local search or a full backtrack with better pruning.
    # Given the small size (39 candidates), a full backtrack might be feasible with good pruning.
    # But the state space of sums is large (2^k vectors).
    
    # Let's try a randomized local search / hill climbing to escape local optima if greedy is stuck.
    # Or a simple backtrack that tries to find a set of size best_size + 1.
    
    # A more robust approach: Iterative Deepening or just a well-optimized Backtrack.
    # The previous program used backtrack. Let's try to make it faster.
    # The key is the collision check.
    
    # Let's try a different heuristic for ordering in backtrack.
    # Order candidates by "difficulty".
    # Candidates with more prime factors are harder to place?
    # Let's sort by norm descending for the search?
    
    search_candidates = sorted(candidates, key=lambda x: -heuristic_score(vectors[x]))
    
    def backtrack(start_idx, current_set, current_sums, current_size):
        nonlocal best_set, best_size
        
        if time.time() > _DEADLINE:
            return
            
        if current_size > best_size:
            best_size = current_size
            best_set = list(current_set)
            
        # Pruning
        if current_size + (len(search_candidates) - start_idx) <= best_size:
            return
            
        for i in range(start_idx, len(search_candidates)):
            if time.time() > _DEADLINE:
                return
                
            x = search_candidates[i]
            v = vectors[x]
            
            # Check collision
            can_add_flag = True
            # Optimization: Pre-check?
            for s in current_sums:
                nv = tuple(s[j] + v[j] for j in range(num_primes))
                if nv in current_sums:
                    can_add_flag = False
                    break
                    
            if can_add_flag:
                # Add
                new_sums = set(current_sums)
                for s in current_sums:
                    nv = tuple(s[j] + v[j] for j in range(num_primes))
                    new_sums.add(nv)
                    
                current_set.append(x)
                backtrack(i + 1, current_set, new_sums, current_size + 1)
                current_set.pop()
                
    # Start backtrack with the greedy result? Or empty?
    # Starting with empty allows finding a different optimal set if greedy was bad.
    # But starting with greedy is faster if greedy is good.
    # Let's start with empty to be safe, but use the greedy size as lower bound.
    
    # Wait, the greedy set is a valid set. We can use it to prune.
    # But we need to search for a set of size > best_size.
    # So we can start backtrack with empty set, but prune if current_size + remaining <= best_size.
    
    initial_sums = {tuple([0]*num_primes)}
    backtrack(0, [], initial_sums, 0)
    
    # Ensure types are correct
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
