"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import product

def _check_cap_set_valid(vec, accepted, n):
    """Check if adding vec to accepted keeps it a cap set (no 3-term APs)."""
    for i in range(n):
        a = vec[i]
        for j in range(i + 1, n):
            b = vec[j]
            # Check if there exists k such that a, b, k form an AP in some coordinate
            # Actually, we need to check if vec with any two existing vectors forms a 3-AP
            # A 3-AP in F_3^n is a triple (x, y, z) such that x + z = 2y (mod 3)
            # Equivalently, for any two distinct elements u, v in the set, 
            # the "midpoint" m where 2m = u+v mod 3 should not be in the set.
            # But we're checking if vec can be added, so we need to check:
            # For each pair (vec, w) in accepted: does the third point of the AP exist in accepted?
            # Or: does vec form an AP with two existing points?
            pass
    # Proper check: for each pair of existing points (p, q), check if vec is the midpoint
    # or if p is midpoint of (vec, q), etc.
    # Actually the simplest: for each w in accepted, check if vec, w, and some existing point form AP
    # For 3 elements a, b, c to form a 3-AP in F_3^n: a + c = 2b (mod 3)
    # So for candidate vec and two existing points p, q:
    #   Case 1: p + q = 2*vec  => vec = (p+q)*2^{-1} = (p+q)*2 mod 3... wait
    #   In F_3, 2 = -1, so 2*vec = p+q means vec = (p+q)*inv(2) = (p+q)*2 mod 3
    #   Case 2: p + vec = 2*q  => q = (p+vec)*2 mod 3, check if q in set (it is, since q is existing)
    #   Case 3: q + vec = 2*p  => p = (q+vec)*2 mod 3, check if p in set
    # But we only need to check if adding vec creates ANY new 3-AP.
    # A new 3-AP must include vec. So we need pairs (vec, w) where the third point is also in accepted ∪ {vec}.
    # Since vec is new, the third point must be in accepted.
    # For each w in accepted:
    #   Check if there exists v in accepted such that {vec, w, v} is a 3-AP
    #   Three possibilities for which is the middle:
    #     1. vec is middle: 2*vec = w + v => v = 2*vec - w
    #     2. w is middle: 2*w = vec + v => v = 2*w - vec
    #     3. v is middle: 2*v = vec + w => v = (vec + w) * inv(2) = (vec+w)*2 mod 3
    #   In all cases, we compute the required third point and check if it's in accepted.
    
    vec_set = set(accepted)
    for w in vec_set:
        # Case 1: vec is middle, need v = 2*vec - w mod 3
        v1 = tuple((2 * vec[k] - w[k]) % 3 for k in range(n))
        if v1 in vec_set:
            return False
        # Case 2: w is middle, need v = 2*w - vec mod 3
        v2 = tuple((2 * w[k] - vec[k]) % 3 for k in range(n))
        if v2 in vec_set:
            return False
        # Case 3: v is middle, need v = (vec + w) * 2 mod 3
        v3 = tuple(((vec[k] + w[k]) * 2) % 3 for k in range(n))
        if v3 in vec_set:
            return False
    return True


def _balanced_priority(vector, n):
    """Priority that favors vectors with balanced coordinate distribution."""
    counts = [0, 0, 0]
    for v in vector:
        counts[v] += 1
    # Penalize imbalance: reward when counts are as equal as possible
    # Max possible balance: each count ~ n/3
    max_bal = max(counts) - min(counts)
    return -max_bal  # Lower imbalance = higher priority


def run_greedy(n, deadline):
    """
    Greedy algorithm for cap set construction.
    Returns the size of the largest cap set found.
    """
    accepted = set()
    
    # Generate all possible vectors
    all_vectors = list(product(range(3), repeat=n))
    
    # Shuffle with a seed for reproducibility but also use balanced priority
    random.seed(42)
    random.shuffle(all_vectors)
    
    # Sort by balanced priority, with random tie-breaking
    # We'll do a partial approach: sort chunks
    for vec in all_vectors:
        if time.time() > deadline:
            break
        if _check_cap_set_valid(vec, accepted, n):
            accepted.add(vec)
    
    return len(accepted)


def run_greedy_dynamic(n, deadline):
    """
    Dynamic greedy: at each step, sample candidates and pick the one
    that maximizes a diversity heuristic among valid candidates.
    """
    accepted = set()
    accepted_list = []
    
    batch_size = 200
    rng = random.Random(12345)
    
    while time.time() < deadline:
        # Generate a batch of random candidates
        candidates = []
        for _ in range(batch_size):
            candidates.append(tuple(rng.randint(0, 2) for _ in range(n)))
        
        valid_candidates = []
        for vec in candidates:
            if vec not in accepted and _check_cap_set_valid(vec, accepted, n):
                valid_candidates.append(vec)
        
        if not valid_candidates:
            # Try more random candidates
            for _ in range(batch_size * 5):
                vec = tuple(rng.randint(0, 2) for _ in range(n))
                if vec not in accepted and _check_cap_set_valid(vec, accepted, n):
                    valid_candidates.append(vec)
        
        if not valid_candidates:
            break
        
        # Pick the candidate with the best "diversity" score
        # Diversity: minimize the number of 3-APs that would form if we added this vector
        # Actually all valid candidates form 0 APs, so use balanced coordinate priority
        best_vec = max(valid_candidates, key=lambda v: _balanced_priority(v, n))
        accepted.add(best_vec)
        accepted_list.append(best_vec)
    
    return len(accepted)


def run_greedy_static(n, deadline):
    """
    Static greedy with a better initial ordering.
    Sort all vectors by a composite priority before doing the greedy pass.
    """
    accepted = set()
    
    all_vectors = list(product(range(3), repeat=n))
    
    # Create a priority score for each vector
    def composite_priority(vec):
        # Balanced coordinates
        counts = [0, 0, 0]
        for v in vec:
            counts[v] += 1
        imbalance = max(counts) - min(counts)
        
        # Sum-based: prefer moderate sums
        total = sum(vec)
        ideal_sum = n  # n*1 = n, since each coordinate averages to 1
        sum_penalty = abs(total - ideal_sum)
        
        # Hash-based randomness for tie-breaking
        h = hash(vec) % 10000
        
        return (-imbalance, -sum_penalty, h)
    
    all_vectors.sort(key=composite_priority)
    
    for vec in all_vectors:
        if time.time() > deadline:
            break
        if _check_cap_set_valid(vec, accepted, n):
            accepted.add(vec)
    
    return len(accepted)


def priority(vector, n):
    """
    Entry point: returns a priority value for a vector.
    The evaluator uses this to greedily add vectors in descending priority order.
    We return a value that reflects balanced coordinate distribution.
    """
    counts = [0, 0, 0]
    for v in vector:
        counts[v] += 1
    imbalance = max(counts) - min(counts)
    return -float(imbalance)
# EVOLVE-BLOCK-END
