import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 270.0

# Pre-compute a frequency table of pair-sums for each dimension
# pair_sum_table[n][sum_vector] = count of pairs (a,b) in the greedy set with a+b ≡ sum_vector
# This allows O(n) conflict counting per priority call

def _build_pair_sum_table(n, max_iters=50000):
    """Build a pair-sum frequency table using a fast greedy heuristic.
    
    The greedy set is built by prioritizing vectors with few zeros (more 1s and 2s)
    and balanced 1/2 counts, which empirically yields larger cap sets.
    """
    if n < 2 or n > 12:
        return None
    
    # Generate all vectors sorted by a strong heuristic
    all_vectors = []
    for i in range(3 ** n):
        v = []
        x = i
        for _ in range(n):
            v.append(x % 3)
            x //= 3
        all_vectors.append(tuple(v))
    
    # Heuristic: prefer vectors with fewer zeros, balanced 1s and 2s
    def _heuristic(v):
        c0 = v.count(0)
        c1 = v.count(1)
        c2 = v.count(2)
        imbalance = abs(c1 - c2)
        return -100.0 * c0 - 10.0 * imbalance - 0.1 * (c1 * c1 + c2 * c2)
    
    all_vectors.sort(key=_heuristic, reverse=True)
    
    # Greedy build with pair-sum tracking
    cap_set = []
    cap_set_set = set()
    # pair_sum_freq: dict mapping tuple(sum_vec) -> count
    pair_sum_freq = {}
    
    for v in all_vectors:
        if time.time() > _DEADLINE - 30:
            break
        
        # Count conflicts: number of pairs (a,b) in cap_set with a+b ≡ -v (mod 3)
        neg_v = tuple((-x) % 3 for x in v)
        conflicts = pair_sum_freq.get(neg_v, 0)
        
        if conflicts == 0:
            # Add v to cap set
            cap_set.append(v)
            cap_set_set.add(v)
            
            # Update pair_sum_freq: for each existing a in cap_set, add (a+v) to freq
            for a in cap_set[:-1]:  # exclude v itself
                s = tuple((a[i] + v[i]) % 3 for i in range(n))
                pair_sum_freq[s] = pair_sum_freq.get(s, 0) + 1
    
    return pair_sum_freq, cap_set

# Pre-compute pair sum tables at module load
_PAIR_SUM_TABLES = {}
for _n in range(2, 13):
    try:
        result = _build_pair_sum_table(_n)
        if result is not None:
            _PAIR_SUM_TABLES[_n] = result
    except:
        pass

def _hash_vec(v):
    h = 1469598103934665603
    for x in v:
        h = (h ^ x) * 1099511628211
    h = (h ^ (h >> 32)) & 0xFFFFFFFF
    return float(h)

def priority(vector, n):
    if time.time() > _DEADLINE:
        return -1e10
    
    h = _hash_vec(vector)
    
    if n in _PAIR_SUM_TABLES:
        pair_sum_freq, _ = _PAIR_SUM_TABLES[n]
        
        # Count conflicts: pairs (a,b) with a+b ≡ -vector (mod 3)
        neg_v = tuple((-x) % 3 for x in vector)
        conflicts = pair_sum_freq.get(neg_v, 0)
        
        # Strong penalty for conflicts, plus structural features
        c0 = 0
        c1 = 0
        c2 = 0
        for x in vector:
            if x == 0:
                c0 += 1
            elif x == 1:
                c1 += 1
            else:
                c2 += 1
        
        imbalance = abs(c1 - c2)
        
        # Score: heavily penalize conflicts, then structural quality
        score = -1000.0 * conflicts - 50.0 * c0 - 10.0 * imbalance - 0.1 * (c1 * c1 + c2 * c2)
        
        return score + h / (3 ** n)
    
    # Fallback for dimensions without pre-computed table
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    imbalance = abs(c1 - c2)
    
    return h - 50.0 * c0 - 10.0 * imbalance - 0.1 * (c1 * c1 + c2 * c2)
