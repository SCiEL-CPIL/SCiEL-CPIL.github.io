"""Maximum Weight Independent Set on 3-AP hypergraph via specialized branch-and-bound."""

from fractions import Fraction
import time

N = 256

def construct():
    start_time = time.time()
    
    # 1. Precompute conflict pairs
    # conflict_pairs[v] is a list of tuples (a, b) with a < b such that {a, b, v} is a 3-AP.
    # We use a list of lists for faster iteration than sets.
    conflict_pairs = [[] for _ in range(N + 1)]
    
    for v in range(1, N + 1):
        # v as smallest: (v, v+d, v+2d) -> pair (v+d, v+2d)
        d = 1
        while v + 2 * d <= N:
            conflict_pairs[v].append((v + d, v + 2 * d))
            d += 1
        # v as middle: (v-d, v, v+d) -> pair (v-d, v+d)
        d = 1
        while v - d >= 1:
            conflict_pairs[v].append((v - d, v + d))
            d += 1
        # v as largest: (v-2d, v-d, v) -> pair (v-2d, v-d)
        d = 1
        while v - 2 * d >= 1:
            conflict_pairs[v].append((v - 2 * d, v - d))
            d += 1
            
    # Sort conflict pairs by frequency? Or just leave as is.
    # Convert to list for fast access.
    
    # 2. Order vertices by decreasing weight (1/x)
    # Weight is 1/x, so order is 1, 2, ..., 256.
    order = list(range(1, N + 1))
    
    # 3. Precompute weights and suffix sums using Fraction
    weights = [Fraction(0)] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = Fraction(1, i)
        
    suffix_sum = [Fraction(0)] * (N + 2)
    for i in range(N - 1, -1, -1):
        v = order[i]
        suffix_sum[i] = suffix_sum[i + 1] + weights[v]
    
    # 4. Greedy initialization
    def can_add(v, included_set):
        # Check if any conflict pair is fully contained in included_set
        for a, b in conflict_pairs[v]:
            if a in included_set and b in included_set:
                return False
        return True

    greedy_set = []
    greedy_included = set()
    for v in order:
        if can_add(v, greedy_included):
            greedy_set.append(v)
            greedy_included.add(v)
            
    best_set = list(greedy_set)
    best_weight = sum(weights[v] for v in greedy_set)
    
    # 5. Branch and Bound
    # We use a recursive B&B.
    # To speed it up, we can use a heuristic: if we include v, we might prune more.
    # We also check time limit.
    
    import sys
    sys.setrecursionlimit(10000)
    
    def bnb(i, included, included_set, current_weight):
        nonlocal best_set, best_weight
        
        # Time check
        if time.time() - start_time > 58:
            return
        
        # Base case
        if i == N:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = list(included)
            return
        
        # Upper bound: current weight + sum of all remaining weights
        # This is a weak bound, but necessary.
        # Stronger bound: if we include v, we can't include any u that conflicts with v AND some w in S?
        # No, the conflict is hypergraph.
        
        ub = current_weight + suffix_sum[i]
        if ub <= best_weight:
            return
            
        v = order[i]
        
        # Option 1: Include v
        if can_add(v, included_set):
            included.append(v)
            included_set.add(v)
            bnb(i + 1, included, included_set, current_weight + weights[v])
            included.pop()
            included_set.discard(v)
            
        # Option 2: Exclude v
        bnb(i + 1, included, included_set, current_weight)
        
    # Start B&B
    bnb(0, [], set(), Fraction(0))
    
    return best_set
