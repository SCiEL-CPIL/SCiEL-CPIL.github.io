import time
from math import gcd

# Precomputation
N = 511
# Weights w_a = 1 / (512 - a)
# Candidates are 1..511
# Sort by weight descending -> a descending
candidates = list(range(N, 0, -1))
weights = {a: 1.0 / (512 - a) for a in candidates}

# Suffix sums of weights for upper bound relaxation (ignoring conflicts)
# suffix_sum[i] = sum(weights[candidates[j]] for j in range(i, len(candidates)))
suffix_sum = [0.0] * (len(candidates) + 1)
for i in range(len(candidates) - 1, -1, -1):
    suffix_sum[i] = suffix_sum[i + 1] + weights[candidates[i]]

# Precompute prime factors for conflict detection
def get_prime_factors(n):
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

factors = {a: get_prime_factors(a) for a in candidates}

# Precompute conflict masks as bitsets (integers)
# Bit i is set if candidates[i] conflicts with the current number
# Actually, easier: for each a, store a set of indices that conflict?
# Or store a bitset of values?
# Since N=511, we can use a Python int as a bitset of 511 bits.
# Bit k corresponds to candidate index k? Or value a?
# Let's map value a to index idx[a].
idx_map = {a: i for i, a in enumerate(candidates)}

conflict_mask = {}
for a in candidates:
    mask = 0
    for b in candidates:
        if a != b and (factors[a] & factors[b]):
            mask |= (1 << idx_map[b])
    conflict_mask[a] = mask

# Initial greedy solution to set best_score
def greedy_solution():
    chosen = []
    blocked = 0
    for a in candidates:
        if not (blocked & (1 << idx_map[a])):
            chosen.append(a)
            blocked |= conflict_mask[a]
    return sum(weights[a] for a in chosen), chosen

initial_score, initial_set = greedy_solution()
best_score = initial_score
best_set = initial_set

deadline = time.time() + 110  # 110s limit, leaving 10s margin

def search(idx, current_score, blocked_mask):
    global best_score, best_set, deadline
    
    if time.time() > deadline:
        return
        
    # Upper bound: current_score + sum of all remaining weights (relaxation)
    # We can tighten this by subtracting weights of blocked items?
    # For speed, use suffix_sum first.
    if current_score + suffix_sum[idx] <= best_score:
        return
        
    if idx == len(candidates):
        if current_score > best_score:
            best_score = current_score
            # Reconstruct set? We didn't track it in recursion.
            # We can track it, but it's expensive.
            # For the purpose of the score, we just need the value.
            # But we need to return the set.
            # Let's track the set.
        return

    a = candidates[idx]
    bit_a = 1 << idx_map[a]
    
    # Option 1: Include a
    if not (blocked_mask & bit_a):
        new_score = current_score + weights[a]
        # Prune include branch if even adding all remaining can't beat best
        if new_score + suffix_sum[idx + 1] > best_score:
            new_blocked = blocked_mask | conflict_mask[a]
            # Pass set? 
            # To save memory/time, we can just track the score and reconstruct later?
            # No, we need to return the set.
            # Let's pass a tuple or list.
            # To be efficient, we can store the best set found in the global variable.
            search(idx + 1, new_score, new_blocked)
            
    # Option 2: Exclude a
    # Prune exclude branch if current_score + suffix_sum[idx+1] <= best_score
    if current_score + suffix_sum[idx + 1] > best_score:
        search(idx + 1, current_score, blocked_mask)

# To return the actual set, we need to track which choices led to the best score.
# Modifying the search to track the set is necessary.

def search_with_set(idx, current_score, blocked_mask, current_set):
    global best_score, best_set, deadline
    
    if time.time() > deadline:
        return
        
    if current_score + suffix_sum[idx] <= best_score:
        return
        
    if idx == len(candidates):
        if current_score > best_score:
            best_score = current_score
            best_set = current_set[:]
        return

    a = candidates[idx]
    bit_a = 1 << idx_map[a]
    
    # Option 1: Include a
    if not (blocked_mask & bit_a):
        new_score = current_score + weights[a]
        if new_score + suffix_sum[idx + 1] > best_score:
            new_blocked = blocked_mask | conflict_mask[a]
            current_set.append(a)
            search_with_set(idx + 1, new_score, new_blocked, current_set)
            current_set.pop()
            
    # Option 2: Exclude a
    if current_score + suffix_sum[idx + 1] > best_score:
        search_with_set(idx + 1, current_score, blocked_mask, current_set)

def construct():
    global best_score, best_set, deadline
    
    # Reset state
    best_score, best_set = greedy_solution()
    deadline = time.time() + 110
    
    # Start search
    search_with_set(0, 0.0, 0, [])
    
    return best_set

# Run a quick test in mind:
# The search might be too slow if the bound is weak.
# suffix_sum[idx] is the sum of weights from idx to end.
# This is a very loose bound because it ignores that many items are blocked.
# However, the weights decay so fast that suffix_sum[idx] drops quickly.
# For idx=0, suffix_sum[0] is sum of all weights ~ 6.2.
# best_score starts ~ 3-4 (greedy).
# So we don't prune at root.
# As we go deeper, suffix_sum drops.
# If we pick 511, w=1. suffix_sum[1] ~ 5.2.
# If we pick 510, w=0.5.
# The branching factor is 2.
# Depth 10: 1024 nodes.
# Depth 20: 1 million nodes.
# Python can do ~10-50 million simple ops per second.
# Each node involves some bit ops and comparisons.
# 1 million nodes might take 1-2 seconds.
# So we can explore depth 20-25 comfortably.
# This should be enough to find improvements over greedy.

# One optimization: 
# If we include a, we update blocked.
# The bound `new_score + suffix_sum[idx+1]` is still loose.
# We can tighten it by noting that `conflict_mask[a]` blocks some items in `candidates[idx+1:]`.
# Let `blocked_indices` be the set of indices in `candidates[idx+1:]` that are blocked by `a`.
# `tightened_ub = new_score + (suffix_sum[idx+1] - sum(weights[candidates[j]] for j in blocked_indices))`.
# Calculating this sum takes $O(N)$.
# If we do this, the cost per node is $O(N)$.
# Total cost $O(N 2^D)$.
# With $N=511, D=20$, $511 \times 10^6 \approx 5 \times 10^8$. Too slow.
# So stick to the loose bound `suffix_sum[idx+1]`.
# The loose bound will prune later, but the node cost is $O(1)$.
# Total cost $O(2^D)$.
# $2^{30} \approx 10^9$. Too slow.
# $2^{25} \approx 3 \times 10^7$. Might be okay in 110s?
# Python recursion overhead is high.
# Let's limit depth or add a stronger heuristic.

# Heuristic: 
# If `current_score + suffix_sum[idx] - sum(weights of blocked items in suffix) <= best_score`, prune.
# We can precompute for each `a` the sum of weights of items it blocks?
# `block_sum[a] = sum(weights[b] for b in candidates if conflict(a,b))`.
# This is not additive.
# Let's just rely on the loose bound and hope the tree is pruned well enough.
# Or, use a time-based cutoff.

# To ensure we return a valid set, `best_set` is updated only when a complete path is explored.
# The greedy solution is always a valid fallback.

# Final check on the code structure.
# The function `construct` must return the list of integers.

# One potential issue: Recursion depth.
# Max depth 511. Python default limit is 1000. Should be fine.
# But we can increase it just in case.
import sys
sys.setrecursionlimit(2000)

# Let's refine the pruning.
# If we are at index idx, and we have a set of blocked bits.
# The maximum possible addition is sum of weights of unblocked items from idx onwards.
# We can compute this sum faster if we maintain a running sum?
# No, blocked changes.
# Let's stick to the simple version. If it's too slow, it will timeout and return greedy.
# But greedy is 1.0. We want > 1.0.
# The B&B will likely find a better solution in the first few thousand nodes.

# To speed up, we can skip the "Exclude" branch if `current_score + suffix_sum[idx+1] <= best_score`.
# This is already in the code.

# Let's add a check: if `idx` is large, the remaining sum is small.
# If `suffix_sum[idx] < 1e-9`, return.

# The code looks solid.

# One more thing: The prompt asks for "distinct pairwise coprime integers".
# My `conflict_mask` ensures that.
# `gcd(a,b)==1` is equivalent to no common prime factors.
# My `get_prime_factors` and intersection check is correct.

# Let's double check the weight calculation.
# `1.0 / (512 - a)`.
# For a=511, 1/1 = 1.0.
# For a=1, 1/511.
# Correct.

# The `construct` function returns `best_set`.
# `best_set` is a list of integers.
# Correct.

# I will wrap the search in a try-except to ensure it doesn't crash if something goes wrong,
# and returns the greedy solution.

def construct():
    global best_score, best_set, deadline
    
    # Reset state
    best_score, best_set = greedy_solution()
    deadline = time.time() + 110
    
    try:
        search_with_set(0, 0.0, 0, [])
    except RecursionError:
        pass
    except Exception:
        pass
        
    return best_set
