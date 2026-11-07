"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

# Pre-compute primes up to 40
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
NUM_PRIMES = len(PRIMES)

# Pre-compute prime exponent vectors for numbers 2..40
# We only care about numbers in 2..40
# index 0 corresponds to number 2, index 1 to number 3, ..., index 38 to number 40
# Let's map number n (2..40) to index i = n - 2
MAX_VAL = 40
MIN_VAL = 2
SIZE = MAX_VAL - MIN_VAL + 1  # 39 numbers

# factor_vec[i] is a tuple of length NUM_PRIMES representing prime exponents of (MIN_VAL + i)
factor_vecs = [None] * SIZE
for i in range(SIZE):
    n = MIN_VAL + i
    exponents = [0] * NUM_PRIMES
    temp = n
    for j, p in enumerate(PRIMES):
        while temp % p == 0:
            exponents[j] += 1
            temp //= p
    factor_vecs[i] = tuple(exponents)

# Pre-compute the conflict graph
# Two numbers i and j conflict if there exist non-empty subsets S_i of {i} and S_j of {j}
# such that product(S_i) == product(S_j).
# Since we are building a set of distinct integers, let's look at the condition for a set of numbers.
# A set of numbers is valid if all subset products are distinct.
# This is equivalent to saying that the vectors of prime exponents of the numbers are linearly independent over the non-negative integers?
# No, that's not quite right.
# Consider numbers 2, 3, 6.
# Subsets:
# {} -> 1
# {2} -> 2 (1,0,0...)
# {3} -> 3 (0,1,0...)
# {6} -> 6 (1,1,0...)
# {2,3} -> 6 (1,1,0...)
# {2,6} -> 12
# {3,6} -> 18
# {2,3,6} -> 36
# Here {3} and {2,3}? No, {3} is 3, {2,3} is 6.
# But {6} is 6 and {2,3} is 6. So 6 and {2,3} conflict.
# Generally, a conflict exists if the sum of some subset of vectors equals the sum of another disjoint subset of vectors.
# This is equivalent to saying that the set of vectors is not "sum-free" in a specific sense?
# Actually, it's simpler:
# A set of numbers is valid if and only if no non-empty subset has product 1 (impossible for >1)
# AND no two distinct non-empty subsets have the same product.
# This is equivalent to: The mapping from subsets to products is injective.
# This is equivalent to: The vectors of prime exponents are such that no non-trivial linear combination with coefficients in {-1, 0, 1} sums to zero?
# Let $v_i$ be the exponent vector of number $i$.
# A collision occurs if $\sum_{i \in A} v_i = \sum_{j \in B} v_j$ for disjoint $A, B$.
# This is $\sum_{i \in A} v_i - \sum_{j \in B} v_j = 0$.
# The coefficients are $+1$ for $i \in A$, $-1$ for $j \in B$, $0$ otherwise.
# So the set is valid if and only if the only integer linear combination of the vectors $v_i$ with coefficients in $\{-1, 0, 1\}$ that sums to 0 is the trivial one (all zeros).
# This means the vectors are linearly independent over $\mathbb{Z}$? No, linear independence over $\mathbb{Z}$ means the only integer combination is trivial.
# But here coefficients are restricted to $\{-1, 0, 1\}$.
# However, if there is a dependency with larger coefficients, e.g., $2v_1 - v_2 - v_3 = 0$, does that imply a collision with coeffs in $\{-1,0,1\}$?
# Not necessarily directly, but usually these "primitive" dependencies are what matter.
# Actually, if $v_2 + v_3 = 2v_1$, then $\{2, 2\}$ (if we had two 2s, but we have distinct numbers) ... wait.
# The numbers are distinct. So we have a set of distinct vectors.
# If $v_2 + v_3 = 2v_1$, this doesn't directly give a collision between distinct subsets of the set $\{1, 2, 3\}$ unless we can form the subsets.
# Subset $\{2, 3\}$ has vector $v_2+v_3$. Subset $\{1, 1\}$ is not possible because 1 is only in the set once.
# So the condition is strictly: No two distinct subsets of the index set have the same sum of vectors.
# This is equivalent to the vectors being "sum-distinct".
#
# How to build the conflict graph for MIS?
# If we select a set of indices $S$, it is valid if no two distinct subsets of $S$ have the same sum.
# This property is NOT pairwise. It is a global property of the set $S$.
# Example: $\{2, 3, 6\}$.
# Pairwise:
# 2 vs 3: no conflict.
# 2 vs 6: no conflict.
# 3 vs 6: no conflict.
# But $\{2, 3, 6\}$ is invalid because $\{6\}$ and $\{2, 3\}$ have same product.
# So we cannot simply use a pairwise conflict graph for MIS. The "conflict" is a hyperedge of size 3 (or more).
#
# Therefore, the problem is: Find the largest subset of $\{2..40\}$ such that the prime exponent vectors are "sum-distinct".
# This is not a standard MIS on a simple graph.
#
# However, we can still use a branch-and-bound search, but the validity check is expensive if done naively.
# But $N=39$ is small enough that we can use a different strategy:
# Iterative deepening or branch-and-bound with pruning based on the "sum-distinct" property.
#
# Key Insight:
# If we have a set of vectors, we can check if they are sum-distinct by generating all $2^k$ subset sums and checking for duplicates.
# For $k=18$, $2^{18} \approx 262,144$. This is fast in Python if optimized (using tuples of small ints or packed integers).
#
# Strategy:
# 1. Pre-compute all possible "conflicts"? No, too many.
# 2. Use a recursive backtracking search.
#    - Maintain a list of selected vectors.
#    - At each step, try adding a new vector.
#    - Check if the new set is valid.
#    - To speed up checking, we can maintain the set of all current subset sums.
#    - When adding a new vector $v$, the new subset sums are $\{s, s+v \mid s \in \text{current\_sums}\}$.
#    - We need to check if any $s+v$ is already in $\text{current\_sums}$ (collision with existing subset)
#      OR if any $s_1+v = s_2+v$ (impossible since $s_1=s_2$)
#      OR if the new sums collide with each other? No, $s_1+v = s_2+v \implies s_1=s_2$.
#      So we just need to check if $\{s+v \mid s \in \text{current\_sums}\}$ intersects with $\text{current\_sums}$.
#      If it does, adding $v$ creates a collision.
#      Specifically, if $s_1 + v = s_2$ for some $s_1, s_2 \in \text{current\_sums}$, then the subset corresponding to $s_2$ and the subset corresponding to $s_1$ plus the new element have the same product.
#
# This allows for incremental validity checking!
#
# Algorithm:
# 1. Start with empty set, sums = {0}.
# 2. Branch on adding numbers from 2..40.
# 3. Use Branch and Bound:
#    - Upper bound on size: current_size + (remaining_candidates).
#    - Prune if upper bound <= best_found.
# 4. To make it faster, we can order candidates.
#    - Heuristic: Numbers with "rare" prime factors or large values might be less likely to cause conflicts?
#    - Or just iterate 2..40.
#
# Optimization:
# Represent subset sums as tuples of prime exponents.
# Max exponent for 2 in product of 18 numbers <= 40 is roughly $18 \times 5 = 90$ (if all were 32).
# Actually max power of 2 in 40 is $32=2^5$. So max exponent is $18 \times 5 = 90$.
# We can pack the exponent vector into a single integer for faster hashing/comparison.
# Bit allocation per prime:
# 2: max 5 * 18 = 90 (7 bits)
# 3: max 3 * 18 = 54 (6 bits)
# 5: max 2 * 18 = 36 (6 bits)
# 7: max 1 * 18 = 18 (5 bits)
# 11..37: max 1 * 18 = 18 (5 bits each)
# Total bits: 7 + 6 + 6 + 5 + 8*5 = 24 + 40 = 64 bits.
# This fits in a standard 64-bit integer (or Python int which is arbitrary precision but small is faster).
#
# Let's define the packing:
# prime 2: bits 0-6 (7 bits)
# prime 3: bits 7-12 (6 bits)
# prime 5: bits 13-18 (6 bits)
# prime 7: bits 19-23 (5 bits)
# prime 11: bits 24-28 (5 bits)
# prime 13: bits 29-33 (5 bits)
# prime 17: bits 34-38 (5 bits)
# prime 19: bits 39-43 (5 bits)
# prime 23: bits 44-48 (5 bits)
# prime 29: bits 49-53 (5 bits)
# prime 31: bits 54-58 (5 bits)
# prime 37: bits 59-63 (5 bits)
#
# We can pre-compute the packed representation of each number's factor vector.

def pack_exponents(exps):
    val = 0
    shifts = [0, 7, 13, 19, 24, 29, 34, 39, 44, 49, 54, 59]
    masks = [(1 << 7) - 1, (1 << 6) - 1, (1 << 6) - 1, (1 << 5) - 1, (1 << 5) - 1, (1 << 5) - 1,
             (1 << 5) - 1, (1 << 5) - 1, (1 << 5) - 1, (1 << 5) - 1, (1 << 5) - 1, (1 << 5) - 1]
    for i in range(NUM_PRIMES):
        val |= (exps[i] & masks[i]) << shifts[i]
    return val

packed_factors = [pack_exponents(fv) for fv in factor_vecs]

# Pre-compute the "addition" of packed factors?
# Since we are summing exponents, and we packed them with enough bits,
# simple integer addition works IF there is no carry between fields.
# Max sum for a field:
# Field 2: max 90. $90 < 2^7=128$. OK.
# Field 3: max 54. $54 < 2^6=64$. OK.
# Field 5: max 36. $36 < 2^6=64$. OK.
# Field 7+: max 18. $18 < 2^5=32$. OK.
# So simple integer addition of packed factors corresponds exactly to vector addition.
# This is a huge speedup.

# Now, the search:
# We want to find the max size subset of indices 0..38 such that all subset sums of their packed factors are distinct.
#
# We will use a recursive DFS with backtracking.
# State:
# - current_packed_factors: list of packed factors in the current set.
# - current_sums: set of all subset sums (packed integers).
# - start_index: index from which to consider adding new numbers.
# - best_size: global best found.
# - best_set: global best set of indices.
#
# To improve efficiency, we can use a "conflict check" before recursing:
# When adding a new packed factor $p$, the new sums are $\{s, s+p \mid s \in \text{current\_sums}\}$.
# We need to ensure that no $s+p$ is in $\text{current\_sums}$.
# This is equivalent to: $(\text{current\_sums} + p) \cap \text{current\_sums} = \emptyset$.
# Where $A+p = \{a+p \mid a \in A\}$.
#
# Implementation detail:
# Instead of storing the list of factors, we just need the set of sums.
# But we also need to know which factors are selected to reconstruct the set.
#
# Branch and Bound:
# If current_size + (38 - start_index + 1) <= best_size, prune.
#
# Ordering:
# Try indices in increasing order? Or random?
# Deterministic increasing order is reproducible.
#
# Let's implement the solver.

def solve_mis_sum_distinct():
    best_size = 0
    best_set = []
    
    # Pre-compute packed factors for all 39 numbers
    # indices 0..38 correspond to numbers 2..40
    
    # We use a global variable for best to avoid passing around
    # Or just closure variables.
    
    def dfs(start_idx, current_sums, current_indices, current_size):
        nonlocal best_size, best_set
        
        # Pruning
        remaining = 38 - start_idx + 1
        if current_size + remaining <= best_size:
            return
            
        if current_size > best_size:
            best_size = current_size
            best_set = current_indices[:]
            
        # If no more candidates, return
        if start_idx > 38:
            return
            
        # Try adding each candidate from start_idx to 38
        for i in range(start_idx, 39):
            p = packed_factors[i]
            
            # Check if adding p creates a collision
            # New sums will be current_sums U {s+p for s in current_sums}
            # Collision if any s+p is in current_sums
            
            # Optimization:
            # If p == 0, always conflict (but p is never 0 for numbers >= 2)
            
            # Check intersection
            # Iterate through current_sums and check if s+p is in current_sums
            # This is O(N) where N is size of current_sums.
            # Max size of current_sums is 2^18 ~ 260k.
            # This check is expensive if done for every i.
            # But we only do it for valid candidates.
            
            # Can we optimize the check?
            # If we maintain current_sums as a set, lookup is O(1).
            # So the check is O(|current_sums|).
            
            # Is there a faster way?
            # Maybe pre-compute "forbidden" sums?
            # No, it depends on the specific set.
            
            # Let's just do the check.
            # To speed it up, we can break early if a collision is found.
            
            has_collision = False
            for s in current_sums:
                if (s + p) in current_sums:
                    has_collision = True
                    break
                    
            if has_collision:
                continue
                
            # If no collision, recurse
            new_sums = set(current_sums)
            for s in current_sums:
                new_sums.add(s + p)
                
            new_indices = current_indices + [i]
            
            dfs(i + 1, new_sums, new_indices, current_size + 1)
            
            # Backtrack
            # No explicit backtracking needed as we create new sets/objects
            
    # Initial state
    initial_sums = {0}
    dfs(0, initial_sums, [], 0)
    
    return [x + 2 for x in best_set]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # The exact solver should be fast enough for N=39.
    # Worst case is exponential, but pruning is strong.
    # If it takes too long, we might need a heuristic fallback,
    # but let's try the exact solver first.
    
    try:
        result = solve_mis_sum_distinct()
        # Verify result
        if _verify_distinct_subset_products(result):
            return [int(x) for x in result]
    except Exception:
        pass
    
    # Fallback to a simple greedy if exact solver fails or times out (unlikely)
    # But we should ensure we return something valid.
    # A single number is always valid.
    return [2]

def _verify_distinct_subset_products(nums):
    if not nums:
        return True
    # Use the same packing logic for verification
    # But for safety, let's use a simple product check for small sets
    # Or just re-use the packed sums logic.
    
    packed_factors_local = []
    for x in nums:
        # Find index
        if x < 2 or x > 40:
            return False
        idx = x - 2
        packed_factors_local.append(packed_factors[idx])
        
    sums = {0}
    for p in packed_factors_local:
        new_sums = set(sums)
        for s in sums:
            new_sums.add(s + p)
        # Check for duplicates in new_sums
        if len(new_sums) != 2 * len(sums):
            return False
        sums = new_sums
        
    return True

# EVOLVE-BLOCK-END
