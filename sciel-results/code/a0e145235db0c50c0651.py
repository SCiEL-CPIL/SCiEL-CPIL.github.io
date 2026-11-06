import time
import random
import numpy as np
from itertools import combinations

N = 16
K = 4
MAX_SETS = 250

# Precompute all 4-subsets of [16]
ALL_SETS = []
SET_TO_IDX = {}

def _init():
    global ALL_SETS, SET_TO_IDX
    if ALL_SETS:
        return
    for combo in combinations(range(N), K):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        ALL_SETS.append(mask)
        SET_TO_IDX[mask] = len(ALL_SETS) - 1

_init()
N_SETS = len(ALL_SETS)

# Precompute conflict structure
# For each pair (i, j), find all k such that {i, j, k} forms a sunflower
# Sunflower condition: A ∩ B = A ∩ C = B ∩ C
# For a given pair (A, B), let I = A ∩ B, S = A ∪ B
# C must satisfy: C ∩ A = I and C ∩ B = I
# This means: C contains I, C contains no elements from S \ I, and C \ I ⊆ complement of S
# So C = I ∪ P where P is a (K - |I|)-subset of [N] \ S
# Additionally, C must be distinct from A and B

# We'll build a conflict matrix as a list of bitmasks
# conflict[i] is a bitmask of all j such that i and j conflict with some k
# Actually, we need: for each pair (i, j), which k's are blocked?
# Let's store: for each i, a list of (j, blocked_mask) where blocked_mask has bits for k's

# More efficient: precompute for each pair (i, j) the set of k's that form a sunflower
# Store as a list of frozensets or as bitmasks

# Let's use a 2D list where conflict_mask[i][j] is a bitmask of k indices
# This is N_SETS * N_SETS bitmasks, each N_SETS bits wide
# N_SETS = C(16,4) = 1820
# 1820 * 1820 * (1820/8) bytes ≈ 1820*1820*228 ≈ 753 MB - too much!

# Alternative: for each i, store a list of (j, k_mask) where k_mask is a bitmask
# Only non-empty entries

# Even better: for the greedy algorithm, when we add i, we need to know which j's are now blocked
# We can precompute: for each pair (i, j), the set of k's that complete a sunflower with i and j
# Store this in a dictionary or sparse structure

# Let me think about the size. For a given pair (i, j) with intersection size t:
# Number of k's = C(N - |S|, K - t) where S = A ∪ B, |S| = 2K - t = 8 - t
# So N - |S| = 16 - (8 - t) = 8 + t
# Number of k's = C(8+t, K-t) = C(8+t, 4-t)

# For t=0: C(8,4) = 70
# For t=1: C(9,3) = 84
# For t=2: C(10,2) = 45
# For t=3: C(11,1) = 11
# For t=4: C(12,0) = 1 (but this would be C=A=B, not distinct, so 0)

# Total number of (i,j,k) sunflower triples is bounded.

# Let's precompute for each i, a list of pairs (j, k_mask) where k_mask has bits for all k
# that form a sunflower with i and j.

# Actually, for the greedy, when I select i, for each already selected j, I need to block all k
# that form a sunflower with i and j. So I need to know: for pair (i, j), what k's are blocked?

# Let me store: sunflower_block[i][j] = bitmask of k's
# But that's too much memory. Instead, store only for i < j, and use symmetry.

# Better: store as a dictionary: (i, j) -> bitmask of k's, only for non-empty cases.

# Or: for each i, store a dict j -> bitmask of k's.

# Let me estimate: number of pairs (i,j) where sunflower is possible.
# For each i, j can be any of the other 1819 sets. But most pairs won't have any k.
# Actually, every pair (i, j) with |A∩B| < 4 will have some k's (unless C(8+t, 4-t) = 0).
# t=0: C(8,4)=70 > 0
# t=1: C(9,3)=84 > 0
# t=2: C(10,2)=45 > 0
# t=3: C(11,1)=11 > 0
# t=4: C(12,0)=1, but C would equal A and B, so excluded.
# So every pair with intersection < 4 has at least one k.
# Number of pairs with intersection < 4: almost all pairs.
# Total pairs: C(1820, 2) ≈ 1.65M
# This is too many to store as a dict with bitmasks.

# Alternative approach: don't precompute all conflicts. Instead, when adding i to the selected set,
# for each already selected j, compute the blocked k's on the fly.
# But that might be slow.

# Let me try a different approach: use a bipartite graph or something.

# Actually, let me reconsider the memory. 
# For each i, I can store a list of (j, k_list) where k_list is a list of k indices.
# Total number of (i, j, k) triples: let's estimate.
# For each triple of 4-sets, how many form a sunflower?
# This is hard to estimate directly.

# Let me just try the on-the-fly approach with memoization.

# For the greedy:
# selected = set of indices
# When adding i:
#   for each j in selected:
#     compute all k such that {i, j, k} is a sunflower
#     add those k to blocked
#
# To compute k's for pair (i, j):
#   I = ALL_SETS[i] & ALL_SETS[j]
#   S = ALL_SETS[i] | ALL_SETS[j]
#   t = popcount(I)
#   if t == 4: continue  # A and B are identical, skip
#   U = FULL_MASK ^ S  # complement of S
#   p_size = K - t
#   # k = I ∪ P where P is a p_size-subset of U
#   # Also k must not equal i or j
#   for P in combinations(U_bits, p_size):
#       k_mask = I | P_mask
#       k_idx = SET_TO_IDX[k_mask]
#       if k_idx != i and k_idx != j:
#         block k_idx

# This is feasible if we do it efficiently.

# Let me precompute for each possible intersection pattern.

# Actually, let me just precompute the conflicts using a more memory-efficient method.
# I'll store for each i, a numpy array of length N_SETS, where conflict_block[i][j] is a boolean
# indicating whether j is blocked by i... no, that's not right.

# Let me use a different representation.
# For the greedy, I maintain:
#   selected: set of indices
#   blocked: set of indices
#
# When I add i to selected:
#   for each j in selected:
#     for each k that forms sunflower with i and j:
#       blocked.add(k)
#
# I need to quickly find k's for a given (i, j).
# Let me precompute a dictionary: pair_key -> list of k indices
# pair_key = (min(i,j), max(i,j))

# Number of distinct pairs with sunflowers: up to 1.65M
# Each list has at most 84 elements (for t=1)
# Total storage: 1.65M * 84 * 8 bytes ≈ 1.1 GB - too much!

# I need a smarter approach.

# Alternative: use the structure of the problem.
# A sunflower with core I (size t) has petals of size K-t = 4-t.
# For t=0: core empty, petals are disjoint 4-sets. Three pairwise disjoint 4-sets from [16].
# For t=1: core is 1 element, petals are 3-element sets from the remaining 15, disjoint from core.
# etc.

# Let me group the 4-sets by their "type" relative to each other.

# Actually, let me just use a simpler approach: 
# Precompute for each pair (i, j) the blocked k's, but store them compactly.
# Since the maximum number of k's is 84, I can store them as a single 64-bit integer if I use
# a mapping from index to bit position... but 1820 > 64.

# Let me use Python's built-in int as a bitmask. 1820 bits is fine.
# conflict_mask[i][j] = int with bits for k's.
# But storing 1820 * 1820 such ints: each int is ~1820/30 ≈ 61 machine words ≈ 244 bytes.
# Total: 1820 * 1820 * 244 ≈ 805 MB. Still too much.

# I'll use a sparse representation: for each i, a dict j -> bitmask.
# Only store entries where the bitmask is non-zero.
# Most pairs will have non-zero bitmasks (as shown above), so this is still ~1.65M entries.
# Each entry: key (int, ~28 bytes) + value (int, ~244 bytes) ≈ 272 bytes.
# Total: 1.65M * 272 ≈ 450 MB. Still a lot, but might work.

# Let me try a different strategy: don't precompute all conflicts.
# Instead, during the greedy, when adding i, for each j in selected, compute the k's on the fly.
# The computation for a single pair is O(C(8+t, 4-t)) which is at most 84.
# With |selected| up to 250, and 1820 candidates to check, the total work is:
#   1820 * 250 * 84 ≈ 38M operations per full greedy pass.
# That's feasible.

# Let me implement this approach.

def _compute_blocked(i, j):
    """Compute the set of k indices such that {ALL_SETS[i], ALL_SETS[j], ALL_SETS[k]} is a sunflower."""
    A = ALL_SETS[i]
    B = ALL_SETS[j]
    I = A & B
    S = A | B
    t = bin(I).count('1')
    if t >= K:
        return []
    U = FULL_MASK ^ S
    p_size = K - t
    if p_size < 0:
        return []
    # Get bits of U
    U_bits = []
    temp = U
    while temp:
        lsb = temp & -temp
        U_bits.append(lsb)
        temp ^= lsb
    if len(U_bits) < p_size:
        return []
    result = []
    for P_combo in combinations(U_bits, p_size):
        P_mask = 0
        for b in P_combo:
            P_mask |= b
        C_mask = I | P_mask
        k_idx = SET_TO_IDX.get(C_mask)
        if k_idx is not None and k_idx != i and k_idx != j:
            result.append(k_idx)
    return result

# Precompute for speed: cache the U_bits for each possible U
# There are only 2^16 possible U values, but we only need those that arise from pairs of 4-sets.
# Let me cache it.

_U_BITS_CACHE = {}

def _get_U_bits(U):
    if U in _U_BITS_CACHE:
        return _U_BITS_CACHE[U]
    bits = []
    temp = U
    while temp:
        lsb = temp & -temp
        bits.append(lsb)
        temp ^= lsb
    _U_BITS_CACHE[U] = bits
    return bits

FULL_MASK = (1 << N) - 1

def _compute_blocked_fast(i, j):
    A = ALL_SETS[i]
    B = ALL_SETS[j]
    I = A & B
    S = A | B
    t = bin(I).count('1')
    if t >= K:
        return []
    U = FULL_MASK ^ S
    p_size = K - t
    if p_size < 0:
        return []
    U_bits = _get_U_bits(U)
    if len(U_bits) < p_size:
        return []
    result = []
    for P_combo in combinations(U_bits, p_size):
        P_mask = 0
        for b in P_combo:
            P_mask |= b
        C_mask = I | P_mask
        k_idx = SET_TO_IDX.get(C_mask)
        if k_idx is not None and k_idx != i and k_idx != j:
            result.append(k_idx)
    return result

# For even more speed, precompute all (U, p_size) combinations results
# Actually, let me precompute for each (I, U, p_size) the list of C_masks.
# But there are many distinct (I, U) pairs.

# Let me just use the fast version and hope it's fast enough.

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    best_size = 0
    best_indices = []
    
    indices = list(range(N_SETS))
    
    while time.time() < deadline:
        random.shuffle(indices)
        
        selected = set()
        blocked = set()
        selected_list = []
        
        for u in indices:
            if u in selected or u in blocked:
                continue
            
            # Check if adding u creates any sunflower with pairs in selected
            # For each j in selected, check if there exists k in selected such that {u, j, k} is sunflower
            # But we already maintain blocked, so if u is not blocked, it should be safe?
            # Wait, the blocked set is computed when we add elements. Let me re-examine.
            
            # Actually, my approach was: when adding i, for each j in selected, block all k
            # that form sunflower with i and j. So blocked contains all k that would form a sunflower
            # with any pair in selected.
            
            # So if u is not in blocked, it's safe to add.
            
            selected.add(u)
            selected_list.append(u)
            
            # Update blocked
            # For each j in selected (excluding u), compute k's that form sunflower with u and j
            for j in selected_list:
                if j == u:
                    continue
                for k in _compute_blocked_fast(u, j):
                    blocked.add(k)
            
            # Remove u from blocked if it was there (shouldn't be, but just in case)
            blocked.discard(u)
        
        current_size = len(selected_list)
        if current_size > best_size:
            best_size = current_size
            best_indices = selected_list[:]
            if best_size >= MAX_SETS:
                break
        
        if time.time() > deadline - 5.0:
            break
    
    result = []
    for idx in best_indices:
        mask = ALL_SETS[idx]
        elems = []
        for i in range(N):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
        
    return result

# Test with a small example
if __name__ == "__main__":
    result = construct()
    print(f"Found {len(result)} sets")
    for s in result[:5]:
        print(s)
