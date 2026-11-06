"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# Precompute all 4-element subsets of [1..16]
N = 16
K = 4
ALL_SUBSETS = list(combinations(range(1, N + 1), K))
NUM_SUBSETS = len(ALL_SUBSETS)  # C(16,4) = 1820

# Convert to bitmasks for fast intersection
def subset_to_mask(s):
    m = 0
    for x in s:
        m |= (1 << (x - 1))
    return m

SUBSET_MASKS = [subset_to_mask(s) for s in ALL_SUBSETS]

# Precompute pairwise intersection sizes
# For two 4-sets A, B: |A ∩ B| can be 0,1,2,3,4
# Since all are distinct 4-sets, |A∩B| <= 3 (if equal, they're the same set)
# Actually two distinct 4-subsets can have intersection size 0,1,2,3

# Sunflower condition: three distinct sets A,B,C such that |A∩B| = |A∩C| = |B∩C|
# This includes the empty core case where all three intersections are empty.

# Precompute intersection matrix
# inter[i][j] = |subset_i ∩ subset_j|
INTER = [[0] * NUM_SUBSETS for _ in range(NUM_SUBSETS)]
for i in range(NUM_SUBSETS):
    mi = SUBSET_MASKS[i]
    for j in range(i + 1, NUM_SUBSETS):
        mj = SUBSET_MASKS[j]
        inter_size = bin(mi & mj).count('1')
        INTER[i][j] = inter_size
        INTER[j][i] = inter_size

# Precompute conflict: for each pair (i,j), does there exist a k such that
# |i∩j| = |i∩k| = |j∩k|?
# Actually we need: for a family, no three distinct sets A,B,C have all three pairwise intersections equal.
# So for each pair (A,B), the "conflict" depends on which other set C is also in the family.
# This is a 3-uniform hypergraph problem.

# Alternative approach: greedy with bitmask
# For a given pair (i,j) in the chosen family, a new set k conflicts if:
#   INTER[i][j] == INTER[i][k] == INTER[j][k]
# i.e., INTER[i][k] == INTER[j][k] == INTER[i][j]

# Precompute for each pair (i,j), the set of k that would form a sunflower with them
# conflict_with_pair[i][j] = bitmask of all k such that INTER[i][j] == INTER[i][k] == INTER[j][k]

# This is expensive to precompute fully (1820^2 pairs * 1820 checks), but let's try
# Actually, for each pair (i,j), we need k != i, k != j such that INTER[i][k] == INTER[j][k] == INTER[i][j]

# Let's precompute for efficiency: for each k, and each intersection size s, which pairs (i,j) have
# INTER[i][k]=s and INTER[j][k]=s? That's not quite right either.

# Better: for each k, precompute the set of i where INTER[i][k] = s, for s in {0,1,2,3}
# Then for a pair (i,j) with INTER[i][j]=s, k conflicts if INTER[i][k]=s AND INTER[j][k]=s

# Let's precompute: for each k and each s in {0,1,2,3}, the bitmask of all i with INTER[i][k]=s
# Then for pair (i,j) with inter_size s, the conflicting k's are those where i is in group(k,s) and j is in group(k,s)
# This is still complex. Let's just do greedy with on-the-fly checks but optimized.

# Actually, let's precompute a conflict bitmask for each pair (i,j):
# conflict_mask[i][j] = bitmask of all k (k != i, k != j) such that INTER[i][j] == INTER[i][k] == INTER[j][k]

# This is 1820*1820/2 * 1820 operations ≈ 3 billion, too slow in Python.

# Let's use a different approach: for each k, and for each s in {0,1,2,3},
# let G[k][s] = set of i such that INTER[i][k] = s
# Then for pair (i,j) with INTER[i][j] = s, k is a conflict iff i ∈ G[k][s] and j ∈ G[k][s]

# Precompute G
G = [[set() for _ in range(4)] for _ in range(NUM_SUBSETS)]
for k in range(NUM_SUBSETS):
    for i in range(NUM_SUBSETS):
        if i != k:
            s = INTER[i][k]
            G[k][s].add(i)

# Now for greedy: when we have chosen set C, and we want to add k,
# we need to check that for every pair (i,j) in C, k is NOT a conflict.
# k is a conflict for pair (i,j) iff INTER[i][j] == INTER[i][k] == INTER[j][k]
# i.e., if s = INTER[i][j], then i ∈ G[k][s] and j ∈ G[k][s]

# So for a candidate k, the set of pairs in C that would conflict is:
# For each s in {0,1,2,3}: pairs (i,j) in C such that i ∈ G[k][s] and j ∈ G[k][s] and INTER[i][j] == s

# This is still O(|C|^2) per candidate. Let's just do it directly but with early termination.

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # seconds, safety margin
    
    best_family = []
    
    # Try multiple random orderings
    for attempt in range(200):
        if time.time() - start_time > DEADLINE:
            break
        
        # Random permutation of subsets
        indices = list(range(NUM_SUBSETS))
        random.shuffle(indices)
        
        chosen = []
        chosen_set = set()
        
        for idx in indices:
            if time.time() - start_time > DEADLINE:
                break
            
            k = idx
            # Check if k can be added
            can_add = True
            inter_k = [INTER[k][i] for i in chosen]
            
            for ai in range(len(chosen)):
                if not can_add:
                    break
                i = chosen[ai]
                s_ij = INTER[i][k]
                # For each j in chosen before i (or all j != i), check if INTER[i][j] == s_ij == INTER[j][k]
                for aj in range(ai):
                    j = chosen[aj]
                    if INTER[i][j] == s_ij == INTER[j][k]:
                        can_add = False
                        break
            
            if can_add:
                chosen.append(k)
                chosen_set.add(k)
        
        if len(chosen) > len(best_family):
            best_family = chosen[:]
            if len(best_family) >= 250:
                break
    
    # Convert to sorted lists
    result = [sorted(ALL_SUBSETS[i]) for i in best_family]
    return result

# EVOLVE-BLOCK-END
