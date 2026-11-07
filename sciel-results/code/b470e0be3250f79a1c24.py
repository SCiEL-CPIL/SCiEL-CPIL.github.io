"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

def _comparable_count(sets):
    """Count unordered pairs (i,j) where one set is a strict subset of the other."""
    n = len(sets)
    # Precompute frozensets for quick subset checks
    fs = [frozenset(s) for s in sets]
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if fs[i] < fs[j] or fs[j] < fs[i]:
                count += 1
    return count

def _construct_optimal():
    """
    Construct 40 distinct subsets of {0..11} maximizing comparable pairs.
    
    Strategy: Use a down-set (ideal) structure.
    We want a family F of 40 subsets such that for any A, B in F, A⊂B or B⊂A or they are comparable.
    Actually, we want to maximize pairs where one is strict subset of other.
    
    Optimal structure for small n:
    Take all subsets of a small set S. If |S|=k, we have 2^k subsets.
    For k=5, 2^5=32. We need 40.
    For k=6, 2^6=64. We can pick 40 from these.
    
    Let S = {0,1,2,3,4,5}. Consider all 64 subsets of S.
    We want to choose 40 subsets to maximize comparable pairs.
    The comparable pairs in the full power set of S are: for each pair A⊂B, that's one comparable pair.
    Total comparable pairs in full power set = sum_{A} (2^{|S|-|A|} - 1) = 3*2^6 - 64 = 192 - 64 = 128? No.
    Actually, number of comparable pairs (A,B) with A⊂B in P(S) is:
    Sum over all A of (number of B superset of A, B != A) = Sum_{k=0}^6 C(6,k) (2^(6-k) - 1)
    = 2^6 * 3 - 2^6 = 3*64 - 64 = 128? Let's compute:
    k=0: 1*(64-1)=63
    k=1: 6*(32-1)=186
    k=2: 15*(16-1)=225
    k=3: 20*(8-1)=140
    k=4: 15*(4-1)=45
    k=5: 6*(2-1)=6
    k=6: 1*(1-1)=0
    Total = 63+186+225+140+45+6 = 665.
    
    We need to pick 40 subsets from 64 to maximize comparable pairs among them.
    This is equivalent to: pick 40 nodes from the subset lattice to maximize edges in the induced subgraph.
    
    Greedy approach: Start with all 64, remove 24 to minimize loss of comparable pairs.
    Removing a set X loses: (number of subsets of X) + (number of supersets of X) - 1 (for X itself, but we count pairs).
    Actually, removing X breaks all comparable pairs involving X.
    Number of comparable pairs involving X = (number of proper subsets of X) + (number of proper supersets of X).
    = (2^{|X|} - 1) + (2^(6-|X|) - 1) = 2^{|X|} + 2^(6-|X|) - 2.
    
    To minimize loss, remove sets with smallest (2^{|X|} + 2^(6-|X|) - 2).
    This is minimized when |X| is as close to 3 as possible (since 2^3 + 2^3 = 16 is min for k=3).
    k=3: 8+8-2=14
    k=2: 4+16-2=18
    k=4: 16+4-2=18
    k=1: 2+32-2=32
    k=5: 32+2-2=32
    k=0: 1+64-2=63
    k=6: 64+1-2=63
    
    So remove 24 sets of size 3 first? There are C(6,3)=20 sets of size 3. Remove all 20.
    Then remove 4 more from size 2 or 4 (loss 18 each).
    
    Let's try: Remove all 20 sets of size 3, and 4 sets of size 2 (or 4).
    Remaining: 
    Size 0: 1
    Size 1: 6
    Size 2: C(6,2)-4 = 15-4 = 11
    Size 3: 0
    Size 4: 15
    Size 5: 6
    Size 6: 1
    Total: 1+6+11+0+15+6+1 = 40.
    
    Let's compute comparable pairs for this family.
    We can compute this exactly.
    """
    S = frozenset(range(6))
    all_subsets = [frozenset(c) for r in range(7) for c in itertools.combinations(range(6), r)]
    
    # We want to remove 24 sets.
    # Priority: remove size 3 (20 sets), then size 2 (4 sets).
    to_remove = []
    size3 = [s for s in all_subsets if len(s) == 3]
    size2 = [s for s in all_subsets if len(s) == 2]
    
    to_remove.extend(size3)  # 20 sets
    to_remove.extend(size2[:4])  # 4 sets
    
    keep = [s for s in all_subsets if s not in set(to_remove)]
    
    # Convert to lists
    result = [list(s) for s in keep]
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # Use the analytically derived optimal structure
    sets = _construct_optimal()
    
    # Verify we have exactly 40 distinct sets
    if len(sets) != 40:
        # Fallback: simple construction
        sets = [[b for b in range(12) if (mask >> b) & 1] for mask in range(40)]
    
    # Ensure all are lists of distinct integers
    for i, s in enumerate(sets):
        sets[i] = list(s)
    
    return sets
# EVOLVE-BLOCK-END
