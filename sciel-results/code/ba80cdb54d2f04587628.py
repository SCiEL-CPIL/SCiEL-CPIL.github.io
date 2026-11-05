"""
Strategy: Model as maximum independent set in a conflict graph where an edge
exists between two 4-subsets if they could be part of a sunflower triple with
some third set. Actually, we need: no three distinct sets A,B,C such that
A∩B = A∩C = B∩C (the common core).

The condition for three sets A,B,C to form a "sunflower" (in the sense of the
task) is that all three pairwise intersections are equal. Let's denote:
  AB = A∩B, AC = A∩C, BC = B∩C
We need AB = AC = BC.

Note: If AB = AC = BC = S, then S ⊆ A, S ⊆ B, S ⊆ C. Also, for any element
in A∪B∪C, its membership pattern across A,B,C must be consistent.

Actually, let's think about this more carefully. If A∩B = A∩C = B∩C = S,
then:
- Every element in S is in all three sets.
- No element outside S is in two or more of the sets.

So the three sets have a common core S, and outside of S, each set has
elements that appear in exactly one of the three sets.

For k=4, |A|=|B|=|C|=4. Let |S|=s. Then each set has 4-s elements outside S,
and these are all disjoint. So the total number of distinct elements in
A∪B∪C is s + 3*(4-s) = 12 - 2s.

Since we're working with [1,16], this is always possible as long as 12-2s ≤ 16,
which is always true for s ≤ 4.

The key insight: two sets A and B "conflict" if there exists a third set C
such that A∩B = A∩C = B∩C. But checking this for all triples is expensive.

Better approach: Precompute for each pair (A,B) what the common core would be.
If A∩B = S, then any C that forms a sunflower with A and B must satisfy
A∩C = S and B∩C = S. This means C contains S, and C has no other elements
from A\S or B\S.

So C must be a 4-set containing S, with the remaining 4-|S| elements chosen
from [1,16] \ (A ∪ B).

Let's build a conflict graph: edge between A and B if there exists C ≠ A, B
that forms a sunflower triple with A and B.

Actually, a simpler and more direct approach: just check all triples. But
there are C(1820,3) ≈ 5 billion triples, which is too many.

Let me think differently. For each pair (A,B), compute S = A∩B. Then the
"forbidden" third sets C are those 4-sets containing S with the other elements
from [1,16]\(A∪B). The number of such C is C(16 - |A∪B|, 4-|S|).

|A∪B| = |A| + |B| - |S| = 8 - |S|.
So 16 - |A∪B| = 8 + |S|.
Number of forbidden C = C(8+|S|, 4-|S|).

For |S|=0: C(8,4)=70
For |S|=1: C(9,3)=84
For |S|=2: C(10,2)=45
For |S|=3: C(11,1)=11
For |S|=4: C(12,0)=1 (but then C=A=B, so no distinct C)

So for each pair, at most 84 forbidden third sets. Total pairs: C(1820,2)≈1.65M.
Total forbidden (A,B,C) incidences: ~1.65M * 84 ≈ 139M. That's a lot but maybe
manageable if we're smart.

Actually, let me just do a greedy approach with efficient conflict checking.

For a greedy construction, when considering adding set X to the family F,
I need to check that no two existing sets A,B in F form a sunflower with X.
That is, for all pairs (A,B) in F, check if A∩B = A∩X = B∩X.

This is O(|F|^2) per addition, which is too slow for large families.

Better: For each existing set A, precompute which new sets X would conflict
with A and some other set B. 

Alternative: For each pair (A,B) in F, compute the set of X that would form
a sunflower. When adding X, check if X is in any of these forbidden sets.

Let me use a different approach: maintain for each possible "core" S (subset of
[1,16]), the count of sets in F that contain S. When adding a new set X, for
each subset S of X (there are 16 subsets), check if there are already two sets
in F that contain S and have no other elements in common with X outside S... 

This is getting complicated. Let me just do a straightforward greedy with
bitmask optimization and a time limit.

Actually, let me think about the structure. The maximum family size is bounded.
For k=4, n=16, the sunflower-free family size... Let me just implement a
greedy algorithm that's fast enough.

I'll use bitmasks for the 16 elements. Each 4-set is a 16-bit integer.
Intersection is bitwise AND.

For the greedy: try all 1820 sets in some order, add if it doesn't create
a sunflower triple.

To check if adding X creates a sunflower: for each pair (A,B) in current family,
check if (A&B) == (A&X) == (B&X).

This is O(m^2) per insertion where m is current family size. With m up to ~250,
that's 62,500 checks per insertion, times 1820 insertions = ~114M checks.
Each check is a few bitwise ops. Should be fine in Python with optimization.

Let me implement this with a time guard.
"""

import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin
    
    # Generate all 4-subsets of [1,16] as bitmasks
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        all_sets.append(mask)
    
    n_total = len(all_sets)  # 1820
    
    # Greedy: try to add sets in a good order
    # Sort by... maybe just try all in order, or use a heuristic
    # Let's try multiple orders and keep the best
    
    best_family = []
    
    # Order 1: natural order
    for trial in range(3):
        if time.time() > deadline:
            break
        
        if trial == 0:
            order = list(range(n_total))
        elif trial == 1:
            # Reverse order
            order = list(range(n_total - 1, -1, -1))
        else:
            # Random-ish shuffle using a simple LCG
            order = list(range(n_total))
            seed = 12345
            for i in range(n_total - 1, 0, -1):
                seed = (seed * 1103515245 + 12345) % (2**31)
                j = seed % (i + 1)
                order[i], order[j] = order[j], order[i]
        
        family = []
        family_set = set()
        
        for idx in order:
            if time.time() > deadline:
                break
            
            x = all_sets[idx]
            x_set = x  # bitmask
            
            # Check if x conflicts with any pair in family
            conflict = False
            m = len(family)
            for i in range(m):
                a = family[i]
                ab = a & x_set
                if conflict:
                    break
                for j in range(i + 1, m):
                    b = family[j]
                    # Check if a, b, x form a sunflower
                    # Need a&b == a&x == b&x
                    if (a & b) == ab and (b & x_set) == ab:
                        conflict = True
                        break
            
            if not conflict:
                family.append(x_set)
                family_set.add(x_set)
        
        if len(family) > len(best_family):
            best_family = family
    
    # Convert bitmasks back to sorted lists
    result = []
    for mask in best_family:
        elements = []
        for i in range(16):
            if mask & (1 << i):
                elements.append(i + 1)
        result.append(sorted(elements))
    
    return result
