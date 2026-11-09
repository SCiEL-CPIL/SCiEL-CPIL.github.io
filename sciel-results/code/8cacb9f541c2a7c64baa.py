"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 60s limit, keep 5s margin
    
    n_max = 192
    max_size = 48
    
    # Pre-compute conflicts for all subsets? No, too large.
    # We need to solve MIS on a hypergraph where vertices are 1..192.
    # Edge is a set of 4 vertices {a,b,c,d} if a divides b+c+d (and permutations).
    # We want max independent set.
    
    # To make this tractable, we restrict the candidate pool to larger numbers.
    # Why? Smaller numbers divide more sums.
    # If we pick numbers > 96, the sum of any 3 is > 288.
    # If a > 96, a can only divide sum if sum is 2a or 3a.
    # sum of 3 distinct others from pool [97..192] is in [97+98+99, 190+191+192] = [294, 573].
    # So a can divide sum only if sum = 2a (since 3a >= 288, but max sum 573, 3*97=291).
    # Actually, if a is small, it divides more.
    # Let's start with a pool of [100, 192] (93 elements).
    # We want to find an independent set of size up to 48.
    
    # Branch and Bound for MIS in 4-uniform hypergraph.
    # Vertices: V = {100, ..., 192}
    # Hyperedges: E = { {a,b,c,d} | a|b+c+d, b|a+c+d, c|a+b+d, d|a+b+c }
    
    # Optimization:
    # 1. Precompute for each vertex v, the set of vertices u such that {v,u} is contained in some hyperedge?
    #    No, hyperedges are size 4.
    #    Two vertices u, v conflict if there exist w, z such that {u,v,w,z} is a hyperedge.
    #    This is equivalent to: u and v cannot both be in the independent set if there is a hyperedge containing them.
    #    Wait, an independent set in a hypergraph means no hyperedge is fully contained in the set.
    #    So {u,v} can be in the set, as long as no w,z exist such that {u,v,w,z} is a hyperedge.
    #    This is NOT the same as a graph MIS.
    #    However, we can define a "conflict graph" where u~v if they appear in a common hyperedge?
    #    No. If u~v, it means we CAN'T have u,v,w,z all together.
    #    Standard MIS on graph G where edges are pairs that CANNOT co-exist?
    #    No, in 4-uniform hypergraph, 3 vertices can co-exist. 4 cannot if they form an edge.
    #    
    #    So, we are looking for a set S such that no 4-subset of S is a hyperedge.
    #    
    #    Algorithm:
    #    Branch on vertices.
    #    State: (Current Set S, Remaining Candidates R, Forbidden 4-sets F?)
    #    This is hard.
    
    #    Alternative:
    #    Since k=3, the condition is: For all a in S, a does not divide sum of any 3 distinct others in S\{a}.
    #    
    #    Let's use a Greedy approach with Backtracking (Limited Depth/Width) or Local Search with Bitsets.
    #    Given the time limit, a robust Local Search with fast validity checks is safer than a complex B&B that might time out.
    #    
    #    Let's stick to the family of Local Search but optimize it heavily.
    #    Current score 1.0667 (41 elements). Max 48.
    #    
    #    Improvement:
    #    1. Use bitsets for "divisibility by a".
    #       For each a, precompute set of pairs (u,v) such that a | (u+v+x) for any x? No.
    #       For each a, and each pair (u,v), we know if a | (u+v+x).
    #       Actually, for fixed a, u, v, the condition is x != a, x != u, x != v, and (x+u+v) % a != 0.
    #       
    #    2. Faster validity check for adding x to S:
    #       - Check if x divides sum of any triple in S.
    #       - Check if any a in S divides sum of (x + pair in S\{a,x}).
    #       
    #    Let's implement a "Simulated Annealing" or "Random Restarts" Local Search.
    #    
    #    Pool: [1, 192]
    #    Start with a random valid set of size ~40? Or greedy from high end.
    #    
    #    Let's try a different pool: [150, 192] has 43 elements.
    #    If we can keep all 43, that's better than 41.
    #    Let's check if [150..192] is valid.
    #    Max sum of 3 from [150..192] is 190+191+192 = 573.
    #    Min a is 150.
    #    150 divides 573? No.
    #    150 divides sum of 3 others?
    #    Others are > 150. Sum > 450.
    #    150 * 3 = 450.
    #    So 150 could divide a sum if sum = 450, 600.
    #    Max sum 573. So only 450 is possible.
    #    Can we have 3 distinct numbers in [151..192] summing to 450?
    #    Min sum 151+152+153 = 456.
    #    So 150 cannot divide any sum of 3 distinct others from [151..192].
    #    So 150 is "safe" against others in [150..192]?
    #    Wait, if a=150, others must be distinct from 150.
    #    Sum of 3 distinct from [151..192] is >= 456.
    #    150 * 3 = 450. 150 * 4 = 600.
    #    So 150 never divides the sum.
    #    
    #    What about a=151?
    #    Sum >= 456.
    #    151 * 3 = 453.
    #    151 * 4 = 604.
    #    So 151 never divides.
    #    
    #    Generally, if a > (max_sum_of_3_others) / 3, then a cannot divide the sum?
    #    No, a divides sum if sum = k*a.
    #    If sum < 2a, then k=1. But sum of 3 positive ints is > a if a is small?
    #    Here a ~ 150, sum ~ 450-570.
    #    So k can be 3.
    #    If sum = 3a, then a divides sum.
    #    For a=150, 3a=450. Min sum 456. So no.
    #    For a=151, 3a=453. Min sum 456. So no.
    #    For a=152, 3a=456. Min sum 456 (151+152+153? No, distinct from a).
    #    If a=152, others from [150,192]\{152}.
    #    Min sum 150+151+153 = 454.
    #    3*152 = 456.
    #    Can we get 456? 150+151+155 = 456. Yes.
    #    So 152 divides 150+151+155.
    #    So {150, 151, 152, 155} is a violation.
    #    
    #    So [150..192] is NOT valid.
    #    But it's very close.
    #    We can start with [150..192] and remove violations.
    #    
    #    Let's implement a fast removal algorithm on this pool.
    
    pool = list(range(150, 193))
    
    def is_valid_subset(subset):
        """Check if subset is valid. Optimized for small sets."""
        if len(subset) < 4:
            return True
        s_set = set(subset)
        for a in subset:
            others = [v for v in subset if v != a]
            for i in range(len(others)):
                for j in range(i+1, len(others)):
                    for k in range(j+1, len(others)):
                        if (others[i] + others[j] + others[k]) % a == 0:
                            return False
        return True

    def find_violations_fast(subset):
        """Return list of (a, b, c, d) violations."""
        violations = []
        if len(subset) < 4:
            return violations
        for i, a in enumerate(subset):
            others = [v for j, v in enumerate(subset) if j != i]
            for t in combinations(others, 3):
                if sum(t) % a == 0:
                    violations.append((a, t[0], t[1], t[2]))
        return violations

    # Greedy Removal
    current = list(pool)
    while True:
        if time.time() > deadline - 5:
            break
        violations = find_violations_fast(current)
        if not violations:
            break
        
        # Count occurrences
        counts = {x: 0 for x in current}
        for v in violations:
            for x in v:
                counts[x] += 1
        
        # Remove element with highest count
        max_count = max(counts.values())
        candidates = [x for x in current if counts[x] == max_count]
        # Prefer removing smaller numbers? Or larger?
        # Smaller numbers are more likely to divide sums.
        to_remove = min(candidates)
        current.remove(to_remove)
        
    # Now we have a valid set. Let's try to add more from [1, 149].
    # Candidates: [1, 149]
    candidates = [x for x in range(1, 150) if x not in current]
    
    # Local Search: Add one if valid.
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for x in candidates:
            if time.time() > deadline - 2:
                break
            if x in current:
                continue
            # Check validity
            valid = True
            # 1. x divides sum of 3 in current
            if len(current) >= 3:
                for t in combinations(current, 3):
                    if sum(t) % x == 0:
                        valid = False
                        break
            if not valid:
                continue
            
            # 2. a in current divides x + pair in current\{a}
            for a in current:
                others = [v for v in current if v != a]
                if len(others) >= 2:
                    for i in range(len(others)):
                        for j in range(i+1, len(others)):
                            if (x + others[i] + others[j]) % a == 0:
                                valid = False
                                break
                        if not valid:
                            break
                if not valid:
                    break
            
            if valid:
                current.append(x)
                candidates.remove(x)
                improved = True
                break
                
    # Final Polish: Try to swap elements to increase size?
    # Or just return current.
    
    # Ensure distinct and positive
    current = list(set(current))
    if not current:
        return [192, 191, 190, 189]
        
    return current

# EVOLVE-BLOCK-END
