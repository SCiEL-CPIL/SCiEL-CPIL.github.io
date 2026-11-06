"""Program for large integer sets avoiding equal-gcd triples."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    """
    Return nonempty subset of [1,128] with no three distinct elements whose
    three pairwise greatest common divisors are equal. Maximize cardinality.
    
    Approach: Model as maximum independent set in a 3-uniform hypergraph.
    Use branch-and-bound with bitset-based conflict propagation.
    """
    N = 128
    start_time = time.time()
    time_limit = 55  # Leave buffer for overhead
    
    # Precompute gcd table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # For each pair (a, b), precompute the set of c's that form a bad triple with them.
    # bad_with_pair[a][b] is a bitset of c's such that gcd(a,b) == gcd(a,c) == gcd(b,c)
    bad_with_pair = [[0] * (N + 1) for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd_table[a][b]
            mask = 0
            for c in range(1, N + 1):
                if c != a and c != b and gcd_table[a][c] == g_ab and gcd_table[b][c] == g_ab:
                    mask |= (1 << c)
            bad_with_pair[a][b] = mask
            bad_with_pair[b][a] = mask
    
    # Also precompute for each element x, the set of pairs (a,b) that with x form a bad triple
    # This is useful for checking conflicts when adding x
    # For element x, conflict_pairs[x] is a bitset of pairs... but pairs are 2D.
    # Instead, let's compute: for each x, forbidden[x] is the set of y such that
    # there exists z where (x,y,z) is a bad triple. But this isn't quite right for the search.
    
    # Actually, for the branch and bound, when we add x to the current set S,
    # we need to check if there exist a, b in S such that (a,b,x) is a bad triple.
    # This is equivalent to: for each a in S, check if bad_with_pair[a][x] intersects S.
    # Or more efficiently: precompute for each x, the "conflict set" given S.
    
    # Let's use a different representation:
    # For each x, precompute conflict_with[x] as a bitset of y's such that
    # there exists some z where (min(x,y,z), mid, max) forms a bad triple with x and y.
    # Wait, that's not right. The constraint is that NO THREE elements form a bad triple.
    # So if we have S and want to add x, we need: for all pairs (a,b) in S, (a,b,x) is NOT a bad triple.
    # This is equivalent to: for all a in S, bad_with_pair[a][x] does not intersect S.
    
    # Let's precompute for each x, a list of (a, mask_a) where mask_a = bad_with_pair[a][x]
    # This way, when checking if x can be added to S, we just check if any mask_a intersects S.
    
    # Actually, let's think about this differently. Let's use the following approach:
    # State: (current_set as bitset, remaining as bitset)
    # When we add x to current_set, the new forbidden set for future additions is:
    # For each y in remaining, y is forbidden if there exists z in current_set such that
    # (z, y, x) or (z, x, y) forms a bad triple with some other element... 
    # No, that's not right either.
    
    # Let me reconsider. The constraint is: no three elements in the final set form a bad triple.
    # So when we build the set incrementally, adding x is valid iff for all pairs (a,b) already in the set,
    # (a,b,x) is not a bad triple.
    
    # For efficiency, let's precompute:
    # For each x, and for each a, bad_with_pair[a][x] gives the set of b's that with a and x form a bad triple.
    # So when adding x to set S, we need to check: for all a in S, (bad_with_pair[a][x] & S) == 0.
    # But this is O(|S|) checks per addition, which might be slow.
    
    # Alternative: precompute for each x, a "conflict mask" that depends on S.
    # Actually, let's just use the direct check but optimize with bitsets.
    
    # Let's try a greedy approach first to get a good initial solution, then use local search.
    
    # Greedy: add elements in order of increasing "conflict count"
    # Conflict count for x: number of pairs (a,b) such that (a,b,x) is a bad triple
    conflict_count = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            mask = bad_with_pair[a][b]
            while mask:
                lsb = mask & -mask
                c = lsb.bit_length() - 1
                conflict_count[c] += 1
                mask ^= lsb
    
    # Try multiple greedy orderings
    def greedy_solve(order):
        selected = []
        selected_set = set()
        for x in order:
            can_add = True
            for a in selected:
                if bad_with_pair[a][x] & 0:  # placeholder
                    pass
                # Check if there exists b in selected such that (a,b,x) is bad
                # i.e., b is in bad_with_pair[a][x]
                if bad_with_pair[a][x]:
                    # Check if any element in selected is in bad_with_pair[a][x]
                    for b in selected:
                        if b != a and (1 << b) & bad_with_pair[a][x]:
                            can_add = False
                            break
                if not can_add:
                    break
            if can_add:
                selected.append(x)
                selected_set.add(x)
        return selected
    
    # This is too slow. Let me use a bitset-based approach.
    
    def greedy_solve_fast(order):
        selected_bits = 0
        for x in order:
            x_bit = 1 << x
            can_add = True
            # Check all a in selected
            temp = selected_bits
            while temp:
                lsb = temp & -temp
                a = lsb.bit_length() - 1
                if (bad_with_pair[a][x] & selected_bits):
                    can_add = False
                    break
                temp ^= lsb
            if can_add:
                selected_bits |= x_bit
        return selected_bits
    
    # Try different orderings
    order1 = sorted(range(1, N + 1), key=lambda x: conflict_count[x])
    order2 = list(range(1, N + 1))
    order3 = list(range(N, 0, -1))
    order4 = sorted(range(1, N + 1), key=lambda x: -conflict_count[x])
    
    best_bits = 0
    best_size = 0
    
    for order in [order1, order2, order3, order4]:
        bits = greedy_solve_fast(order)
        size = bits.bit_count()
        if size > best_size:
            best_size = size
            best_bits = bits
    
    # Now try local search: try to swap elements to improve
    # Or try to add elements by removing some
    
    # Let's try a branch and bound approach with time limit
    # State: (current_bits, remaining_bits)
    # We want to maximize popcount(current_bits)
    
    # Precompute: for each x, the list of (a, bad_with_pair[a][x]) for quick conflict checking
    # Actually, let's just use the direct bitset check.
    
    global_best = best_size
    global_best_bits = best_bits
    
    # Branch and bound
    def search(current_bits, remaining_bits, depth):
        nonlocal global_best, global_best_bits
        
        if time.time() - start_time > time_limit:
            return
        
        size = current_bits.bit_count()
        rem_size = remaining_bits.bit_count()
        
        if size + rem_size <= global_best:
            return
        
        if remaining_bits == 0:
            if size > global_best:
                global_best = size
                global_best_bits = current_bits
            return
        
        # Heuristic: pick the element in remaining with highest "conflict" in current set
        # Or just pick the first one for simplicity
        # Let's pick the element that appears in the most bad triples with current set elements
        
        best_x = -1
        best_score = -1
        temp = remaining_bits
        while temp:
            lsb = temp & -temp
            x = lsb.bit_length() - 1
            # Score: how many conflicts x has with current set
            score = 0
            temp2 = current_bits
            while temp2:
                lsb2 = temp2 & -temp2
                a = lsb2.bit_length() - 1
                if bad_with_pair[a][x] & current_bits:
                    score += 1
                temp2 ^= lsb2
            if score > best_score:
                best_score = score
                best_x = x
            temp ^= lsb
        
        if best_x == -1:
            return
        
        x = best_x
        x_bit = 1 << x
        
        # Branch 1: Exclude x
        search(current_bits, remaining_bits & ~x_bit, depth + 1)
        
        # Branch 2: Include x (if valid)
        can_add = True
        temp = current_bits
        while temp:
            lsb = temp & -temp
            a = lsb.bit_length() - 1
            if bad_with_pair[a][x] & current_bits:
                can_add = False
                break
            temp ^= lsb
        
        if can_add:
            search(current_bits | x_bit, remaining_bits & ~x_bit, depth + 1)
    
    all_bits = (1 << (N + 1)) - 2  # bits 1 through N set
    search(best_bits, all_bits & ~best_bits, 0)
    
    # Convert back to list
    result = []
    b = global_best_bits
    while b:
        lsb = b & -b
        v = lsb.bit_length() - 1
        result.append(v)
        b ^= lsb
    
    if not result:
        result = [1]
    
    return sorted(result)

# EVOLVE-BLOCK-END
