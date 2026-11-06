"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    deadline = time.time() + 85.0  # 85s safety margin within 90s limit
    n = 128
    
    # Precompute gcd matrix
    gcd_mat = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            gcd_mat[i][j] = gcd(i, j)
            gcd_mat[j][i] = gcd_mat[i][j]
    
    # Precompute conflict bitmasks:
    # conflict_mask[x] is a bitmask where bit y is set if x and y are in a "bad pair"
    # i.e., there exists some z such that gcd(x,y) == gcd(x,z) == gcd(y,z)
    # Actually, we need: for element x, which pairs (a,b) in the selected set would
    # form a bad triple with x. Let's use a different approach.
    
    # For each element x, precompute conflict_with[x] as a set of elements y
    # such that adding x to a set containing y might create a conflict.
    # More precisely: x conflicts with pair (a,b) if gcd(a,b)==gcd(a,x)==gcd(b,x).
    # So for a given x, the "conflicting elements" are those a,b where (a,b) is a bad pair for x.
    
    # Let's define: for each x, bad_pairs_of_x = set of (a,b) pairs that form bad triple with x
    # Then x can be added to S iff no pair in S is in bad_pairs_of_x
    
    # For fast checking, let's use: for each x, conflict_mask[x] is a bitmask of all elements
    # that appear in any bad pair with x. But that's not quite right either.
    
    # Better approach: for each x, precompute the set of "forbidden pairs" as a frozenset
    # and check if any pair from selected is in that set.
    # For speed, use a dictionary mapping pair to a bitmask or just check membership.
    
    # Precompute: for each x, the set of (min(a,b), max(a,b)) pairs that form bad triple with x
    bad_pairs_of_x = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            g_ab = gcd_mat[a][b]
            for x in range(1, n + 1):
                if x == a or x == b:
                    continue
                if gcd_mat[a][x] == g_ab and gcd_mat[b][x] == g_ab:
                    bad_pairs_of_x[x].add((a, b))
                    bad_pairs_of_x[a].add((x, b)) if x > b else bad_pairs_of_x[a].add((b, x))
                    bad_pairs_of_x[b].add((a, x)) if x > a else bad_pairs_of_x[b].add((x, a))
    
    # Even better: use bitmask representation
    # conflict_bitmask[x] = bitmask of elements y such that (x,y) is part of some bad triple
    # No, that's still not right. Let me think differently.
    
    # Key insight: x can be added to S iff for all pairs (a,b) in S, (a,b) not in bad_pairs_of_x[x]
    # So we need: does S contain any pair from bad_pairs_of_x[x]?
    # This is equivalent to: is there a pair (a,b) in S where a,b are both "conflict partners" of x?
    
    # Let's precompute for each x: conflict_partners[x] = set of all elements that appear in bad pairs with x
    # But even that's not sufficient - we need the PAIR to be in bad_pairs_of_x[x]
    
    # For n=128, we can use a 128-bit integer as a bitmask for the selected set
    # and precompute for each x and each possible pair, whether it's bad.
    
    # Alternative: for each x, precompute a list of (a,b) bad pairs.
    # To check if x can be added to S: iterate over bad pairs of x, check if both are in S.
    # If |bad_pairs_of_x[x]| is small on average, this is fast.
    
    # Let's count average size
    # Actually, let's just use the bitmask approach for the selected set
    # and check pair membership via a precomputed table.
    
    # Precompute: for each pair (a,b) with a<b, a bitmask of all x that form bad triple with (a,b)
    # bad_triple_mask[a][b] = bitmask of x such that gcd(a,b)==gcd(a,x)==gcd(b,x)
    bad_triple_mask = [[0] * (n + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            g_ab = gcd_mat[a][b]
            mask = 0
            for x in range(1, n + 1):
                if x == a or x == b:
                    continue
                if gcd_mat[a][x] == g_ab and gcd_mat[b][x] == g_ab:
                    mask |= (1 << x)
            bad_triple_mask[a][b] = mask
            bad_triple_mask[b][a] = mask
    
    def can_add(x, selected_mask):
        """Check if x can be added to the set represented by selected_mask."""
        # x can be added iff no pair (a,b) in selected_mask has x in bad_triple_mask[a][b]
        # Equivalently: for all a,b in selected_mask, x is NOT in bad_triple_mask[a][b]
        # Which means: for all a,b in selected_mask, (selected_mask & bad_triple_mask[a][b]) has no x bit
        # Wait, that's not right. x is already determined. We need: for all pairs (a,b) in S,
        # gcd(a,b) != gcd(a,x) or gcd(a,b) != gcd(b,x)
        # i.e., x is NOT in bad_triple_mask[a][b] for any pair (a,b) in S
        
        # Iterate over all pairs in S and check
        # For speed, iterate over elements a in S, then b in S with b > a
        # But this is O(|S|^2) which is slow for large S.
        
        # Better: for each a in S, check if there exists b in S such that x in bad_triple_mask[a][b]
        # For fixed a and x, the set of b such that x in bad_triple_mask[a][b] is:
        # {b : gcd(a,b) == gcd(a,x) == gcd(b,x)}
        # Let g_ax = gcd(a,x). Then we need b such that gcd(a,b) == g_ax and gcd(b,x) == g_ax.
        # Precompute for each (a, x): the set of b that would make (a,b,x) a bad triple.
        
        # Actually, let's precompute conflict_mask[a][x] = bitmask of b such that (a,b,x) is bad
        # Then x can be added to S iff for all a in S, (conflict_mask[a][x] & selected_mask) == 0
        # Wait, that's still not quite right. We need: for all pairs (a,b) in S, not bad.
        # x is bad with (a,b) iff a in S, b in S, and x in bad_triple_mask[a][b].
        # Equivalently: there exist a,b in S such that x in bad_triple_mask[a][b].
        # For fixed x and a, the b's that cause trouble are those with x in bad_triple_mask[a][b].
        # So: x can be added iff for all a in S, no b in S satisfies x in bad_triple_mask[a][b].
        # i.e., for all a in S: (selected_mask & conflict_mask[a][x]) == 0
        # where conflict_mask[a][x] = {b : x in bad_triple_mask[a][b]}
        
        # Let's precompute this.
        pass  # We'll compute this below
    
    # Precompute conflict_mask[a][x] = bitmask of b such that (a,b,x) forms a bad triple
    # i.e., x in bad_triple_mask[a][b]
    conflict_mask = [[0] * (n + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for x in range(1, n + 1):
            if a == x:
                continue
            mask = 0
            for b in range(1, n + 1):
                if b == a or b == x:
                    continue
                if x in bad_triple_mask[a][b] if False else (bad_triple_mask[a][b] >> x) & 1:
                    mask |= (1 << b)
            conflict_mask[a][x] = mask
    
    def can_add_fast(x, selected_mask):
        """Check if x can be added to the set represented by selected_mask."""
        for a in range(1, n + 1):
            if (selected_mask >> a) & 1:
                if selected_mask & conflict_mask[a][x]:
                    return False
        return True
    
    def greedy_add(order):
        """Greedy add elements in given order."""
        selected_mask = 0
        for x in order:
            if not (selected_mask >> x) & 1:
                if can_add_fast(x, selected_mask):
                    selected_mask |= (1 << x)
        return selected_mask
    
    def mask_to_set(mask):
        return [i for i in range(1, n + 1) if (mask >> i) & 1]
    
    def set_to_mask(s):
        mask = 0
        for x in s:
            mask |= (1 << x)
        return mask
    
    # Greedy removal using bitmask
    def greedy_remove():
        """Start with all elements, remove those that create conflicts."""
        selected_mask = 0
        for i in range(1, n + 1):
            selected_mask |= (1 << i)
        
        changed = True
        while changed:
            changed = False
            for x in range(1, n + 1):
                if not ((selected_mask >> x) & 1):
                    continue
                # Check if x has any conflict with the rest
                rest_mask = selected_mask & ~(1 << x)
                # x conflicts if there exist a,b in rest_mask such that (a,b,x) is bad
                # i.e., for some a in rest_mask, rest_mask & conflict_mask[a][x] != 0
                has_conflict = False
                for a in range(1, n + 1):
                    if (rest_mask >> a) & 1:
                        if rest_mask & conflict_mask[a][x]:
                            has_conflict = True
                            break
                if has_conflict:
                    selected_mask &= ~(1 << x)
                    changed = True
        return selected_mask
    
    # Get initial solution
    best_mask = greedy_remove()
    best_size = bin(best_mask).count('1')
    
    # Try multiple greedy-add restarts
    random.seed(42)
    
    for attempt in range(300):
        if time.time() > deadline - 5.0:
            break
        
        if attempt % 4 == 0:
            order = list(range(1, n + 1))
            random.shuffle(order)
        elif attempt % 4 == 1:
            order = sorted(range(1, n + 1), key=lambda x: bin(conflict_mask[x][0] if False else 0).count('1'))
            # Sort by degree (number of conflicts)
            degs = []
            for x in range(1, n + 1):
                deg = 0
                for y in range(1, n + 1):
                    if x != y and (conflict_mask[x][y] or conflict_mask[y][x]):
                        deg += 1
                degs.append((x, deg))
            order = [x for x, d in sorted(degs, key=lambda t: t[1])]
        elif attempt % 4 == 2:
            order = list(range(n, 0, -1))
            random.shuffle(order)
        else:
            order = list(range(1, n + 1))
            random.shuffle(order)
        
        candidate_mask = greedy_add(order)
        cand_size = bin(candidate_mask).count('1')
        if cand_size > best_size:
            best_mask = candidate_mask
            best_size = cand_size
    
    # Local search: 1-for-1 swaps and 1-for-2 swaps
    if time.time() < deadline - 10.0:
        for iteration in range(100):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            selected_list = mask_to_set(best_mask)
            
            # Try 1-for-1 swaps: remove x, add y
            for x in selected_list:
                if time.time() > deadline - 5.0:
                    break
                rest_mask = best_mask & ~(1 << x)
                for y in range(1, n + 1):
                    if (best_mask >> y) & 1:
                        continue
                    if can_add_fast(y, rest_mask):
                        new_mask = rest_mask | (1 << y)
                        # This is same size, so no improvement from 1-for-1
                        # But it might enable further improvements
                        # Skip for now, focus on 1-for-2
                        pass
            
            # Try 1-for-2 swaps: remove x, add y and z
            for x in selected_list:
                if time.time() > deadline - 5.0:
                    break
                rest_mask = best_mask & ~(1 << x)
                
                # Find all y that can be added to rest_mask
                addable = []
                for y in range(1, n + 1):
                    if (best_mask >> y) & 1:
                        continue
                    if can_add_fast(y, rest_mask):
                        addable.append(y)
                
                # Try pairs from addable
                for i in range(len(addable)):
                    if time.time() > deadline - 5.0:
                        break
                    y1 = addable[i]
                    temp_mask = rest_mask | (1 << y1)
                    for j in range(i + 1, len(addable)):
                        y2 = addable[j]
                        if can_add_fast(y2, temp_mask):
                            new_mask = temp_mask | (1 << y2)
                            new_size = bin(new_mask).count('1')
                            if new_size > best_size:
                                best_mask = new_mask
                                best_size = new_size
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
    
    # Final: try to add any remaining elements
    if time.time() < deadline - 3.0:
        for x in range(1, n + 1):
            if (best_mask >> x) & 1:
                continue
            if can_add_fast(x, best_mask):
                best_mask |= (1 << x)
                best_size += 1
    
    return mask_to_set(best_mask)
# EVOLVE-BLOCK-END
