"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    n = 180
    
    def can_add(a, s):
        """Check if a can be added to set s without creating a forbidden triple."""
        a2 = a * a
        for d in range(1, a2 + 1):
            if a2 % d == 0:
                e = a2 // d
                b = a + d
                c = a + e
                if b != c and b in s and c in s:
                    return False
        return True
    
    # Start with the safe set {91, ..., 180}
    current_set = set(range(91, n + 1))
    
    # Try to add numbers from 1 to 90 greedily
    # Sort candidates by how many conflicts they have with current_set
    candidates = list(range(1, 91))
    
    def num_conflicts(a, s):
        cnt = 0
        a2 = a * a
        for d in range(1, a2 + 1):
            if a2 % d == 0:
                e = a2 // d
                b = a + d
                c = a + e
                if b != c and b in s and c in s:
                    cnt += 1
        return cnt
    
    # Greedily add candidates with fewest conflicts
    remaining = set(candidates)
    changed = True
    while changed:
        changed = False
        best_a = None
        best_cnt = float('inf')
        for a in remaining:
            if can_add(a, current_set):
                cnt = num_conflicts(a, current_set)
                if cnt < best_cnt:
                    best_cnt = cnt
                    best_a = a
        if best_a is not None:
            current_set.add(best_a)
            remaining.discard(best_a)
            changed = True
    
    return sorted(current_set)
# EVOLVE-BLOCK-END
