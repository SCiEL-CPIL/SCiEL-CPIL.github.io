"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    n = p - 1  # 1008
    
    # Compute modular inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    best_start = 2
    best_len = 0
    
    # For each possible start position, find the maximum valid end
    for a in range(1, n + 1):
        # Skip self-inverse residues: 1 and 1008 (since 1*1 % 1009 = 1, 1008*1008 % 1009 = 1)
        if a == 1 or a == 1008:
            continue
        
        # Find the maximum b >= a such that for all x, y in [a, b], x*y % 1009 != 1
        # The constraint is: for every x in [a,b], inv(x) must NOT be in [a,b]
        # (since if inv(x) is in [a,b], then x * inv(x) % 1009 = 1)
        # Also need to check x=x: x*x % 1009 != 1, which means x != 1 and x != 1008 (already handled)
        
        # For a fixed start a, the interval [a, b] is valid iff for all x in [a, b]:
        #   inv(x) is not in [a, b]
        # This means: for all x in [a, b], either inv(x) < a or inv(x) > b
        
        # So b must be less than min(inv(x) for x in [a,b] if inv(x) > a)
        # Equivalently, as we extend b, the constraint is that b < min(inv(x) for x in [a, b] where inv(x) > a)
        
        # We can compute this greedily: start with b = a, and extend as far as possible
        # The blocking constraint for extending to b is: for all x in [a, b], inv(x) not in [a, b]
        # i.e., for all x in [a, b], inv(x) < a or inv(x) > b
        
        # Let's compute the minimum "right endpoint" that blocks us.
        # For each x in [a, b], if inv(x) > a, then b must be < inv(x).
        # So b_max = min(inv(x) for x in [a, b_max] where inv(x) > a) - 1
        
        # We can compute this iteratively or use a sweep.
        # Let's try: for each a, compute the maximum valid b by scanning.
        
        # Actually, let's think differently. For a fixed a, define:
        #   f(a) = max b such that [a, b] is valid
        # [a, b] is valid iff for all x in [a, b], inv(x) not in [a, b]
        # iff for all x in [a, b], inv(x) < a or inv(x) > b
        
        # The tightest constraint comes from x in [a, b] with inv(x) > a.
        # For such x, we need b < inv(x).
        # So b < min{inv(x) : x in [a, b], inv(x) > a}
        
        # This is a self-referential constraint. Let's compute it by expanding b.
        
        b = a
        # Check if [a, a] is valid: need inv(a) not in [a, a], i.e., inv(a) != a
        # inv(a) == a means a^2 % p == 1, so a == 1 or a == p-1 = 1008. Already skipped.
        # So [a, a] is always valid for a not in {1, 1008}.
        
        # Now try to extend b.
        # We need: for all x in [a, b], inv(x) not in [a, b].
        # As b increases, we add new x = b. We need inv(b) not in [a, b].
        # Also, for all previous x in [a, b-1], we need inv(x) not in [a, b].
        # But if inv(x) was not in [a, b-1] and inv(x) != b, then inv(x) not in [a, b].
        # So the only new constraint when extending from b-1 to b is:
        #   inv(b) not in [a, b] AND for all x in [a, b-1], inv(x) != b
        
        # Wait, the second condition: for x in [a, b-1], inv(x) != b means x != inv(b).
        # So we need inv(b) not in [a, b], which already implies inv(b) != b and inv(b) < a or inv(b) > b.
        # But also, if inv(b) is in [a, b-1], then for x = inv(b), inv(x) = b, so x * inv(x) = b * inv(b) = 1 mod p.
        # So the condition "inv(b) not in [a, b]" covers both cases.
        
        # So the algorithm is:
        #   b = a
        #   while b+1 <= n and inv(b+1) not in [a, b+1]:
        #       b += 1
        #   But wait, we also need to check that for all x in [a, b], inv(x) not in [a, b].
        #   The greedy extension works because when we extend to b+1, we only need to check
        #   that inv(b+1) not in [a, b+1] AND for all x in [a, b], inv(x) != b+1.
        #   The latter is equivalent to: for all x in [a, b], x != inv(b+1).
        #   Since inv is a permutation, inv(b+1) is a single value. If inv(b+1) in [a, b], then
        #   the pair (inv(b+1), b+1) both in [a, b+1] gives product 1. So we need inv(b+1) not in [a, b+1].
        
        # So the greedy works!
        
        while b < n:
            nb = b + 1
            if inv[nb] >= a and inv[nb] <= nb:
                break
            b = nb
        
        length = b - a + 1
        if length > best_len:
            best_len = length
            best_start = a
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
