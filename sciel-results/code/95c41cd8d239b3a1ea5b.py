"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    n = 256
    # Start with the safe upper tail: elements where 3*a > 2*n, so no b+c (with b,c>a) can be a multiple of a
    # 3*a > 512 => a > 170.67, so a >= 171
    current_set = set(range(171, n + 1))
    
    # Try to add elements from 170 down to 1
    # For element a, it's compatible if for all b, c in current_set with b > a, c >= b:
    #   a does not divide (b + c)
    
    def is_compatible(a, s):
        """Check if a can be added to set s without violating constraints."""
        # For all b, c in s with b > a and c >= b (and b != c or we handle b=c):
        # Actually we need: for all a < b <= c in (s ∪ {a}), a ∤ (b+c)
        # Since a is the smallest new element, we only need to check pairs (b,c) where b,c ∈ s and b > a, c >= b
        # AND also pairs where one of them is a? No - a is the new smallest, so in a<b<=c, a is the divisor.
        # We need: for all b,c in s with b > a and b <= c: a ∤ (b+c)
        
        elements_above = [x for x in s if x > a]
        if not elements_above:
            return True
        
        # Check all pairs b <= c from elements_above
        # To be efficient, we can check: does there exist b,c in elements_above with b<=c and (b+c) % a == 0?
        # (b+c) % a == 0 means b+c is a multiple of a.
        # b+c ranges from 2*(a+1) to 2*n (but b,c <= n)
        # Actually b,c are in s which is subset of [1,n], and b,c > a, so b+c in [2(a+1), 2n]
        # Multiples of a in that range: a*k where a*k >= 2(a+1) and a*k <= 2n
        # i.e., k >= ceil(2(a+1)/a) and k <= floor(2n/a)
        
        min_sum = 2 * (a + 1)
        max_sum = 2 * n
        
        # Find if any multiple of a in [min_sum, max_sum] can be expressed as b+c for b,c in elements_above with b<=c
        # For efficiency, use a set for O(1) lookup
        above_set = set(elements_above)
        
        k_min = (min_sum + a - 1) // a  # ceil(min_sum / a)
        k_max = max_sum // a  # floor(max_sum / a)
        
        for k in range(k_min, k_max + 1):
            target = k * a
            # Need b + c = target with b, c in above_set, b <= c
            # So b can range from max(a+1, target - max(elements_above)) to min(target - (a+1), target//2)
            # But simpler: for b in above_set, check if target - b in above_set and b <= target - b
            # That's O(|above_set|) per k, which could be slow.
            # Alternative: iterate over b in elements_above, check if (target - b) in above_set and b <= target - b
            for b in elements_above:
                c = target - b
                if c >= b and c in above_set:
                    return False
        return True
    
    # Greedy: try adding from high to low (larger elements are "easier" to add since fewer constraints)
    # Actually, let's try from 170 down to 1
    for a in range(170, 0, -1):
        if time.time() > deadline:
            break
        if is_compatible(a, current_set):
            current_set.add(a)
    
    return sorted(current_set)
# EVOLVE-BLOCK-END
