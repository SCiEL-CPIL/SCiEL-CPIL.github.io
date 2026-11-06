from math import gcd
from fractions import Fraction

def construct():
    # Greedy construction
    # We want to maximize cardinality.
    # Let's try adding elements from 24 down to 1.
    # If adding x violates the condition, skip it.
    
    S = []
    # Use Fractions for exact arithmetic
    # Condition: For all x in S, 1/x != sum(1/y for y in T) for T subset of S \ {x}
    
    for x in range(24, 0, -1):
        # Check if x can be added to S
        # 1. Check if 1/x is sum of reciprocals of subset of S
        # 2. Check if for any y in S, 1/y is sum of reciprocals of subset of S U {x} \ {y}
        
        can_add = True
        
        # Check 1: 1/x == sum(1/y for y in T) where T subset S
        # Since S is valid, no element in S is sum of others.
        # But 1/x might be.
        # We can check all subsets of S? S size is small.
        # Actually, if we build S greedily, S is always valid.
        # So we just need to check if x creates a new violation.
        
        # Violation type A: 1/x = sum(1/y for y in T) with T subset S
        # Violation type B: 1/y = 1/x + sum(1/z for z in U) with y in S, U subset S \ {y, x}
        
        # Check A
        target = Fraction(1, x)
        # Iterate subsets of S
        # S size is at most 17. 2^17 is 131072. This is fine.
        # But we can do better.
        # Since S is valid, the sums of subsets of S are unique? No.
        # But we can just iterate.
        
        n = len(S)
        # To optimize, we can precompute sums? No, just check.
        # Actually, if S is large, this might be slow if done for every x.
        # But 24 iterations * 2^17 is ~3 million operations. Python might be slow.
        # Let's optimize the subset sum check.
        
        # Alternative: Just check if x can be added.
        
        # Let's use a simpler check:
        # If we add x, the new set is S' = S U {x}.
        # We need S' to be valid.
        # Since S is valid, we only need to check equations involving x.
        
        # Eq 1: 1/x = sum(1/y for y in T), T subset S
        # Eq 2: 1/y = 1/x + sum(1/z for z in U), y in S, U subset S \ {y}
        
        # Check Eq 1
        # We can use a recursive search or bitmasks.
        # Given N=24, let's just use Fractions and iterate subsets.
        
        # Optimization: If sum(1/y for y in S) < 1/x, then no subset can sum to 1/x.
        sum_S = sum(Fraction(1, y) for y in S)
        if target > sum_S:
            # No subset of S can sum to 1/x
            pass
        else:
            # Check subsets
            # We can prune if current sum > target
            # But let's just do it simply first.
            
            # To make it fast, we can check if x is "safe"
            # A known heuristic: if x > 2 * max(S), then 1/x < 1/(2*max(S)).
            # Sum of reciprocals of any subset of S is at most sum(1/y for y in S).
            
            # Let's just iterate subsets for now.
            found = False
            # Use bitmasks for subsets of S
            for mask in range(1, 1 << n):
                s = Fraction(0)
                for i in range(n):
                    if mask & (1 << i):
                        s += Fraction(1, S[i])
                        if s > target:
                            break
                if s == target:
                    found = True
                    break
            if found:
                can_add = False
        
        if can_add:
            # Check Eq 2: 1/y = 1/x + sum(1/z for z in U)
            # => 1/x = 1/y - sum(1/z for z in U)
            # => 1/x = (1/y - sum(1/z for z in U))
            # For each y in S, we need to check if 1/x can be formed by 1/y - sum(U)
            # This is equivalent to: sum(1/z for z in U) = 1/y - 1/x
            # Let target_y = 1/y - 1/x
            # Check if target_y is a sum of reciprocals of a subset of S \ {y, x}
            
            for y in S:
                target_y = Fraction(1, y) - Fraction(1, x)
                if target_y <= 0:
                    continue
                # Check if target_y is sum of subset of S \ {y}
                # Note x is not in S yet, so S \ {y} is just the other elements.
                # We need to check subsets of S \ {y}
                # This is similar to Check 1 but for each y.
                
                # Optimization: if sum(S \ {y}) < target_y, skip
                sum_S_minus_y = sum_S - Fraction(1, y)
                if target_y > sum_S_minus_y:
                    continue
                
                found = False
                # Iterate subsets of S \ {y}
                # We can iterate all masks of S and ignore y
                for mask in range(1, 1 << n):
                    if mask & (1 << S.index(y)):
                        continue
                    s = Fraction(0)
                    for i in range(n):
                        if mask & (1 << i):
                            s += Fraction(1, S[i])
                            if s > target_y:
                                break
                    if s == target_y:
                        found = True
                        break
                if found:
                    can_add = False
                    break
        
        if can_add:
            S.append(x)
            
    return S
