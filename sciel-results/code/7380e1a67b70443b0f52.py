import time
from fractions import Fraction
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    N = 256
    # Precompute reciprocals as Fractions for exact arithmetic
    recip_frac = [Fraction(0)] * (N + 1)
    recip_float = [0.0] * (N + 1)
    for i in range(1, N + 1):
        recip_frac[i] = Fraction(1, i)
        recip_float[i] = 1.0 / i
    
    best_score_frac = Fraction(0)
    best_S = set()
    
    # Precompute all 3-APs
    # A 3-AP is (a, b, c) where a + c = 2b, a != c
    # For each element x, precompute which other elements form a 3-AP with x
    # conflicts[x] = set of y such that {x, y} is part of some 3-AP
    # Actually, we need: for a set S, an element x conflicts with S if there exist a, b in S
    # such that {a, b, x} forms a 3-AP.
    
    # Better approach: For each pair (a, c) with a != c, the middle is (a+c)/2.
    # For each element x, the pairs (a, c) such that x is the middle: a + c = 2x, a != c
    # For each element x, the pairs (a, c) such that x is an endpoint: 
    #   x as first: x + c = 2b => c = 2b - x
    #   x as last: a + x = 2b => a = 2b - x
    
    # Let's precompute for each x, the set of "forbidden partners"
    # If x is in S, then for any other y in S, we must not have a 3-AP.
    # This is equivalent to: for each x, precompute which y cannot coexist with x in S
    # such that together with some z they form a 3-AP. But this depends on z.
    
    # Actually, the constraint is: no three distinct elements a, b, c in S with a+c=2b.
    # So for a given set S, x can be added iff for all a in S:
    #   1. 2x - a not in S (x as middle)
    #   2. (a+x)/2 not in S if a+x even (x as endpoint)
    
    # For B&B, we need efficient conflict checking.
    # Let's use a bitset representation for speed.
    
    # Represent S as a bitset (integer with bits set)
    # For each x, precompute:
    #   mid_conflicts[x] = set of a such that 2x - a is in [1,N] and a != x
    #   end_conflicts[x] = set of a such that (a+x)/2 is in [1,N] and (a+x) even and a != x
    
    # Actually, let's precompute for each x, the list of pairs that would form a 3-AP.
    # For B&B, we'll maintain S as a set and check validity efficiently.
    
    def is_valid_addition(S, x):
        """Check if x can be added to S without creating a 3-AP."""
        # x as middle: for each a in S, check if 2x-a in S
        for a in S:
            c = 2 * x - a
            if c != x and c in S:
                return False
        # x as endpoint: for each a in S, check if (a+x)/2 in S
        for a in S:
            s = a + x
            if s % 2 == 0:
                c = s // 2
                if c != x and c != a and c in S:
                    return False
        return True
    
    def get_conflicts_with_new(S, x):
        """Return set of elements that conflict with adding x to S.
        These are elements y not in S that would form a 3-AP with x and some element in S."""
        conflicts = set()
        # If we add x, then for each a in S:
        # 1. 2x - a is forbidden (x as middle with a)
        # 2. (a+x)/2 is forbidden (x as endpoint with a) if valid
        for a in S:
            c = 2 * x - a
            if 1 <= c <= N and c != x and c != a:
                conflicts.add(c)
            s = a + x
            if s % 2 == 0:
                c = s // 2
                if 1 <= c <= N and c != x and c != a:
                    conflicts.add(c)
        return conflicts
    
    # Heuristic: Greedy + Local Search
    def heuristic_solve():
        S = set()
        # Greedy: add smallest numbers first (they have larger reciprocals)
        for x in range(1, N + 1):
            if is_valid_addition(S, x):
                S.add(x)
        
        # 1-swap hill climb
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            # Try removing each element and adding a smaller one
            for x_out in sorted(S, reverse=True):
                if time.time() > deadline - 10.0:
                    break
                S_temp = S.copy()
                S_temp.remove(x_out)
                # Try adding elements smaller than x_out
                for x_in in range(1, x_out):
                    if x_in in S_temp:
                        continue
                    if is_valid_addition(S_temp, x_in):
                        S = S_temp
                        S.add(x_in)
                        improved = True
                        break
                if improved:
                    break
        return S
    
    # Initial heuristic solution
    S_init = heuristic_solve()
    best_score_frac = sum(recip_frac[x] for x in S_init)
    best_S = S_init.copy()
    
    # B&B with exact Fraction upper bounds
    # State: S (set), C (set of candidates), current_sum (Fraction)
    
    def dfs(S, C, current_sum):
        nonlocal best_score_frac, best_S
        
        if time.time() > deadline:
            return
        
        # Upper bound: current_sum + sum of reciprocals of all elements in C
        # This is a valid upper bound since we can't get more than all of C
        ub = current_sum
        for y in C:
            ub += recip_frac[y]
        
        if ub <= best_score_frac:
            return
        
        if not C:
            if current_sum > best_score_frac:
                best_score_frac = current_sum
                best_S = S.copy()
            return
        
        # Pick the smallest x in C (largest reciprocal)
        x = min(C)
        
        # Branch 1: Include x
        if is_valid_addition(S, x):
            new_S = S.copy()
            new_S.add(x)
            
            # Compute new C: remove x and all elements that conflict with new_S
            new_C = set()
            for y in C:
                if y == x:
                    continue
                if is_valid_addition(new_S, y):
                    new_C.add(y)
            
            dfs(new_S, new_C, current_sum + recip_frac[x])
        
        # Branch 2: Exclude x
        new_C_excl = C.copy()
        new_C_excl.remove(x)
        dfs(S, new_C_excl, current_sum)
    
    # Start B&B from empty set
    dfs(set(), set(range(1, N + 1)), Fraction(0))
    
    return list(best_S)
