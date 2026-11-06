"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import sys

def construct():
    n = 192
    
    def can_add(a, S, S_set):
        """
        Check if element a can be added to set S without violating constraints.
        S is a list, S_set is a set of elements in S.
        """
        if len(S) < 3:
            return True
            
        # Check 1: a divides sum of any 3 distinct elements in S
        # x, y, z in S, x!=y!=z!=x
        # a | (x+y+z)
        # Iterate pairs (x, y) in S
        # z = k*a - x - y
        # z must be in S, z != x, z != y
        
        # Optimization: 
        # min_sum = sum of 3 smallest in S
        # max_sum = sum of 3 largest in S
        # But we can just iterate pairs and check possible z.
        
        for i in range(len(S)):
            x = S[i]
            for j in range(i+1, len(S)):
                y = S[j]
                s_xy = x + y
                # z = k*a - s_xy
                # z must be in S
                # k*a = s_xy + z
                # z in [min(S), max(S)]
                min_z = min(S)
                max_z = max(S)
                min_ma = s_xy + min_z
                max_ma = s_xy + max_z
                
                m_start = (min_ma + a - 1) // a
                m_end = max_ma // a
                
                for m in range(m_start, m_end + 1):
                    z = m * a - s_xy
                    if z in S_set and z != x and z != y:
                        return False
        
        # Check 2: Any u in S divides sum of any 3 distinct elements in S U {a}
        # where the 3 elements include a and two from S (distinct from u)
        # Let the 3 elements be a, x, y where x, y in S \ {u}
        # u | (a + x + y)
        # Iterate pairs (x, y) in S
        # val = a + x + y
        # Check if any divisor u of val is in S \ {x, y}
        
        for i in range(len(S)):
            x = S[i]
            for j in range(i+1, len(S)):
                y = S[j]
                val = a + x + y
                # Check divisors of val
                # We only care about divisors u that are in S
                # and u != x, u != y
                
                # Iterate through S to check? O(|S|)
                # Or iterate divisors of val?
                # Divisors of val are few.
                # Let's iterate divisors of val up to sqrt(val)
                
                # Actually, iterating S is O(|S|) which is small (<=48).
                # So total O(|S|^3) for this part?
                # No, the loop is O(|S|^2) for pairs.
                # Inside, we check divisors.
                # If we iterate divisors, it's fast.
                
                # Let's find all divisors of val
                divs = []
                d = 1
                while d * d <= val:
                    if val % d == 0:
                        divs.append(d)
                        if d != val // d:
                            divs.append(val // d)
                    d += 1
                
                for u in divs:
                    if u in S_set and u != x and u != y:
                        return False
                        
        return True

    def greedy(order):
        S = []
        S_set = set()
        for a in order:
            if can_add(a, S, S_set):
                S.append(a)
                S_set.add(a)
        return S

    best = []
    
    # Descending order
    order_desc = list(range(n, 0, -1))
    res = greedy(order_desc)
    if len(res) > len(best):
        best = res
        
    # Ascending order
    order_asc = list(range(1, n + 1))
    res = greedy(order_asc)
    if len(res) > len(best):
        best = res
        
    # Random restarts
    random.seed(42)
    for _ in range(200):
        order = list(range(1, n + 1))
        random.shuffle(order)
        res = greedy(order)
        if len(res) > len(best):
            best = res
            
    return best

# EVOLVE-BLOCK-END
