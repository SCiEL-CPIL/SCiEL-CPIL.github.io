import time
import random
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin

    # Attempt to construct a 50-vertex coloring using cyclic group Z_50
    # This requires partitioning Z_50\{0} into 4 sum-free sets.
    # We reduce this to partitioning Z_25\{0} into 3 sum-free sets (for the even part)
    # and taking the odd part as the 4th set.
    
    def is_sum_free_in_Z25(S):
        # S is a set of ints in 1..24
        # Check if any a+b mod 25 is in S
        # Note: a+b can be 0..48. If a+b == 25, sum is 0 (not in Z_25\{0})
        # If a+b > 25, sum is a+b-25.
        for a in S:
            for b in S:
                s = (a + b) % 25
                if s != 0 and s in S:
                    return False
        return True

    # Elements to partition: 1..24
    elements = list(range(1, 25))
    # We need to split into 3 sets U1, U2, U3
    # This is a 3-coloring problem with constraints.
    # We can use a simple backtracking search.
    
    # To speed up, we can fix some assignments or use heuristics.
    # However, 24 elements is small enough for a fast backtracking if well-pruned.
    
    def solve_partition():
        # State: list of 3 sets, current index
        # We assign elements one by one.
        # Optimization: Sort elements? Maybe by "difficulty" (degree of constraint).
        # For now, try 0..23.
        
        sets = [set(), set(), set()]
        
        def backtrack(idx):
            if idx == 24:
                # Check if all sets are sum-free
                # Actually, we check incrementally?
                # Checking at the end is cheaper if we prune during assignment?
                # Pruning: When adding x to set k, check if x creates a sum-free violation.
                # If x in S_k, then for any y in S_k, (x+y)%25 must not be in S_k.
                # And for any z in S_k, (x+z)%25 must not be in S_k.
                # Basically, S_k U {x} must be sum-free.
                return True
            
            x = idx + 1
            for k in range(3):
                new_set = sets[k] | {x}
                if is_sum_free_in_Z25(new_set):
                    sets[k] = new_set
                    if backtrack(idx + 1):
                        return True
                    sets[k].remove(x)
            return False

        # Start with 0
        if backtrack(0):
            return [sets[0], sets[1], sets[2]]
        return None

    solution = solve_partition()
    
    if solution:
        # Construct the 50-vertex graph
        n = 50
        colors = [0] * (n * (n - 1) // 2)
        
        # Color 0: Odd differences (1, 3, ..., 49)
        # Color 1, 2, 3: Even differences (2, 4, ..., 48) mapped to 2 * (elements in U1, U2, U3)
        
        # Map difference d to color
        # d is in 1..49
        # If d is odd: color 0
        # If d is even: d = 2 * y, y in 1..24
        #   if y in U1: color 1
        #   if y in U2: color 2
        #   if y in U3: color 3
        
        U1, U2, U3 = solution
        map_even = {}
        for y in U1: map_even[2*y] = 1
        for y in U2: map_even[2*y] = 2
        for y in U3: map_even[2*y] = 3
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                # d is in 1..49
                if d % 2 != 0:
                    c = 0
                else:
                    c = map_even[d]
                colors[idx] = c
                idx += 1
        
        return {'n': n, 'colors': colors}
    else:
        # Fallback to 32-vertex construction
        # (Reuse the logic from the original program or a simpler valid one)
        # For simplicity, return a small valid graph if 50 fails
        # But 50 should work if a cyclic solution exists.
        # If not, we might need to hardcode the non-cyclic one.
        # Given the risk, I will keep the 32-vertex fallback from the original code structure 
        # but simplified.
        
        # Re-implement the 32-vertex Paley-like construction
        def mul(a, b):
            r = 0
            for i in range(4):
                if b >> i & 1:
                    r ^= a << i
            for i in range(7, 3, -1):
                if r >> i & 1:
                    r ^= 0b10011 << (i - 4)
            return r

        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)

        n = 32
        graph = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if i // 16 == j // 16:
                    c = log[(i % 16) ^ (j % 16)] % 3
                else:
                    c = 3
                graph[i][j] = c
                graph[j][i] = c

        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(graph[i][j])
        
        return {'n': n, 'colors': colors}
