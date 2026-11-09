import time
import sys
from math import gcd

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin before 90s kill

    def elapsed():
        return time.time() - start

    N = 256
    numbers = list(range(1, N + 1))
    
    # Conflict check: (a+b) | (a*b)
    # Let g = gcd(a,b), a=gu, b=gv, gcd(u,v)=1
    # Condition: (u+v) | g
    def inc(a, b):
        if a == b:
            return False
        g = gcd(a, b)
        u = a // g
        v = b // g
        return (u + v) % g == 0

    # 1. Separate odds and evens
    odds = [i for i in numbers if i % 2 != 0]
    evens = [i for i in numbers if i % 2 == 0]
    
    # 2. Filter evens: keep only those compatible with ALL odds
    safe_evens = []
    for e in evens:
        compatible = True
        for o in odds:
            if inc(e, o):
                compatible = False
                break
        if compatible:
            safe_evens.append(e)
            
    # 3. Build conflict graph for safe_evens
    n_ev = len(safe_evens)
    # Map value to index in safe_evens for quick lookup if needed, 
    # but we just need adjacency masks.
    # adj[i] is a bitmask of indices j such that safe_evens[i] and safe_evens[j] conflict.
    adj = [0] * n_ev
    
    for i in range(n_ev):
        ei = safe_evens[i]
        mask = 0
        for j in range(n_ev):
            if i == j:
                continue
            ej = safe_evens[j]
            if inc(ei, ej):
                mask |= (1 << j)
        adj[i] = mask

    # 4. Maximum Independent Set (MIS) on safe_evens
    # Using a standard Branch and Bound algorithm.
    # We want to maximize the number of vertices in an independent set.
    # An independent set in the conflict graph is a set of vertices with no edges between them.
    
    # We use a recursive solver with bitsets.
    # State: available (bitmask of vertices available to add), current_size, current_set
    # Upper bound: number of vertices in available.
    # To improve pruning, we can use a greedy coloring bound or simply rely on the small n.
    # n_ev is likely small (evens compatible with all odds are rare).
    # Let's estimate n_ev.
    # If e is even, g=gcd(e,o) is odd. u=e/g even, v=o/g odd. u+v odd.
    # (u+v) | g.
    # Since u = e/g, v = o/g, u+v = (e+o)/g.
    # So (e+o)/g | g => e+o | g^2.
    # g <= e, g <= o.
    # For large e, o, g is small, g^2 is small, e+o is large.
    # So conflicts are rare.
    # Most evens are likely safe.
    # Let's check a few.
    # e=2. g=1. u=2, v=o. u+v=o+2. (o+2)|1 -> No.
    # e=4. g=1. u=4, v=o. u+v=o+4. (o+4)|1 -> No.
    # e=6. o=3. g=3. u=2, v=1. u+v=3. 3|3 -> Yes. Conflict.
    # e=6. o=9. g=3. u=2, v=3. u+v=5. 5|3 -> No.
    # e=6. o=15. g=3. u=2, v=5. u+v=7. 7|3 -> No.
    # e=6. o=21. g=3. u=2, v=7. u+v=9. 9|3 -> No.
    # e=6. o=27. g=3. u=2, v=9. u+v=11. 11|3 -> No.
    # e=6. o=33. g=3. u=2, v=11. u+v=13. 13|3 -> No.
    # e=6. o=39. g=3. u=2, v=13. u+v=15. 15|3 -> No.
    # e=6. o=45. g=3. u=2, v=15. u+v=17. 17|3 -> No.
    # e=6. o=51. g=3. u=2, v=17. u+v=19. 19|3 -> No.
    # e=6. o=57. g=3. u=2, v=19. u+v=21. 21|3 -> No.
    # e=6. o=63. g=3. u=2, v=21. u+v=23. 23|3 -> No.
    # e=6. o=69. g=3. u=2, v=23. u+v=25. 25|3 -> No.
    # e=6. o=75. g=3. u=2, v=25. u+v=27. 27|3 -> No.
    # e=6. o=81. g=3. u=2, v=27. u+v=29. 29|3 -> No.
    # e=6. o=87. g=3. u=2, v=29. u+v=31. 31|3 -> No.
    # e=6. o=93. g=3. u=2, v=31. u+v=33. 33|3 -> No.
    # e=6. o=99. g=3. u=2, v=33. u+v=35. 35|3 -> No.
    # e=6. o=105. g=3. u=2, v=35. u+v=37. 37|3 -> No.
    # e=6. o=111. g=3. u=2, v=37. u+v=39. 39|3 -> No.
    # e=6. o=117. g=3. u=2, v=39. u+v=41. 41|3 -> No.
    # e=6. o=123. g=3. u=2, v=41. u+v=43. 43|3 -> No.
    # e=6. o=129. g=3. u=2, v=43. u+v=45. 45|3 -> No.
    # e=6. o=135. g=3. u=2, v=45. u+v=47. 47|3 -> No.
    # e=6. o=141. g=3. u=2, v=47. u+v=49. 49|3 -> No.
    # e=6. o=147. g=3. u=2, v=49. u+v=51. 51|3 -> No.
    # e=6. o=153. g=3. u=2, v=51. u+v=53. 53|3 -> No.
    # e=6. o=159. g=3. u=2, v=53. u+v=55. 55|3 -> No.
    # e=6. o=165. g=3. u=2, v=55. u+v=57. 57|3 -> No.
    # e=6. o=171. g=3. u=2, v=57. u+v=59. 59|3 -> No.
    # e=6. o=177. g=3. u=2, v=59. u+v=61. 61|3 -> No.
    # e=6. o=183. g=3. u=2, v=61. u+v=63. 63|3 -> No.
    # e=6. o=189. g=3. u=2, v=63. u+v=65. 65|3 -> No.
    # e=6. o=195. g=3. u=2, v=65. u+v=67. 67|3 -> No.
    # e=6. o=201. g=3. u=2, v=67. u+v=69. 69|3 -> No.
    # e=6. o=207. g=3. u=2, v=69. u+v=71. 71|3 -> No.
    # e=6. o=213. g=3. u=2, v=71. u+v=73. 73|3 -> No.
    # e=6. o=219. g=3. u=2, v=73. u+v=75. 75|3 -> No.
    # e=6. o=225. g=3. u=2, v=75. u+v=77. 77|3 -> No.
    # e=6. o=231. g=3. u=2, v=77. u+v=79. 79|3 -> No.
    # e=6. o=237. g=3. u=2, v=79. u+v=81. 81|3 -> No.
    # e=6. o=243. g=3. u=2, v=81. u+v=83. 83|3 -> No.
    # e=6. o=249. g=3. u=2, v=83. u+v=85. 85|3 -> No.
    # e=6. o=255. g=3. u=2, v=85. u+v=87. 87|3 -> No.
    
    # It seems 6 is safe?
    # Wait, o=3 was a conflict.
    # So 6 is NOT safe.
    
    # The number of safe evens will be determined at runtime.
    
    best_size = 0
    best_set = frozenset()
    
    # Recursive MIS solver
    sys.setrecursionlimit(10000)
    
    def popcount(x):
        return bin(x).count('1')
    
    # Greedy upper bound: 
    # A simple upper bound is the number of vertices.
    # A better bound can be obtained by a greedy coloring of the complement graph,
    # but for small n, simple B&B is often fast enough.
    # We will use a simple B&B with vertex ordering by degree.
    
    # Order vertices by degree descending to find large sets faster?
    # Or ascending to prune?
    # Standard MIS: pick vertex with max degree.
    
    def solve(available, current_size, current_set):
        nonlocal best_size, best_set
        
        if elapsed() > deadline:
            return
            
        if available == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = frozenset(current_set)
            return
            
        # Upper bound
        ub = popcount(available)
        if current_size + ub <= best_size:
            return
            
        # Pick a vertex
        # Heuristic: pick vertex with max degree in available subgraph
        temp = available
        best_v = -1
        max_deg = -1
        while temp:
            v = (temp & (-temp)).bit_length() - 1
            deg = popcount(adj[v] & available)
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp &= temp - 1
            
        if best_v < 0:
            return
            
        # Branch 1: Include best_v
        # Remove best_v and its neighbors
        new_available = available & ~(1 << best_v) & ~adj[best_v]
        new_set = set(current_set)
        new_set.add(best_v)
        solve(new_available, current_size + 1, frozenset(new_set))
        
        # Branch 2: Exclude best_v
        new_available = available & ~(1 << best_v)
        solve(new_available, current_size, current_set)
        
    initial_available = (1 << n_ev) - 1
    solve(initial_available, 0, frozenset())
    
    # Construct final set
    final_set = set(odds)
    for idx in best_set:
        final_set.add(safe_evens[idx])
        
    # Verification (optional, but good for safety)
    # The problem guarantees distinct pairs.
    # We just need to ensure the returned list is sorted and valid.
    
    return sorted(final_set)

if __name__ == "__main__":
    result = construct()
    print(len(result))
