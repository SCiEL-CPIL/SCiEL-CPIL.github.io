import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit
    
    # Try to find a valid partition for n=26
    # Elements: 1..13
    # Sets: 5
    # Condition: No 4 elements in any set sum to 0 mod 26 (with distinct vertices)
    
    def is_valid_set(S):
        # Check for 4-sums to 0 mod 26
        # a+b+c+d = 0
        # We need distinct vertices 0, a, a+b, a+b+c
        # This requires a, b, c, d non-zero and partial sums distinct and non-zero
        elems = list(S)
        n = len(elems)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        a, b, c, d = elems[i], elems[j], elems[k], elems[l]
                        if (a + b + c + d) % 26 == 0:
                            # Check distinct vertices
                            v0 = 0
                            v1 = a
                            v2 = (a + b) % 26
                            v3 = (a + b + c) % 26
                            if v1 != 0 and v2 != 0 and v3 != 0 and v1 != v2 and v2 != v3 and v1 != v3:
                                # Also need v3 != 0 (already checked) and edge (v3, 0) diff d
                                # d = (0 - v3) % 26 = (-v3) % 26
                                # We used d as the 4th element.
                                # So we need d == (-v3) % 26
                                if d == (-v3) % 26:
                                    return False
        return True

    def solve_n26():
        elements = list(range(1, 14))
        # Randomized search
        for _ in range(5000):
            if time.time() > deadline:
                return None
            random.shuffle(elements)
            sets = [set() for _ in range(5)]
            # Greedy assignment with backtracking or just random shuffle and check
            # Simple random partition
            for e in elements:
                # Assign to random set
                sets[random.randint(0, 4)].add(e)
            
            # Check validity
            valid = True
            for S in sets:
                if not is_valid_set(S):
                    valid = False
                    break
            if valid:
                return sets
        return None

    sets = solve_n26()
    if sets:
        n = 26
        # Map difference to color
        diff_to_color = {}
        for c, S in enumerate(sets):
            for d in S:
                diff_to_color[d] = c
        
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if d > n // 2:
                    d = n - d
                colors.append(diff_to_color[d])
        return {'n': n, 'colors': colors}

    # Fallback to n=25 if n=26 fails (though it shouldn't)
    # Or just use n=11 seed
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
