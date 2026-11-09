"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    n = 256
    
    def is_feasible(S):
        """Check if set S satisfies the property: for all a<b<=c in S, a does not divide b+c."""
        S_sorted = sorted(S)
        k = len(S_sorted)
        for i in range(k):
            a = S_sorted[i]
            for j in range(i+1, k):
                b = S_sorted[j]
                for k2 in range(j, k):
                    c = S_sorted[k2]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def check_add(S, candidate):
        """Check if adding candidate to S maintains feasibility."""
        S_new = sorted(S + [candidate])
        k = len(S_new)
        # Only need to check triples where candidate is involved
        # But for safety, we check all triples involving the new element
        # Actually, let's just check all triples to be safe for small sets
        for i in range(k):
            a = S_new[i]
            for j in range(i+1, k):
                b = S_new[j]
                for k2 in range(j, k):
                    c = S_new[k2]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def check_add_fast(S_sorted, candidate):
        """Faster check: only check triples where candidate is a, b, or c."""
        S_new = sorted(S_sorted + [candidate])
        k = len(S_new)
        # Find index of candidate in S_new
        # Check all triples, but we can be smarter
        # Actually for correctness let's check all triples
        for i in range(k):
            a = S_new[i]
            for j in range(i+1, k):
                b = S_new[j]
                for k2 in range(j, k):
                    c = S_new[k2]
                    if (b + c) % a == 0:
                        return False
        return True
    
    # Strategy 1: Interval seed - numbers in (2n/3, n]
    # For a < b <= c in [k, n], we need a not to divide b+c
    # Max b+c = 2n, min a = k
    # If k > 2n/3, then b+c < 2n < 3k... wait that's not right
    # We need: for all a in S, b+c is not a multiple of a
    # b+c ranges from 2k to 2n (if b,c >= k)
    # a ranges from k to n
    # b+c = m*a for some integer m >= 1
    # Since b+c >= 2k and a <= n, m <= 2n/k
    # If k > n/2, then m < 4, so m can be 1, 2, 3
    # m=1: b+c = a, impossible since b,c > a (since a < b)
    # m=2: b+c = 2a, possible
    # m=3: b+c = 3a, possible
    # So we need to avoid b+c = 2a and b+c = 3a
    # For interval [k, n]: avoid b+c=2a means no a,b,c in [k,n] with b+c=2a
    # and b+c=3a means no a,b,c in [k,n] with b+c=3a
    
    # Simple interval: [129, 256] has 128 elements
    # Let's verify: a in [129,256], b,c in [129,256], a<b<=c
    # b+c in [258, 512]
    # a divides b+c: b+c = ma
    # m=1: b+c=a, impossible (b>a)
    # m=2: b+c=2a, a in [129,256], 2a in [258,512], b+c in [258,512] - possible!
    # So [129,256] is NOT feasible in general
    
    # Let's try [171, 256]: 86 elements
    # a in [171,256], b,c in [171,256], a<b<=c
    # b+c in [342, 512]
    # m=2: b+c=2a, 2a in [342,512] - possible
    # m=3: b+c=3a, 3a in [513,768] - b+c max is 512, so m=3 impossible
    # So only need to check m=2: b+c=2a
    # a in [171,256], b,c in [171,256] with a<b<=c and b+c=2a
    # This means a is the average of b and c
    # For the interval [171,256], can we have three elements where one is the average of the other two?
    # Yes, e.g., 171, 172, 173: 171 < 172 <= 173, 172+173=345, 2*171=342, no
    # 171, 173, 175: 173+175=348, 2*171=342, no
    # We need b+c=2a exactly. So a=(b+c)/2, meaning b and c have the same parity.
    # E.g., a=172, b=173, c=171? No, need a<b<=c, so 172<173<=c, b+c=173+c=2*172=344, c=171, but c must be >=b=173. Contradiction.
    # Wait, a<b<=c, so b>a and c>=b>a. So b+c > 2a always! 
    # Because b>=a+1 and c>=b>=a+1, so b+c >= 2a+2 > 2a.
    # So b+c=2a is impossible when a<b<=c!
    # Similarly b+c=3a: b+c >= 2a+2, and 3a >= 3(a+1) for a>=1... 
    # For a>=171: 3a >= 513, b+c <= 2*256=512. So b+c < 3a always.
    # Therefore for [171, 256], no a divides b+c. Feasible!
    
    # Can we go lower? Try [170, 256]: 87 elements
    # a=170, b,c in [171,256], b+c in [342,512]
    # 3*170=510. b+c=510 possible? b+c=510 with b,c in [171,256], b<=c
    # e.g., b=254, c=256: 254+256=510=3*170. And 170<254<=256. So a=170 divides b+c=510. NOT feasible.
    
    # Try [171, 256] confirmed feasible with 86 elements.
    
    # Strategy 2: Greedy from small to large
    selected_greedy = []
    for num in range(1, n+1):
        if check_add_fast(sorted(selected_greedy), num):
            selected_greedy.append(num)
    
    # Strategy 3: Greedy from large to small
    selected_large = []
    for num in range(n, 0, -1):
        if check_add_fast(sorted(selected_large), num):
            selected_large.append(num)
    
    # Strategy 4: Interval [171, 256]
    selected_interval = list(range(171, 257))
    
    # Try to improve interval by adding smaller elements
    selected_improved = list(selected_interval)
    for num in range(1, 171):
        if time.time() > deadline:
            break
        if check_add_fast(sorted(selected_improved), num):
            selected_improved.append(num)
    
    # Local search: try to add any element not in the set
    def try_add_elements(S):
        S = list(S)
        S_sorted = sorted(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for num in range(1, n+1):
                if num not in S:
                    if check_add_fast(S_sorted, num):
                        S.append(num)
                        S_sorted = sorted(S)
                        improved = True
        return S
    
    # Apply local search to each strategy
    best = selected_interval
    for S in [selected_greedy, selected_large, selected_improved]:
        if time.time() > deadline:
            break
        S_improved = try_add_elements(S)
        if len(S_improved) > len(best):
            best = S_improved
    
    # Also try local search on the interval itself
    if time.time() < deadline:
        S_improved = try_add_elements(selected_interval)
        if len(S_improved) > len(best):
            best = S_improved
    
    # Try swap-based local search
    def local_search_swaps(S, max_iterations=100):
        S = list(S)
        S_sorted = sorted(S)
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            improved = False
            # Try removing one element and adding two
            for i in range(len(S)):
                if time.time() > deadline:
                    break
                S_test = S[:i] + S[i+1:]
                S_test_sorted = sorted(S_test)
                # Try to add any two elements
                for a in range(1, n+1):
                    if a in S_test:
                        continue
                    if not check_add_fast(S_test_sorted, a):
                        continue
                    S_test2 = S_test + [a]
                    S_test2_sorted = sorted(S_test2)
                    for b in range(1, n+1):
                        if b in S_test2:
                            continue
                        if b == a:
                            continue
                        if check_add_fast(S_test2_sorted, b):
                            S_test3 = S_test2 + [b]
                            if is_feasible(S_test3):
                                S = S_test3
                                S_sorted = sorted(S)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return S
    
    if time.time() < deadline:
        S_swapped = local_search_swaps(best, max_iterations=20)
        if len(S_swapped) > len(best):
            best = S_swapped
    
    # Final verification
    if not is_feasible(best):
        # Fallback to known feasible set
        best = list(range(171, 257))
    
    if not best:
        best = list(range(171, 257))
    
    return best
# EVOLVE-BLOCK-END
