import itertools
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # 15s safety margin before 90s kill
    
    # The goal: find a set S of 12 distinct integers in [0,128] such that
    # the largest Sidon subset (including diagonals, i.e., a_i + a_j for i<=j)
    # is as small as possible.
    
    # Key insight: A Sidon set (with diagonals) requires all sums a_i + a_j 
    # for i <= j to be distinct. So for a subset T of size k, there are k(k+1)/2 
    # pair sums (with repetition allowed for i=j), and all must be distinct.
    
    # We want to MINIMIZE the maximum size of a Sidon subset within our ambient set S.
    
    # Strategy: Model as finding a set S where many 4-element subsets have 
    # sum collisions. A 4-element set {a,b,c,d} fails to be Sidon if there exist
    # i<=j and k<=l (not both same pair) such that a_i+a_j = a_k+a_l.
    
    # For a set S, the largest Sidon subset can be found via maximum independent set
    # in a conflict graph where vertices are elements of S and edges connect pairs 
    # that "conflict" in the sense that including both creates a sum collision 
    # with some other pair. Actually, the proper formulation is:
    
    # Two elements don't directly conflict; rather, a 4-subset conflicts if it has
    # a repeated sum. The largest Sidon subset is the largest subset with no 4 
    # (or fewer) elements having a repeated pair sum.
    
    # Better approach: Use a greedy/local search that:
    # 1. Starts with a candidate set S of 12 elements
    # 2. Computes the largest Sidon subset size via exact search (backtracking)
    # 3. Tries to modify S to reduce that maximum size
    
    # First, let's build a function to compute the exact largest Sidon subset size
    
    def largest_sidon_size(S):
        """Compute the largest subset of S that is Sidon (including diagonals)."""
        n = len(S)
        best = 0
        
        def is_sidon_subset(subset):
            """Check if subset is Sidon including diagonals."""
            sums = set()
            for i in range(len(subset)):
                for j in range(i, len(subset)):
                    s = subset[i] + subset[j]
                    if s in sums:
                        return False
                    sums.add(s)
            return True
        
        def backtrack(idx, current, remaining):
            nonlocal best
            if remaining == 0:
                return
            if len(current) + remaining <= best:
                return
            
            # Try not including element idx
            backtrack(idx + 1, current, remaining - 1)
            
            # Try including element idx if it still allows Sidon
            new_current = current + [S[idx]]
            if is_sidon_subset(new_current):
                best = max(best, len(new_current))
                backtrack(idx + 1, new_current, remaining - 1)
        
        # Try all subsets by size from largest to smallest
        # Use backtracking for efficiency
        backtrack(0, [], n)
        return best
    
    # A more efficient approach for n=12: try all subsets from size 12 down
    # But that's 2^12 = 4096 subsets, which is fine.
    
    def largest_sidon_size_brute(S):
        """Brute force: try all subsets from largest to smallest."""
        n = len(S)
        for size in range(n, 0, -1):
            for subset in itertools.combinations(range(n), size):
                elems = [S[i] for i in subset]
                if is_sidon_subset(elems):
                    return size
        return 0
    
    def is_sidon_subset(elems):
        """Check if elems is Sidon including diagonals."""
        sums = set()
        for i in range(len(elems)):
            for j in range(i, len(elems)):
                s = elems[i] + elems[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    # Now, the search strategy:
    # We want to find S of size 12 in [0,128] minimizing largest_sidon_size(S).
    
    # Lower bound on largest Sidon subset: For any set of 12 integers in [0,128],
    # there's a minimum possible largest Sidon subset. Let's think about what's achievable.
    
    # A Sidon set of size k in [0, 128] needs k(k+1)/2 distinct sums in [0, 256].
    # So k(k+1)/2 <= 257 => k <= 22 (since 22*23/2=253, 23*24/2=276>257).
    # But that's for the whole interval. For our specific 12-element set, the largest 
    # Sidon subset could be as small as... well, at least 2 (any two distinct elements 
    # with distinct self-sums, i.e., a!=b means 2a != 2b, and a+b is different from 
    # 2a and 2b if a!=b). Actually any 2-element set {a,b} with a!=b: sums are 2a, 2b, a+b. 
    # These are distinct iff 2a != 2b (true), 2a != a+b (i.e., a!=b, true), 2b != a+b (true). 
    # So any 2-element set is Sidon. So largest Sidon subset is at least 2.
    
    # Can we make it exactly 2? That would mean every 3-element subset has a sum collision.
    # For 3 elements {a,b,c} with a<b<c: sums are 2a, 2b, 2c, a+b, a+c, b+c.
    # Collision if any two are equal. Since a<b<c, 2a<2b<2c<a+b<a+c<b+c except when...
    # Actually 2a < a+b < a+c < b+c and 2a < 2b < 2c. 
    # Possible collisions: a+b = 2c? No, a+b < 2c since b<c. 
    # a+c = 2b? Yes if c = 2b - a. Or b+c = 2a? No. 
    # 2b = a+c? Same as a+c=2b.
    # So {a,b,c} is NOT Sidon iff a+c = 2b (i.e., b is midpoint of a and c) or 
    # a+b = 2c (impossible for a<b<c) or 2a = b+c (impossible) etc.
    # More carefully: the 6 sums are 2a, a+b, a+c, 2b, b+c, 2c (in some order).
    # With a<b<c: 2a < a+b < a+c < 2b < b+c < 2c is not necessarily true.
    # Let's just check all pairs.
    
    # So to have largest Sidon subset = 2, every triple in S must have a+b=2c type collision.
    # This means S must be such that for every triple, one element is the midpoint of the other two.
    # This is very restrictive. For example, an arithmetic progression {0, d, 2d, ..., 11d} 
    # has many midpoints. Let's check: in AP {0,d,2d,...,11d}, the triple {0,d,2d} has 
    # 0+2d=2d=2*d, so collision. But what about {0,d,3d}? Sums: 0, d, 3d, 2d, 4d, 6d. 
    # All distinct! So {0,d,3d} IS Sidon. So AP has larger Sidon subsets.
    
    # Let's think differently. We'll use local search:
    # 1. Generate random 12-element sets
    # 2. Evaluate largest Sidon subset size
    # 3. Try mutations (swap one element) and accept if it improves
    
    # But first, let's try some structured sets.
    
    # Idea: Use a set with many "almost arithmetic" progressions to create collisions.
    
    # Let's do a systematic search with time limit.
    
    best_S = list(range(12))
    best_score = largest_sidon_size_brute(best_S)
    
    # Try some structured candidates first
    structured_candidates = []
    
    # Arithmetic progressions
    for start in range(0, 100):
        for d in range(1, 15):
            S = [start + i*d for i in range(12)]
            if all(0 <= x <= 128 for x in S):
                structured_candidates.append(S)
    
    # Geometric-like
    for start in range(1, 50):
        for ratio in [2, 3]:
            S = []
            x = start
            for _ in range(12):
                if x > 128:
                    break
                S.append(x)
                x *= ratio
            if len(S) == 12 and all(0 <= v <= 128 for v in S):
                structured_candidates.append(S)
    
    # Mixed patterns
    for base in range(0, 20):
        for d1 in range(1, 10):
            for d2 in range(1, 10):
                S = []
                for i in range(12):
                    if i % 2 == 0:
                        S.append(base + (i//2)*d1)
                    else:
                        S.append(base + ((i-1)//2)*d2 + 5)
                if len(set(S)) == 12 and all(0 <= x <= 128 for x in S):
                    structured_candidates.append(S)
    
    for S in structured_candidates:
        score = largest_sidon_size_brute(S)
        if score < best_score:
            best_score = score
            best_S = S[:]
        
        if time.time() > deadline:
            break
    
    # Now do local search from best_S
    random.seed(42)
    
    def random_neighbor(S):
        """Create a random neighbor by swapping one element."""
        S_new = S[:]
        idx = random.randint(0, 11)
        old_val = S_new[idx]
        # Try to find a new value
        for _ in range(50):
            new_val = random.randint(0, 128)
            if new_val != old_val and new_val not in S_new:
                S_new[idx] = new_val
                return S_new
        return None
    
    for iteration in range(2000):
        if time.time() > deadline:
            break
        
        # Random restart sometimes
        if random.random() < 0.05:
            S = [random.randint(0, 128) for _ in range(12)]
            if len(set(S)) < 12:
                continue
        else:
            S = random_neighbor(best_S)
            if S is None:
                continue
        
        score = largest_sidon_size_brute(S)
        if score < best_score:
            best_score = score
            best_S = S[:]
            if best_score <= 2:
                break
    
    # One more phase: try to improve by specific perturbations
    if best_score > 2 and time.time() < deadline:
        for _ in range(500):
            if time.time() > deadline:
                break
            
            # Try flipping a pair of elements
            i, j = random.sample(range(12), 2)
            S_try = best_S[:]
            a, b = S_try[i], S_try[j]
            
            for _ in range(20):
                new_a = random.randint(0, 128)
                new_b = random.randint(0, 128)
                if new_a == new_b:
                    continue
                if new_a in best_S or new_b in best_S:
                    continue
                S_try[i] = new_a
                S_try[j] = new_b
                if len(set(S_try)) < 12:
                    continue
                score = largest_sidon_size_brute(S_try)
                if score < best_score:
                    best_score = score
                    best_S = S_try[:]
                    break
            S_try = best_S[:]  # reset
    
    return best_S

# EVOLVE-BLOCK-END
