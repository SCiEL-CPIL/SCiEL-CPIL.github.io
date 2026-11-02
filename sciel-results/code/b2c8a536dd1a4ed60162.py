"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s (or 150s) kill

    # The score is 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)
    # We want to maximize this. Note |U-U| >= |U+U| for sets containing 0 (since -U subset U-U and U+U subset U-U? No.)
    # Actually for U containing 0, U+U is the sumset. U-U is the difference set.
    # |U-U| can be as large as |U|(2|U|-1) if all differences are distinct.
    # |U+U| can be as small as 2|U|-1 (if U is an arithmetic progression).
    # So the ratio |U-U|/|U+U| can be large if U has few sums but many differences.
    # This happens when U is "additively structured" but "difference-rich".
    # However, if U is an AP, |U+U| = 2|U|-1 and |U-U| = 2|U|-1, so ratio is 1, score = 1.
    # We want ratio > 1.
    
    # Consider U = {0, 1, ..., n-1} union {M, M+1, ..., M+n-1} for large M.
    # Then U+U = [0, 2n-2] union [M, M+2n-2] union [M, 2M+2n-2]. If M > 2n-2, these are disjoint.
    # |U+U| = (2n-1) + (2n-1) + (n-1) = 5n-3? No.
    # Let's compute more carefully.
    # A = {0..n-1}, B = {M..M+n-1}. U = A ∪ B.
    # A+A = {0..2n-2} (size 2n-1)
    # B+B = {2M..2M+2n-2} (size 2n-1)
    # A+B = {M..M+2n-2} (size 2n-1)
    # If M > 2n-2, all three are disjoint.
    # |U+U| = 3(2n-1) = 6n-3.
    # U-U: A-A = {-(n-1)..n-1} (size 2n-1)
    # B-B = {-(n-1)..n-1} (size 2n-1)
    # A-B = {-M..-M+n-1} ∪ ... wait. A-B = {a-b : a in A, b in B} = {-(M+n-1)..-(M-1)} (size n)
    # B-A = {M-1..M+n-1} (size n)
    # If M is large, these are all disjoint.
    # |U-U| = (2n-1) + (2n-1) + n + n = 6n-2.
    # Ratio = (6n-2)/(6n-3) ≈ 1. Score ≈ 1. Not good.

    # What if we use a Sidon-like set for sums but not for differences?
    # Actually, we want |U-U| large and |U+U| small.
    # This is related to "sum-difference bases".
    
    # Let's try a random set and see.
    # Or better, let's use the known constructions from the paper mentioned.
    # The paper mentions sizes 2003 and 54265.
    # The score for the seed [0,1,3] is:
    # U+U = {0,1,2,3,4,6}, |U+U|=6
    # U-U = {-3,-2,-1,0,1,2,3}, |U-U|=7
    # max(U)=3, 2*max+1=7
    # score = 1 + log(7/6)/log(7) ≈ 1 + 0.154/1.946 ≈ 1.079.
    # Wait, the task says score is 1.0 for the seed. Let me re-read.
    # "score is the lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1) on C6"
    # "The evaluator measures the raw value (higher is better). The search score is value/seed, so 1.0 equals the seed program"
    # So the raw score for [0,1,3] is the seed value.
    # Let's compute: 1 + log(7/6)/log(7) = 1 + ln(1.166)/ln(7) = 1 + 0.154/1.946 = 1.079.
    # So the seed score is 1.079. Any set with raw score > 1.079 will have search score > 1.0.

    # Let's try to find a set with a better ratio.
    # We want |U-U| / |U+U| to be large, and 2*max(U)+1 to be small.
    # This suggests we want U to be small in magnitude but have a large difference set relative to its sum set.
    
    # Let's try a random search with local improvement.
    def score(U):
        if not U or 0 not in U:
            return 0
        U = sorted(set(U))
        max_u = max(U)
        if max_u == 0:
            return 1.0
        base = 2 * max_u + 1
        if base <= 1:
            return 1.0
        # Compute |U+U| and |U-U|
        # For small sets, we can compute directly.
        sum_set = set()
        diff_set = set()
        for a in U:
            for b in U:
                sum_set.add(a + b)
                diff_set.add(a - b)
        n_sum = len(sum_set)
        n_diff = len(diff_set)
        if n_sum == 0:
            return 1.0
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        return 1.0 + math.log(ratio) / math.log(base)

    # Start with the seed
    best_U = [0, 1, 3]
    best_score = score(best_U)
    
    # Try to improve by adding elements or changing elements.
    # Since we want small max(U), we should keep elements small.
    # Let's try random sets of size 3 to 10 with elements up to 50.
    
    random.seed(42)
    
    # Try some specific constructions
    # 1. Random sets
    for _ in range(5000):
        if time.time() > deadline - 10:
            break
        size = random.randint(3, 12)
        max_val = random.randint(10, 100)
        U = set([0])
        while len(U) < size:
            U.add(random.randint(1, max_val))
        U = list(U)
        s = score(U)
        if s > best_score:
            best_score = s
            best_U = U
    
    # 2. Local search: try to add or remove elements from best_U
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current_U = set(best_U)
        # Try adding an element
        for x in range(1, max(current_U) + 20):
            if x not in current_U:
                new_U = list(current_U | {x})
                s = score(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U
                    improved = True
                    break
        # Try removing an element (not 0)
        for x in list(current_U):
            if x != 0:
                new_U = list(current_U - {x})
                if len(new_U) >= 2:
                    s = score(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U
                        improved = True
                        break
        # Try replacing an element
        for x in list(current_U):
            if x != 0:
                for y in range(1, max(current_U) + 20):
                    if y not in current_U and y != x:
                        new_U = list((current_U - {x}) | {y})
                        s = score(new_U)
                        if s > best_score + 1e-9:
                            best_score = s
                            best_U = new_U
                            improved = True
                            break
                if improved:
                    break

    return sorted(best_U)
# EVOLVE-BLOCK-END
