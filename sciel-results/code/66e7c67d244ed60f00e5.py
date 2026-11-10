import time
from math import gcd
from itertools import combinations
import random

def construct():
    deadline = time.time() + 50  # Tighter deadline to ensure safety margin
    
    def distinct_sums(A, B):
        return {a + b for a in A for b in B}
    
    def is_pairwise_coprime(vals):
        lst = list(vals)
        n = len(lst)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd(lst[i], lst[j]) != 1:
                    return False
        return True
    
    def score_of(A, B):
        s = distinct_sums(A, B)
        if len(s) < 2:
            return -1
        if not is_pairwise_coprime(s):
            return -1
        return len(s)
    
    # Start with known valid solution (score 5)
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score_of(best_A, best_B)
    
    # Quick check for a few hardcoded/promising candidates to ensure fast exit if possible
    # We don't know a specific 9-sum one, so we rely on search.
    
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    target_sums_list = []
    
    # Prioritize smaller primes as they are more likely to form valid grids within bounds
    # Generate combinations of 8 primes + 1
    for combo in combinations(small_primes[:15], 8):
        target = [1] + list(combo)
        target.sort()
        target_sums_list.append(target)
        
    # Generate combinations of 9 primes
    for combo in combinations(small_primes[:15], 9):
        target = list(combo)
        target.sort()
        target_sums_list.append(target)
        
    random.seed(42)
    random.shuffle(target_sums_list)
    
    for target in target_sums_list:
        if time.time() > deadline - 5:
            break
        if len(target) != 9:
            continue
        t = sorted(target)
        t1, t2, t3, t4, t5, t6, t7, t8, t9 = t
        
        x_min = max(1, t1 - 128)
        x_max = min(128, t1 - 1)
        y_min = max(1, t9 - 128)
        y_max = min(128, t9 - 1)
        
        x_range = range(x_min, x_max + 1)
        y_range = range(y_min, y_max + 1)
        
        if len(x_range) > 20:
            x_range = list(x_range)[::max(1, len(x_range)//20)]
        if len(y_range) > 20:
            y_range = list(y_range)[::max(1, len(y_range)//20)]
        
        for x in x_range:
            if time.time() > deadline - 3:
                break
            b1 = t1 - x
            if b1 < 1 or b1 > 128:
                continue
            for y in y_range:
                if time.time() > deadline - 3:
                    break
                b3 = t9 - y
                if b3 < 1 or b3 > 128:
                    continue
                
                s_fixed1 = x + t9 - y
                s_fixed2 = y + t1 - x
                
                if s_fixed1 not in t[1:8] or s_fixed2 not in t[1:8]:
                    continue
                
                remaining = set(t[1:8]) - {s_fixed1, s_fixed2}
                
                z_min = max(1, t2 - x, t2 - y)
                z_max = min(128, t8 - x, t8 - y)
                if z_max < z_min:
                    continue
                
                z_step = max(1, (z_max - z_min) // 10)
                for z in range(z_min, z_max + 1, z_step):
                    if time.time() > deadline - 2:
                        break
                    if x + z not in remaining:
                        continue
                    if y + z not in remaining:
                        continue
                    
                    rem3 = remaining - {x + z, y + z}
                    if len(rem3) != 3:
                        continue
                    
                    for rv in rem3:
                        w = rv - b1
                        if w < 1 or w > 128:
                            continue
                        s1 = w + b1
                        s2 = w + z
                        s3 = w + b3
                        if {s1, s2, s3} == rem3:
                            A = sorted([x, w, y])
                            B = sorted([b1, z, b3])
                            if len(set(A)) == 3 and len(set(B)) == 3:
                                if all(1 <= v <= 128 for v in A) and all(1 <= v <= 128 for v in B):
                                    sc = score_of(A, B)
                                    if sc >= 9:
                                        return A, B
                                    if sc > best_score:
                                        best_score = sc
                                        best_A = A
                                        best_B = B
    
    if best_score < 9:
        random.seed(42)
        while time.time() < deadline - 2:
            A = sorted(random.sample(range(1, 129), 3))
            B = sorted(random.sample(range(1, 129), 3))
            sc = score_of(A, B)
            if sc > best_score:
                best_score = sc
                best_A = A
                best_B = B
                if best_score == 9:
                    break
            
            # Also try structured random
            step_a = random.randint(10, 40)
            step_b = random.randint(10, 40)
            a1 = random.randint(1, 128 - 2*step_a)
            b1 = random.randint(1, 128 - 2*step_b)
            A2 = [a1, a1 + step_a, a1 + 2*step_a]
            B2 = [b1, b1 + step_b, b1 + 2*step_b]
            if all(1 <= v <= 128 for v in A2) and all(1 <= v <= 128 for v in B2):
                sc2 = score_of(A2, B2)
                if sc2 > best_score:
                    best_score = sc2
                    best_A = A2
                    best_B = B2
                    if best_score == 9:
                        break
    
    return best_A, best_B
