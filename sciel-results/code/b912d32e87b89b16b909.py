import random
import math
import time
from math import gcd

def construct():
    # Precompute GCD table for values up to 256
    max_val = 256
    gcd_table = [[0] * (max_val + 1) for _ in range(max_val + 1)]
    for i in range(1, max_val + 1):
        for j in range(i, max_val + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
    
    def is_pairwise_coprime(sums_list):
        """Check if all distinct values in sums_list are pairwise coprime."""
        distinct = set(sums_list)
        distinct_list = list(distinct)
        n = len(distinct_list)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd_table[distinct_list[i]][distinct_list[j]] > 1:
                    return False
        return True
    
    def count_distinct_sums(A, B):
        """Count distinct sums in A+B."""
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        return len(sums)
    
    def evaluate(A, B):
        """Score a candidate: number of distinct sums if pairwise coprime, else 0."""
        sums = []
        for a in A:
            for b in B:
                sums.append(a + b)
        
        # Check if all distinct sums are pairwise coprime
        distinct = set(sums)
        distinct_list = list(distinct)
        n = len(distinct_list)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd_table[distinct_list[i]][distinct_list[j]] > 1:
                    return 0
        return len(distinct)
    
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Strategy 1: Search over difference vectors (a0, u1, u2, b0, v1, v2)
    # A = [a0, a0+u1, a0+u1+u2], B = [b0, b0+v1, b0+v1+v2]
    # All values must be in [1, 128] and distinct within each set
    
    def random_diff_vector():
        """Generate random difference vectors ensuring all values in [1,128] and distinct."""
        while True:
            a0 = random.randint(1, 128)
            u1 = random.randint(-30, 30)
            u2 = random.randint(-30, 30)
            b0 = random.randint(1, 128)
            v1 = random.randint(-30, 30)
            v2 = random.randint(-30, 30)
            
            A = [a0, a0 + u1, a0 + u1 + u2]
            B = [b0, b0 + v1, b0 + v1 + v2]
            
            # Check bounds and distinctness
            if all(1 <= x <= 128 for x in A) and len(set(A)) == 3:
                if all(1 <= x <= 128 for x in B) and len(set(B)) == 3:
                    return A, B
    
    # Strategy 2: Random sampling as fallback
    evens = [i for i in range(1, 129) if i % 2 == 0]
    odds = [i for i in range(1, 129) if i % 2 != 0]
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Alternate between strategies
        if attempts % 2 == 0:
            A, B = random_diff_vector()
        else:
            A = random.sample(evens, 3)
            B = random.sample(odds, 3)
        
        score = evaluate(A, B)
        if score > best_score:
            best_score = score
            best_A = A
            best_B = B
            if best_score == 9:
                return [best_A, best_B]
    
    return [best_A, best_B]
