import random
import math
import time
from itertools import combinations

def is_pairwise_coprime(nums):
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(nums[i], nums[j]) != 1:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill

    # Precompute coprime pairs for speed
    # coprime[i][j] = True if gcd(i,j) == 1
    coprime = [[False] * 129 for _ in range(129)]
    for i in range(1, 129):
        for j in range(i, 129):
            coprime[i][j] = math.gcd(i, j) == 1
            coprime[j][i] = coprime[i][j]

    odds = [i for i in range(1, 129) if i % 2 != 0]
    evens = [i for i in range(1, 129) if i % 2 == 0]
    all_vals = list(range(1, 129))

    def check_pairwise_coprime_fast(nums):
        """Use precomputed coprime table for speed."""
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if not coprime[nums[i]][nums[j]]:
                    return False
        return True

    def get_distinct_sums(A, B):
        return set(a + b for a in A for b in B)

    def score_pair(A, B):
        """Return (cardinality, is_coprime) for a pair."""
        sums = get_distinct_sums(A, B)
        card = len(sums)
        # Only check coprime if cardinality is promising
        if card >= 5:
            is_copr = check_pairwise_coprime_fast(list(sums))
        else:
            is_copr = False
        return card, is_copr

    # Initialize with seed
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_sums = get_distinct_sums(best_A, best_B)
    best_cardinality = len(best_sums)  # 5
    best_is_coprime = check_pairwise_coprime_fast(list(best_sums))

    def try_improve(A, B, max_iter=2000):
        """Local search: try replacing elements to improve."""
        current_A = list(A)
        current_B = list(B)
        current_sums = get_distinct_sums(current_A, current_B)
        current_card = len(current_sums)
        current_coprime = check_pairwise_coprime_fast(list(current_sums))
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try all possible replacements in A
            for idx in range(3):
                for new_val in all_vals:
                    if new_val in current_A:
                        continue
                    new_A = list(current_A)
                    new_A[idx] = new_val
                    new_sums = get_distinct_sums(new_A, current_B)
                    new_card = len(new_sums)
                    
                    if new_card > current_card or (new_card == current_card and new_card >= 5):
                        new_coprime = check_pairwise_coprime_fast(list(new_sums))
                        # Accept if better cardinality, or same cardinality but now coprime
                        if (new_card > current_card) or \
                           (new_card == current_card and new_coprime and not current_coprime) or \
                           (new_card == current_card and new_coprime and current_coprime):
                            if new_card > current_card or (new_coprime and not current_coprime):
                                current_A = new_A
                                current_sums = new_sums
                                current_card = new_card
                                current_coprime = new_coprime
                                improved = True
                                break
                if improved:
                    break
            if improved:
                continue
                
            # Try all possible replacements in B
            for idx in range(3):
                for new_val in all_vals:
                    if new_val in current_B:
                        continue
                    new_B = list(current_B)
                    new_B[idx] = new_val
                    new_sums = get_distinct_sums(current_A, new_B)
                    new_card = len(new_sums)
                    
                    if new_card > current_card or (new_card == current_card and new_card >= 5):
                        new_coprime = check_pairwise_coprime_fast(list(new_sums))
                        if (new_card > current_card) or \
                           (new_card == current_card and new_coprime and not current_coprime) or \
                           (new_card == current_card and new_coprime and current_coprime):
                            if new_card > current_card or (new_coprime and not current_coprime):
                                current_B = new_B
                                current_sums = new_sums
                                current_card = new_card
                                current_coprime = new_coprime
                                improved = True
                                break
                if improved:
                    break
                    
        return current_A, current_B, current_card, current_coprime

    # Phase 1: Random sampling with local search
    random.seed(42)
    for iteration in range(500):
        if time.time() >= deadline:
            break
            
        if random.random() < 0.5:
            A = random.sample(odds, 3)
            B = random.sample(evens, 3)
        else:
            A = random.sample(evens, 3)
            B = random.sample(odds, 3)
        
        # Quick check
        sums = get_distinct_sums(A, B)
        card = len(sums)
        
        if card < best_cardinality and card < 5:
            continue
        
        # Run local search
        new_A, new_B, new_card, new_coprime = try_improve(A, B)
        
        if new_card > best_cardinality or (new_card == best_cardinality and new_coprime and not best_is_coprime):
            best_A = new_A
            best_B = new_B
            best_cardinality = new_card
            best_is_coprime = new_coprime
            
            if new_card == 9 and new_coprime:
                return [best_A, best_B]
            
            if new_card >= 7 and new_coprime:
                # Try to push further
                new_A2, new_B2, new_card2, new_coprime2 = try_improve(new_A, new_B)
                if new_card2 > best_cardinality or (new_card2 == best_cardinality and new_coprime2):
                    best_A = new_A2
                    best_B = new_B2
                    best_cardinality = new_card2
                    best_is_coprime = new_coprime2
                    if new_card2 == 9 and new_coprime2:
                        return [best_A, best_B]

    # Phase 2: Systematic search around best found
    if time.time() < deadline:
        # Try all pairs of (odd, even) or (even, odd) with specific patterns
        for a1 in odds[:20]:
            if time.time() >= deadline:
                break
            for a2 in odds[:20]:
                if a2 == a1:
                    continue
                for a3 in odds[:20]:
                    if a3 == a1 or a3 == a2:
                        continue
                    A = [a1, a2, a3]
                    for b1 in evens[:20]:
                        if time.time() >= deadline:
                            break
                        for b2 in evens[:20]:
                            if b2 == b1:
                                continue
                            for b3 in evens[:20]:
                                if b3 == b1 or b3 == b2:
                                    continue
                                B = [b1, b2, b3]
                                sums = get_distinct_sums(A, B)
                                card = len(sums)
                                if card > best_cardinality or (card == 9 and check_pairwise_coprime_fast(list(sums))):
                                    if card == 9 and check_pairwise_coprime_fast(list(sums)):
                                        return [A, B]
                                    if card > best_cardinality:
                                        best_A = A
                                        best_B = B
                                        best_cardinality = card
                                        best_is_coprime = check_pairwise_coprime_fast(list(sums))
                                        if best_cardinality == 9 and best_is_coprime:
                                            return [best_A, best_B]

    return [best_A, best_B]
