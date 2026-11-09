import time
import random
from itertools import combinations

def construct():
    """
    Finds a set of distinct integers in 1..256 maximizing the size,
    such that all non-empty subset sums are distinct and no two distinct
    subset sums divide one another.
    """
    start_time = time.time()
    # The evaluator kills after 90s, but the task contract says 60s.
    # We use a 55s limit to be safe against both interpretations and overhead.
    time_limit = 55.0
    
    best_set = [31, 32, 34]
    best_size = len(best_set)
    
    # Precompute divisors for 1..512 (max sum is 256*2, but realistically much lower for valid sets)
    # Actually max sum for 8 elements is 256*8 = 2048. Let's go up to 2048.
    MAX_SUM = 2048
    divisors = [[] for _ in range(MAX_SUM + 1)]
    for i in range(1, MAX_SUM + 1):
        for j in range(i, MAX_SUM + 1, i):
            divisors[j].append(i)
    
    def is_antichain(sums_set):
        """
        Check if sums_set (excluding 0) is an antichain under divisibility.
        Optimized using precomputed divisors.
        """
        if not sums_set or len(sums_set) <= 1:
            return True
        
        # If 1 is in sums_set, it divides everything
        if 1 in sums_set and len(sums_set) > 1:
            return False
            
        # For each u in sums_set, check if any proper divisor of u is in sums_set
        # Or any multiple. Checking divisors is often faster if we have the list.
        # A set is an antichain iff for all u, no proper divisor of u is in the set.
        for u in sums_set:
            for d in divisors[u]:
                if d != u and d in sums_set:
                    return False
        return True

    def check_validity_incremental(S, x, sums_S):
        """
        Checks if S U {x} is valid, given S is valid.
        Returns (True, new_sums) if valid, (False, None) otherwise.
        """
        if x in S:
            return False, None
            
        # Generate new sums: x + s for s in sums_S
        # sums_S contains 0.
        new_sums = {x + s for s in sums_S}
        
        # Check distinctness: sums_S and new_sums must be disjoint
        if sums_S & new_sums:
            return False, None
        
        all_sums = sums_S | new_sums
        non_empty_all = all_sums - {0}
        
        # Check divisibility antichain
        # Optimization: We know sums_S - {0} is an antichain.
        # We need to check:
        # 1. new_sums - {0} is an antichain internally.
        # 2. No u in (sums_S - {0}) divides v in (new_sums - {0}) and vice versa.
        
        # However, the generic is_antichain is fast enough for small sets (<= 2^10 = 1024).
        # Let's stick to the robust check for correctness first, then optimize if needed.
        # Given the size constraint (max ~10-12 elements), 2^12=4096 sums.
        # is_antichain iterates 4096 * avg_divisors(~10) = 40k ops. Very fast.
        
        if not is_antichain(non_empty_all):
            return False, None
        
        return True, all_sums

    def local_search_improve(S, sums_S, remaining_candidates, deadline):
        """
        Try to improve the current set by swapping or adding elements.
        """
        nonlocal best_set, best_size
        
        current_S = list(S)
        current_sums = set(sums_S)
        current_size = len(current_S)
        
        # Sort candidates by "safety" - prefer larger numbers (fewer divisors/multiples)
        sorted_candidates = sorted(remaining_candidates, reverse=True)
        
        # 1. Try to add one more element
        for x in sorted_candidates:
            if time.time() > deadline:
                break
            
            valid, new_sums = check_validity_incremental(current_S, x, current_sums)
            if valid:
                current_S.append(x)
                current_sums = new_sums
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_S[:]
                
                # Try to add another element
                for y in sorted_candidates:
                    if y == x or time.time() > deadline:
                        continue
                    valid2, new_sums2 = check_validity_incremental(current_S, y, current_sums)
                    if valid2:
                        current_S.append(y)
                        current_sums = new_sums2
                        current_size += 1
                        if current_size > best_size:
                            best_size = current_size
                            best_set = current_S[:]
                        
                        # Try to add a third?
                        for z in sorted_candidates:
                            if z == x or z == y or time.time() > deadline:
                                continue
                            valid3, new_sums3 = check_validity_incremental(current_S, z, current_sums)
                            if valid3:
                                current_S.append(z)
                                current_sums = new_sums3
                                current_size += 1
                                if current_size > best_size:
                                    best_size = current_size
                                    best_set = current_S[:]
                                current_S.pop()
                                current_sums = new_sums2
                            else:
                                continue
                        current_S.pop()
                        current_sums = current_sums - {y + s for s in new_sums2 if (y+s) not in new_sums2} # Just recompute to be safe
                        current_sums = compute_sums(current_S)
                        current_size = len(current_S)
        
        # 2. Try swapping: remove one element, try to add two
        # This is expensive, so limit it
        if time.time() < deadline:
            for i in range(len(current_S)):
                if time.time() > deadline:
                    break
                
                removed = current_S[i]
                S_without = current_S[:i] + current_S[i+1:]
                sums_without = compute_sums(S_without)
                
                # Try adding two elements
                for x in sorted_candidates:
                    if x == removed or time.time() > deadline:
                        continue
                    valid1, sums_with_x = check_validity_incremental(S_without, x, sums_without)
                    if not valid1:
                        continue
                    
                    for y in sorted_candidates:
                        if y == x or y == removed or time.time() > deadline:
                            continue
                        valid2, sums_with_xy = check_validity_incremental(S_without + [x], y, sums_with_x)
                        if valid2:
                            new_S = S_without + [x, y]
                            if len(new_S) > best_size:
                                best_size = len(new_S)
                                best_set = new_S[:]
        
        return current_S, current_sums

    def compute_sums(S):
        """Compute all subset sums of S (including 0)."""
        sums = {0}
        for x in S:
            new_sums = set()
            for s in sums:
                new_sums.add(s + x)
            sums = sums | new_sums
        return sums

    # Search Strategy
    random.seed(42)
    
    # Pre-compute a "superincreasing-like" preference order?
    # The inspiration suggests modeling as superincreasing.
    # A superincreasing sequence a1 < a2 < ... < ak where a_i > sum(a_1...a_{i-1})
    # guarantees distinct subset sums.
    # We can try to construct such sequences first, then relax.
    
    # Strategy 1: Greedy construction with various orderings
    # Strategy 2: Local search improvement
    
    num_restarts = 20000 # Increase restarts due to faster checks
    
    # Pre-compute candidates sorted by various heuristics
    # Heuristic 1: Descending (largest first)
    # Heuristic 2: Random
    # Heuristic 3: "Prime-like" (few divisors)
    # Heuristic 4: Superincreasing bias (start small, grow fast? No, start large?)
    
    for attempt in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        S = []
        sums_S = {0}
        
        candidates = list(range(1, 257))
        
        # Different strategies for different attempts
        if attempt % 6 == 0:
            # Descending (largest first) - often good for packing
            candidates.sort(reverse=True)
        elif attempt % 6 == 1:
            # Random shuffle
            random.shuffle(candidates)
        elif attempt % 6 == 2:
            # Bias towards larger numbers
            candidates.sort(key=lambda x: random.random() ** 0.5 * x, reverse=True)
        elif attempt % 6 == 3:
            # Start with medium-large, then random
            large = list(range(128, 257))
            small = list(range(1, 128))
            random.shuffle(large)
            random.shuffle(small)
            candidates = large + small
        elif attempt % 6 == 4:
            # Prefer numbers with few divisors (primes, prime powers)
            # Pre-compute divisor counts for 1..256
            def count_divisors(n):
                count = 0
                for i in range(1, int(n**0.5) + 1):
                    if n % i == 0:
                        count += 2
                        if i != n // i:
                            count -= 1
                return count
            # Cache this?
            candidates.sort(key=lambda x: (count_divisors(x), -x))
        else:
            # Superincreasing attempt
            # Sort by value, then try to pick x > current_sum
            # This is a specific greedy strategy
            candidates.sort()
            # We'll handle the superincreasing logic in the loop by checking a flag?
            # No, let's just use a different ordering that favors gaps.
            # Sort by (value, random)
            candidates.sort(key=lambda x: (x, random.random()))
        
        for x in candidates:
            if time.time() - start_time > time_limit:
                break
            
            valid, new_sums = check_validity_incremental(S, x, sums_S)
            if valid:
                S.append(x)
                sums_S = new_sums
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S[:]
        
        # After greedy, try local improvement
        # Only if we are close to best
        if len(S) >= best_size - 1:
            remaining = [c for c in range(1, 257) if c not in S]
            S, sums_S = local_search_improve(S, sums_S, remaining, start_time + time_limit)
    
    # Final verification
    if best_set:
        final_sums = compute_sums(best_set)
        non_empty_final = final_sums - {0}
        if len(non_empty_final) != (1 << len(best_set)) - 1:
            best_set = [31, 32, 34]
            best_size = 3
        elif not is_antichain(non_empty_final):
            best_set = [31, 32, 34]
            best_size = 3
    
    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(res)
