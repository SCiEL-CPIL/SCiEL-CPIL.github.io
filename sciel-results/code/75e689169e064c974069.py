import time
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # 1. Generate 2-3-5 smooth numbers up to 128
    smooth_numbers = []
    for i in range(1, 129):
        y = i
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth_numbers.append(i)
    
    # Map sum to best (min length) valid antichain tuple
    # We want to minimize total summands, so for each target we pick the min-length rep.
    # We iterate by size to ensure we find the minimum possible size for each target.
    
    # valid_reps[t] will store the best tuple found for target t so far.
    # We update it only if we find a shorter tuple.
    # Since we iterate size 1, then 2, etc., the first one found is the best for that size.
    # But a size 2 rep is always better than size 3.
    # So we can just check if target is unsolved, or if we haven't found a size 1/2 rep yet.
    
    # Actually, simplest: 
    # 1. Find all size 1 reps.
    # 2. Find all size 2 reps.
    # 3. Find all size 3 reps.
    # 4. For each target, pick the one with min length.
    
    # But to be efficient, we can stop searching for a specific target once we have a size 1 rep.
    # And stop searching for size 2 if we have size 1.
    
    # Let's use a dictionary: best_rep[t] = tuple
    best_rep = {}
    
    # Size 1
    for a in smooth_numbers:
        if a <= 128:
            best_rep[a] = (a,)
    
    # Size 2
    # Only consider targets not yet solved by size 1? No, size 2 is worse than size 1, so we only fill gaps.
    # But wait, a target might have a size 1 rep AND a size 2 rep. We prefer size 1.
    # So we only need to find size 2 reps for targets NOT in best_rep.
    
    targets_needing_2 = [t for t in range(1, 129) if t not in best_rep]
    
    # Optimization: Precompute valid pairs?
    # There are ~128 smooth numbers. Pairs are ~8000.
    # Let's iterate pairs and check if sum is in targets_needing_2.
    
    # To speed up, we can use a set for targets_needing_2.
    targets_set_2 = set(targets_needing_2)
    
    # We iterate pairs (a, b) with a < b.
    # If a+b is in targets_set_2, and (a,b) is an antichain, we store it.
    # We should store the FIRST one found? Or any? They all have length 2.
    # So any valid pair is fine.
    
    for i in range(len(smooth_numbers)):
        a = smooth_numbers[i]
        for j in range(i + 1, len(smooth_numbers)):
            b = smooth_numbers[j]
            s = a + b
            if s > 128:
                break
            if s in targets_set_2:
                # Check antichain
                if a % b != 0 and b % a != 0:
                    best_rep[s] = (a, b)
                    targets_set_2.remove(s) # No longer needed
                    if not targets_set_2:
                        break
        if not targets_set_2:
            break
    
    # Size 3
    targets_needing_3 = [t for t in range(1, 129) if t not in best_rep]
    if targets_needing_3:
        targets_set_3 = set(targets_needing_3)
        
        # Iterate triples (a, b, c) with a < b < c.
        # This is O(N^3) which is ~2M iterations. Should be fast enough in Python if optimized.
        # But we can prune.
        
        for i in range(len(smooth_numbers)):
            a = smooth_numbers[i]
            # Prune: if a + next_two_smallest > 128, break?
            # Smallest two distinct smooth numbers > a.
            # Just check a + smooth_numbers[i+1] + smooth_numbers[i+2] > 128?
            if i + 2 >= len(smooth_numbers):
                break
            min_sum_i = a + smooth_numbers[i+1] + smooth_numbers[i+2]
            if min_sum_i > 128:
                break
                
            for j in range(i + 1, len(smooth_numbers)):
                b = smooth_numbers[j]
                # Prune
                if j + 1 >= len(smooth_numbers):
                    break
                min_sum_j = a + b + smooth_numbers[j+1]
                if min_sum_j > 128:
                    break
                    
                for k in range(j + 1, len(smooth_numbers)):
                    c = smooth_numbers[k]
                    s = a + b + c
                    if s > 128:
                        break
                    
                    if s in targets_set_3:
                        # Check antichain
                        # a,b are distinct. a,c distinct. b,c distinct.
                        # Check divisibility.
                        if (a % b != 0 and b % a != 0 and
                            a % c != 0 and c % a != 0 and
                            b % c != 0 and c % b != 0):
                            best_rep[s] = (a, b, c)
                            targets_set_3.remove(s)
                            if not targets_set_3:
                                break
                if not targets_set_3:
                    break
            if not targets_set_3:
                break
        
        if time.time() > deadline:
             # Return best found so far
             # But we need to ensure all targets are covered.
             # If not, we must continue.
             pass

    # If any targets are still missing, we need size 4+
    # This is unlikely, but let's handle it to be safe.
    missing = [t for t in range(1, 129) if t not in best_rep]
    
    if missing:
        # Size 4
        targets_set_4 = set(missing)
        
        for i in range(len(smooth_numbers)):
            a = smooth_numbers[i]
            if i + 3 >= len(smooth_numbers): break
            min_sum_i = a + smooth_numbers[i+1] + smooth_numbers[i+2] + smooth_numbers[i+3]
            if min_sum_i > 128: break
            
            for j in range(i + 1, len(smooth_numbers)):
                b = smooth_numbers[j]
                if j + 2 >= len(smooth_numbers): break
                min_sum_j = a + b + smooth_numbers[j+1] + smooth_numbers[j+2]
                if min_sum_j > 128: break
                
                for k in range(j + 1, len(smooth_numbers)):
                    c = smooth_numbers[k]
                    if k + 1 >= len(smooth_numbers): break
                    min_sum_k = a + b + c + smooth_numbers[k+1]
                    if min_sum_k > 128: break
                    
                    for l in range(k + 1, len(smooth_numbers)):
                        d = smooth_numbers[l]
                        s = a + b + c + d
                        if s > 128: break
                        
                        if s in targets_set_4:
                            # Check antichain for 4 elements
                            # (a,b,c,d)
                            if (a % b != 0 and b % a != 0 and
                                a % c != 0 and c % a != 0 and
                                a % d != 0 and d % a != 0 and
                                b % c != 0 and c % b != 0 and
                                b % d != 0 and d % b != 0 and
                                c % d != 0 and d % c != 0):
                                best_rep[s] = (a, b, c, d)
                                targets_set_4.remove(s)
                                if not targets_set_4:
                                    break
                    if not targets_set_4: break
                if not targets_set_4: break
            if not targets_set_4: break
            
        if time.time() > deadline:
            # Check if all covered
            missing = [t for t in range(1, 129) if t not in best_rep]
            if not missing:
                # Construct result
                answers = [None] * 128
                for t in range(1, 129):
                    answers[t-1] = list(best_rep[t])
                return answers
            # If not covered, we are in trouble. But task says valid reps exist.
            # We will try to return what we have, or raise.
            # Given the constraints, size 4 should cover all.

    # Final check and construction
    # If any are still missing (shouldn't happen), we might need size 5.
    # But let's assume size 4 is enough.
    # To be safe, if missing, do a quick DFS for just those missing targets.
    missing = [t for t in range(1, 129) if t not in best_rep]
    
    if missing:
        for t in missing:
            # DFS for size 5, 6...
            found = False
            for size in range(5, 10):
                # This is expensive, but only for missing targets.
                # We can use itertools.combinations
                for combo in itertools.combinations(smooth_numbers, size):
                    if sum(combo) == t:
                        # Check antichain
                        is_ac = True
                        for i in range(size):
                            for j in range(i+1, size):
                                if combo[i] % combo[j] == 0 or combo[j] % combo[i] == 0:
                                    is_ac = False
                                    break
                            if not is_ac: break
                        if is_ac:
                            best_rep[t] = combo
                            found = True
                            break
                if found:
                    break
            if time.time() > deadline:
                break

    # Construct final answer
    answers = [None] * 128
    for t in range(1, 129):
        if t in best_rep:
            answers[t-1] = list(best_rep[t])
        else:
            # Critical failure
            # Try a greedy fallback? No, must be valid.
            # The task guarantees existence.
            # We return empty list? No, must be 1..30 terms.
            # We'll just leave it as None, which might fail validity.
            # But this should not happen.
            pass

    return answers

# Test locally?
# if __name__ == "__main__":
#     res = construct()
#     print(len(res))
#     total_terms = sum(len(r) for r in res)
#     print(total_terms)
