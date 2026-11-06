"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    time_limit = 55.0  # Leave some buffer for 60s limit
    
    # The task: find up to 80 distinct positive integers <= 4096
    # such that every unordered triple (allowing repetitions) has a unique sum.
    # This is a B_3[1] sequence (Sidon-like for triples).
    
    # Strategy: Greedy with multiple orderings and randomization to find larger sets.
    # The key insight is that the constraint is on triple sums.
    # For a set S, the number of triples is C(|S|+2, 3) = |S|*(|S|+1)*(|S|+2)/6.
    # For |S|=80, that's 80*81*82/6 = 88560 triples.
    # The sums range from 3*1=3 to 3*4096=12288.
    # So we need 88560 distinct sums in [3, 12288], which has 12286 values.
    # 88560 > 12286, so it's impossible to have 80 elements!
    # Wait, let me recalculate. The sums are a+b+c where a<=b<=c and a,b,c in S.
    # The minimum sum is 3*min(S) and maximum is 3*max(S).
    # For S subset of [1,4096], sums are in [3, 12288].
    # Number of possible sums is 12288-3+1 = 12286.
    # For |S|=n, number of triples is n(n+1)(n+2)/6.
    # We need n(n+1)(n+2)/6 <= 12286.
    # For n=40: 40*41*42/6 = 11480. OK.
    # For n=42: 42*43*44/6 = 13364 > 12286.
    # So the maximum possible is 40 or 41.
    # Let me check n=41: 41*42*43/6 = 12411 > 12286.
    # So max is 40.
    
    # Actually wait, the sums don't need to cover all values in [3,12288], they just need to be distinct.
    # But there are only 12286 possible distinct sum values.
    # So we need at most 12286 triples, meaning n(n+1)(n+2)/6 <= 12286.
    # n=40: 11480 <= 12286. OK.
    # n=41: 12411 > 12286. Not OK.
    # So the theoretical maximum is 40.
    
    # Let's try to find a set of size 40.
    # The seed program found a set by greedy from 1 upward.
    # Let's see how large that set is.
    
    def run_greedy(order, time_budget):
        selected = []
        current_sums = set()
        
        for x in order:
            if time.time() - start_time > time_budget:
                break
            # Check if adding x creates any duplicate triple sum
            # New triples involving x: (x,x,x), (x,x,y), (x,y,y), (x,y,z) for y,z in selected
            # where y <= z and x <= y for the ones with x as smallest
            # Actually, since we're adding x to an existing set, we need to check all triples
            # that include at least one x.
            
            new_sums = set()
            valid = True
            
            # (x, x, x)
            s = 3 * x
            if s in current_sums:
                valid = False
            else:
                new_sums.add(s)
            
            if valid:
                # (x, x, y) for y in selected, y != x (but x not in selected yet)
                for y in selected:
                    s = 2 * x + y
                    if s in current_sums or s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
            
            if valid:
                # (x, y, y) for y in selected
                for y in selected:
                    s = x + 2 * y
                    if s in current_sums or s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
            
            if valid:
                # (x, y, z) for y, z in selected, y <= z, y != z
                for i in range(len(selected)):
                    for j in range(i + 1, len(selected)):
                        y, z = selected[i], selected[j]
                        s = x + y + z
                        if s in current_sums or s in new_sums:
                            valid = False
                            break
                        new_sums.add(s)
                    if not valid:
                        break
            
            if valid:
                selected.append(x)
                current_sums.update(new_sums)
        
        return selected
    
    # Try multiple strategies
    best_set = []
    
    # Strategy 1: Greedy from 1 upward (original)
    t1 = time.time()
    s1 = run_greedy(range(1, 4097), time_limit - (time.time() - start_time))
    if len(s1) > len(best_set):
        best_set = s1
    
    # Strategy 2: Greedy from larger numbers downward
    if time.time() - start_time < time_limit - 5:
        s2 = run_greedy(range(4096, 0, -1), time_limit - (time.time() - start_time))
        if len(s2) > len(best_set):
            best_set = s2
    
    # Strategy 3: Random orderings
    random.seed(42)
    for trial in range(50):
        if time.time() - start_time > time_limit - 2:
            break
        order = list(range(1, 4097))
        random.shuffle(order)
        s = run_greedy(order, time_limit - (time.time() - start_time))
        if len(s) > len(best_set):
            best_set = s
    
    # Strategy 4: Biased ordering - favor numbers that are far apart
    if time.time() - start_time < time_limit - 5:
        # Try powers of 2 or similar sparse sequences
        sparse = [1]
        for i in range(2, 4097):
            if i % 100 == 0:
                sparse.append(i)
        s = run_greedy(sparse, time_limit - (time.time() - start_time))
        if len(s) > len(best_set):
            best_set = s
    
    # Strategy 5: Try to extend the best set found so far
    if time.time() - start_time < time_limit - 5:
        base = best_set[:]
        # Try adding candidates not in base
        remaining = [x for x in range(1, 4097) if x not in set(base)]
        extended = run_greedy(remaining, time_limit - (time.time() - start_time))
        if len(extended) > len(base):
            best_set = base + extended
    
    # Verify the result
    def verify(s):
        if len(s) != len(set(s)):
            return False
        if any(x < 1 or x > 4096 for x in s):
            return False
        sums = [sum(t) for t in combinations_with_replacement(s, 3)]
        return len(sums) == len(set(sums))
    
    if not verify(best_set):
        # Fallback to a simple valid set
        best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    
    return best_set
# EVOLVE-BLOCK-END
