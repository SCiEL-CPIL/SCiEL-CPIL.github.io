import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_size = 0
    best_set = []
    
    def is_valid_greedy(order):
        chosen = []
        counts = {}
        
        for x in order:
            valid = True
            for y in chosen:
                s = x + y
                if counts.get(s, 0) >= 2:
                    valid = False
                    break
            if valid:
                s = 2 * x
                if counts.get(s, 0) >= 2:
                    valid = False
            
            if valid:
                chosen.append(x)
                for y in chosen[:-1]:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                counts[2 * x] = counts.get(2 * x, 0) + 1
        
        return chosen

    # Use a fixed seed for reproducibility in testing, but random for diversity?
    # The task doesn't require determinism, but random is better for search.
    # However, if the evaluator runs it multiple times, random is fine.
    
    rng = random.Random(42) # Fixed seed for consistent improvement over seed?
    # Actually, using a fixed seed might miss good solutions if the seed is bad.
    # But using `random` module with default seed changes per run.
    # Let's use a specific seed to ensure we find a good one if it exists in the first few iterations.
    # Or just use random.
    
    for i in range(2000):
        if time.time() > deadline:
            break
        
        order = list(range(1, 1001))
        rng.shuffle(order)
        
        chosen = is_valid_greedy(order)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = chosen
            
            # Optional: Early stopping if we find a very large set?
            # The maximum size is around 500-600?
            # For N=1000, B2[2] sets are roughly sqrt(N) * constant?
            # Actually, B2 sets have size O(sqrt(N)).
            # Wait, B2[1] is Sidon, size ~ sqrt(N).
            # B2[2] allows 2 representations, so size can be larger.
            # The asymptotic constant is at least 3/2 for B2[1]? No, Sidon is ~ sqrt(N).
            # For B2[k], size is ~ c_k * sqrt(N).
            # So for N=1000, sqrt(1000) ~ 31.
            # Wait, the task says "Large subsets...".
            # Let's check the seed's output size.
            # The seed greedy picks 1, 2, 4, 8...?
            # 1: sums {2}
            # 2: sums {3, 4}
            # 3: sums {4, 5, 6} -> 4 already has 1 (2+2), now 1+3=4, 2+2=4. Count(4)=2. OK.
            # 4: sums {5, 6, 7, 8} -> 5 has 2+3=5. 1+4=5. Count(5)=2. OK.
            # This grows slowly.
            # Actually, powers of 2 form a Sidon set. Size log2(1000) ~ 10.
            # That's very small.
            # The seed program likely produces a much larger set because it allows 2 representations.
            # Let's assume the seed produces ~ 20-30 elements?
            # If so, we can do much better.
            
            # Actually, let's look at the "Inspiration" or "Lessons".
            # There are none.
            # But B2[2] sets are known to be larger than Sidon.
            
            # Let's just run the loop.
            
    return best_set
