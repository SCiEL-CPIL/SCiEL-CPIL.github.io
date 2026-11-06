import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Helper to calculate score efficiently
    def get_score(S):
        S_set = S
        S_sorted = sorted(S)
        d_set = set()
        n = len(S_sorted)
        contains = S_set.__contains__
        for i in range(n):
            a = S_sorted[i]
            for j in range(i + 1, n):
                b = S_sorted[j]
                d = b - a
                c = b + d
                if contains(c):
                    d_set.add(d)
        return len(d_set)
    
    # Helper to get the set of differences for a set
    def get_diffs(S):
        S_set = S
        S_sorted = sorted(S)
        d_set = set()
        n = len(S_sorted)
        contains = S_set.__contains__
        for i in range(n):
            a = S_sorted[i]
            for j in range(i + 1, n):
                b = S_sorted[j]
                d = b - a
                c = b + d
                if contains(c):
                    d_set.add(d)
        return d_set

    # Strategy 1: Greedy Construction with Difference-Aware Lookahead
    # We build the set element by element. At each step, we pick the candidate
    # that maximizes the number of NEW distinct differences formed.
    def greedy_construct():
        S = set()
        # Start with a few random seeds to break symmetry
        # Or start with 1, 2, 4... powers of 2? Or just random?
        # Let's try starting with a small random set
        for _ in range(3):
            S.add(random.randint(1, 512))
        
        current_diffs = get_diffs(S)
        
        while len(S) < 32:
            best_candidate = None
            best_new_diffs = -1
            best_total = -1
            
            # Sample a subset of candidates to keep it fast, or iterate all?
            # 512 is small. Let's iterate all valid candidates.
            candidates = [x for x in range(1, 513) if x not in S]
            
            # To speed up, we can sample if it's too slow, but 512*30 steps is ~15k checks.
            # Each check involves get_diffs which is O(N^2). 32^2 = 1024.
            # 15k * 1024 = 15M ops. Might be slow in Python.
            # Let's sample 50 candidates per step to be safe.
            sample_size = min(50, len(candidates))
            sampled = random.sample(candidates, sample_size)
            
            for x in sampled:
                new_S = S | {x}
                # We can optimize diff calculation by only checking new APs involving x
                # But for simplicity and correctness first, let's use a faster incremental check
                # Actually, let's just calculate new diffs.
                
                # Incremental diff calculation:
                # New diffs are those d where (x, x+d, x+2d) or (x-2d, x-d, x) or (x-d, x, x+d) is in S
                # This is complex. Let's just use get_diffs for the candidate if we keep N small.
                # Or better: compute the set of diffs for new_S quickly?
                # Given the time limit, let's try a slightly smarter heuristic:
                # Count how many existing elements form an AP with x.
                
                # For a candidate x, it forms a 3-AP with two existing elements a, b if:
                # 1. x is the middle: a, x, b -> b = 2x - a
                # 2. x is the end: a, b, x -> x = 2b - a => b = (x+a)/2 (must be int)
                # 3. x is the start: x, a, b -> b = 2a - x
                
                # This is getting complicated to track "new" diffs specifically.
                # Let's stick to a simpler heuristic: 
                # Score = number of 3-APs containing x.
                # This correlates well with adding new diffs.
                
                ap_count = 0
                for a in S:
                    # x is middle
                    b = 2 * x - a
                    if b in S and b != a and b != x:
                        ap_count += 1
                    # x is end (a, b, x) => b = (x+a)/2
                    if (x + a) % 2 == 0:
                        b = (x + a) // 2
                        if b in S and b != a and b != x:
                            ap_count += 1
                    # x is start (x, a, b) => b = 2a - x
                    b = 2 * a - x
                    if b in S and b != a and b != x:
                        ap_count += 1
                        
                # Tie-breaker: random or larger value?
                # We want to maximize distinct diffs, not just AP count.
                # But AP count is a good proxy.
                
                if ap_count > best_new_diffs:
                    best_new_diffs = ap_count
                    best_candidate = x
                    best_total = ap_count
                elif ap_count == best_new_diffs:
                    # If tie, maybe prefer the one that adds a "new" diff?
                    # Hard to check without full diff set.
                    # Let's just pick randomly among top candidates.
                    if random.random() < 0.5:
                        best_candidate = x
                        best_total = ap_count

            if best_candidate is not None:
                S.add(best_candidate)
            else:
                # Fallback: add random
                S.add(random.choice(candidates))
                
            # Periodically update current_diffs? Not needed for this heuristic.
            
        return S

    # Strategy 2: Local Search with Difference-Preserving Swaps
    # Start with a good set (from greedy or random).
    # Move: Replace one element with another.
    # Accept if:
    # 1. Score increases (new diffs added)
    # 2. Score stays same (but we might accept to escape local minima? No, stick to strict improvement or slight noise)
    # 3. Score decreases but we are in "exploration" mode (Simulated Annealing style)
    
    # To make it "difference-aware", we can bias the acceptance probability
    # or the move generation.
    # Move generation: Instead of random replacement, try replacing an element
    # with one that "completes" an AP with two other elements in the set.
    
    def local_search(initial_set, time_limit):
        S = set(initial_set)
        current_score = get_score(S)
        best_S = set(S)
        best_score = current_score
        
        # Precompute neighbors? No, dynamic.
        
        end_time = time.time() + time_limit
        temp = 5.0
        cooling = 0.99
        
        while time.time() < end_time:
            # Generate move
            # 50% chance: Random swap
            # 50% chance: Targeted swap (complete an AP)
            
            if random.random() < 0.5:
                # Random swap
                elem_to_replace = random.choice(list(S))
                new_val = random.randint(1, 512)
                if new_val in S:
                    continue
            else:
                # Targeted swap: Find an element x in S such that there exist a, b in S
                # where a, x, b is NOT an AP, but if we move x to x', it might form one?
                # Actually, let's find a "broken" AP.
                # Pick two elements a, b in S. The midpoint m = (a+b)/2.
                # If m is not in S, try adding m (by removing some other element).
                # Which element to remove? The one that contributes least to the score?
                # Or just random?
                
                # Let's pick a random pair a, b from S
                S_list = list(S)
                if len(S_list) < 2:
                    continue
                a, b = random.sample(S_list, 2)
                if (a + b) % 2 != 0:
                    continue
                m = (a + b) // 2
                if m in S or m < 1 or m > 512:
                    continue
                
                # m is a good candidate to add.
                # We must remove one element to keep size 32.
                # Remove a random element (not a or b)
                candidates_to_remove = [x for x in S if x != a and x != b]
                if not candidates_to_remove:
                    continue
                elem_to_remove = random.choice(candidates_to_remove)
                
                new_S = set(S)
                new_S.remove(elem_to_remove)
                new_S.add(m)
                
                # Evaluate
                new_score = get_score(new_S)
                delta = new_score - current_score
                
                if delta > 0:
                    S = new_S
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_S = set(S)
                else:
                    # Simulated Annealing acceptance
                    if temp > 0.01:
                        prob = math.exp(delta / temp)
                        if random.random() < prob:
                            S = new_S
                            current_score = new_score
                            
            # Cooling
            temp *= cooling
            if temp < 0.01:
                temp = 0.01

        return best_S, best_score

    # Main Loop
    best_overall_score = -1
    best_overall_set = None
    
    # 1. Try Greedy Construction multiple times
    greedy_time_budget = 30.0
    greedy_deadline = start_time + greedy_time_budget
    
    while time.time() < greedy_deadline:
        S_greedy = greedy_construct()
        score = get_score(S_greedy)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_set = S_greedy
        # Early exit if we hit a very high score?
        # Max possible diffs?
        # With 32 elements, max pairs is 496. Max diffs is bounded by range 511.
        # If we get > 100, it's likely good.
        if score > 150:
            break

    # 2. Use the best greedy set as seed for Local Search
    # Also try a few random starts for Local Search
    
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        # Split remaining time among a few local search runs
        num_ls_runs = 3
        time_per_ls = (remaining_time - 5) / num_ls_runs
        
        seeds = []
        if best_overall_set:
            seeds.append(best_overall_set)
        
        # Add a couple of random seeds
        for _ in range(2):
            seeds.append(set(random.sample(range(1, 513), 32)))
            
        for seed in seeds:
            if time.time() >= deadline - 2.0:
                break
            ls_set, ls_score = local_search(seed, time_per_ls)
            if ls_score > best_overall_score:
                best_overall_score = ls_score
                best_overall_set = ls_set

    if best_overall_set is None:
        # Fallback
        return sorted(random.sample(range(1, 513), 32))
        
    return sorted(best_overall_set)

# For debugging/testing if run directly
if __name__ == "__main__":
    res = construct()
    print(f"Result size: {len(res)}")
    print(f"Sample: {res[:10]}")
    # Calculate score
    S = set(res)
    d_set = set()
    S_sorted = sorted(res)
    for i in range(len(S_sorted)):
        a = S_sorted[i]
        for j in range(i+1, len(S_sorted)):
            b = S_sorted[j]
            d = b - a
            c = b + d
            if c in S:
                d_set.add(d)
    print(f"Score: {len(d_set)}")
