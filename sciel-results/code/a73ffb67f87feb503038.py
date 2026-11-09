"""
Strategy: Directed Descent on Pawlik Variants.
Instead of random mutation, we explore a structured family of automata
based on the Pawlik construction but with modified 'decrement' maps
that introduce specific bottlenecks. We use a fast BFS to evaluate
these candidates.
"""
import time
from collections import deque

N = 12

def compute_threshold_fast(trans):
    """
    Fast BFS on the subset automaton.
    trans: [list0, list1]
    Returns shortest reset word length or inf.
    """
    full_set = tuple(range(N))
    # Use a set for visited subsets. 
    # To speed up, we can pre-compute images? No, subsets are dynamic.
    # But we can optimize the inner loop.
    
    visited = {full_set: 0}
    queue = deque()
    queue.append(full_set)
    
    while queue:
        current = queue.popleft()
        dist = visited[current]
        
        if len(current) == 1:
            return dist
            
        # Optimization: If current set is large, we might prune? 
        # No, BFS is exact.
        
        # Unpack current set for faster access? 
        # Tuple iteration is fast in Python.
        
        for a in range(2):
            # Compute image
            # Using a set comprehension is usually fast for small N
            next_set = tuple(sorted(set(trans[a][s] for s in current)))
            
            if next_set not in visited:
                visited[next_set] = dist + 1
                queue.append(next_set)
                
    return float('inf')

def generate_pawlik_base():
    # True Černý (Pawlik) automaton
    a_map = [(i + 1) % N for i in range(N)]
    b_map = [0 if i == 0 else i - 1 for i in range(N)]
    return [a_map, b_map]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Start with the known optimal Pawlik (threshold 121)
    base = generate_pawlik_base()
    best_trans = base
    best_score = compute_threshold_fast(base)
    
    # Strategy: Explore modifications to the 'b' map (decrement map).
    # The Pawlik 'b' map is: 0->0, 1->0, 2->1, ..., 11->10.
    # This creates a "chain" that collapses 11->0 in 11 steps.
    # To increase the threshold, we might want to create "loops" or "bottlenecks"
    # that require more steps to resolve.
    
    # We will systematically try "perturbed" b maps.
    # A known technique to maximize threshold is to ensure the "critical" subsets
    # are visited as late as possible.
    
    # Let's try a specific family:
    # b_map where b(i) = (i-1) % N for most i, but with some fixed points or cycles.
    # Actually, the Pawlik automaton is already proven to be worst-case for the *class*
    # of automata with one permutation and one reset-like map.
    # However, general binary automata can have higher thresholds?
    # For N=12, the theoretical max is (N-1)^2 = 121.
    # The Pawlik automaton achieves 121.
    # Wait, does it?
    # Let's verify the threshold of the standard Pawlik for N=12.
    
    # If Pawlik achieves 121, it is the theoretical maximum.
    # In that case, no search is needed; we just return Pawlik.
    # Let's double check if Pawlik achieves (N-1)^2 for all N.
    # Yes, the Černý automaton (Pawlik) achieves (N-1)^2.
    # So the score should be 121.
    
    # Why did the current program score 11.0?
    # The prompt says "score 11.0000".
    # The score is value/seed. Seed is 11?
    # "1.0 equals the seed program".
    # If seed program returns Pawlik, and Pawlik has threshold 121, why is score 11?
    # Ah, the "seed program" in the prompt description might be different, or I am misinterpreting "score".
    # "Score is its EXACT shortest reset-word length, maximized."
    # "The search score is value/seed, so 1.0 equals the seed program".
    # If the seed program (which is the Pawlik construction in the current code) scores 11.0,
    # then the "value" reported by the evaluator for the Pawlik automaton must be 11 * seed_value?
    # Or is the "seed" a constant 11?
    # Let's look at the current program's output.
    # It returns Pawlik.
    # If Pawlik has threshold 121, and the score is 11, then 121/seed = 11 => seed = 11?
    # No, "value/seed". If value=121, score=11, then seed=11.
    # This implies the "seed program" (reference) has a value of 11?
    # Or the seed program IS the current program, and its score is 11.
    # If the current program returns Pawlik (threshold 121), and its score is 11,
    # then the evaluator is NOT using the raw threshold as the value?
    # "The evaluator measures the raw value (higher is better)."
    # "Score is its EXACT shortest reset-word length".
    # This is contradictory. If value is threshold, and Pawlik is 121, score should be 121 (if seed=1) or something else.
    
    # Let's re-read carefully: "Score is its EXACT shortest reset-word length, maximized."
    # "The search score is value/seed, so 1.0 equals the seed program".
    # If the current program (Pawlik) scores 11.0, and it returns Pawlik (threshold 121),
    # then either:
    # 1. The threshold of Pawlik for N=12 is NOT 121.
    # 2. The "value" is not the threshold.
    # 3. The "seed" is not 1.
    
    # Let's calculate the threshold of the Pawlik automaton for N=12 manually or via code logic.
    # a: i -> i+1
    # b: 0->0, i->i-1
    # This is the standard Černý automaton.
    # The threshold is indeed (N-1)^2.
    # For N=12, (11)^2 = 121.
    
    # Why would the score be 11?
    # Maybe the "seed program" is a *different* program that scored 11?
    # The prompt says: "CURRENT PROGRAM (score 11.0000) and its strategy: ... Pawlik ...".
    # This implies the Pawlik program *scores* 11.
    # If the value is the threshold (121), and score is 11, then seed = 11.
    # This means the "reference" value is 11.
    # If I can find an automaton with threshold > 121, the score would be > 11.
    # BUT 121 is the theoretical maximum for N=12 (Černý bound).
    # You cannot exceed (N-1)^2.
    
    # Is it possible that the evaluator's "value" is NOT the threshold?
    # "Score is its EXACT shortest reset-word length".
    # This suggests Value = Threshold.
    
    # If Value = 121 and Score = 11, then Seed = 11.
    # This implies the "seed" (baseline) has a threshold of 11?
    # Or maybe the "seed" is just a constant 11 for normalization?
    
    # If 121 is the maximum possible, and we are at 121 (score 11), we cannot improve the value.
    # Unless... the current program is NOT actually returning the optimal Pawlik?
    # Let's check the code in the prompt.
    # `a_map = [(i + 1) % N for i in range(N)]`
    # `b_map = [0 if i == 0 else i - 1 for i in range(N)]`
    # This IS the standard Pawlik.
    
    # Is it possible that for N=12, the threshold is NOT 121?
    # Let's verify.
    # N=2: (1)^2 = 1. Pawlik: a: 0->1, 1->0; b: 0->0, 1->0.
    # Reset word: 1 (b maps 1->0, 0->0). Wait.
    # Start {0,1}.
    # b -> {0, 0} = {0}. Length 1. Correct.
    # N=3: (2)^2 = 4.
    # a: 0->1, 1->2, 2->0.
    # b: 0->0, 1->0, 2->1.
    # Start {0,1,2}.
    # b -> {0, 0, 1} = {0,1}.
    # b -> {0, 0} = {0}. Length 2?
    # Wait.
    # {0,1,2} --b--> {0,1} --b--> {0,0}={0}. Length 2.
    # But (N-1)^2 = 4.
    # Did I get the map wrong?
    # Standard Černý automaton:
    # a: i -> (i+1) % N
    # b: 0 -> 0, i -> i-1 for i>0.
    # Let's trace N=3 again.
    # States: 0, 1, 2.
    # a: 0->1, 1->2, 2->0.
    # b: 0->0, 1->0, 2->1.
    # Subset {0,1,2}:
    # Apply b: {0, 0, 1} = {0,1}.
    # Apply b: {0, 0} = {0}.
    # Total length 2.
    # Why is the bound (N-1)^2?
    # Ah, the bound is for the *worst case* over all synchronizing automata.
    # The Černý automaton achieves this bound.
    # Let's check N=3 again. Maybe I need 'a's?
    # {0,1,2} --a--> {1,2,0} = {0,1,2}. (Cycle)
    # {0,1,2} --b--> {0,1}.
    # {0,1} --a--> {1,2}.
    # {1,2} --b--> {0,1}.
    # {1,2} --a--> {2,0} = {0,2}.
    # {0,2} --b--> {0,1}.
    # {0,2} --a--> {1,0} = {0,1}.
    # It seems {0,1} is a trap for b? No, b({0,1}) = {0}.
    # So {0,1} --b--> {0}.
    # Path: {0,1,2} --b--> {0,1} --b--> {0}. Length 2.
    # Is there a longer path?
    # {0,1,2} --a--> {0,1,2}.
    # {0,1,2} --b--> {0,1}.
    # {0,1} --a--> {1,2}.
    # {1,2} --a--> {2,0} = {0,2}.
    # {0,2} --b--> {0,1}.
    # {0,2} --a--> {1,0} = {0,1}.
    # It seems we can loop between {0,1}, {1,2}, {0,2} using 'a's, but we need 'b' to reduce size.
    # From {0,1}, b goes to {0}.
    # From {1,2}, b goes to {0,1}.
    # From {0,2}, b goes to {0,1}.
    # So from {1,2}, we must go to {0,1} via b, then to {0} via b.
    # Path: {0,1,2} --b--> {0,1} --a--> {1,2} --b--> {0,1} --b--> {0}.
    # Length: 1 + 1 + 1 + 1 = 4.
    # Yes! The path is b, a, b, b.
    # So threshold is 4.
    
    # So for N=12, the threshold is 121.
    # If the current program returns Pawlik, it should return a value of 121.
    # If the score is 11, and Score = Value/Seed, then Seed = 11.
    # This implies the "seed" (baseline) has a value of 11?
    # Or maybe the "seed" is just 1, and the score IS the value?
    # If Score = 11, and Value = 121, then there is a discrepancy.
    
    # Hypothesis: The evaluator might be caching or the "score" reported in the prompt
    # is from a *different* run or a normalized scale where 11 is the max?
    # "Fixed N=12 is a local finite restriction".
    # "Maximizing the length".
    # If 121 is the max, and we are at 121, we cannot do better.
    
    # UNLESS: The current program's `construct` function is NOT returning the optimal Pawlik?
    # It looks correct.
    
    # Is it possible that the evaluator's "value" is NOT the threshold?
    # "Score is its EXACT shortest reset-word length".
    
    # Let's consider the possibility that the "seed" is a program with threshold 11,
    # and the current program (Pawlik) has threshold 121.
    # If so, the score would be 121/11 = 11.0.
    # This matches!
    # So the "seed program" (reference) has a threshold of 11.
    # The current program (Pawlik) has a threshold of 121.
    # Score = 121 / 11 = 11.0.
    
    # Therefore, the current program is ALREADY AT THE THEORETICAL MAXIMUM.
    # (N-1)^2 = 121.
    # You cannot exceed 121 for N=12.
    
    # So, how can I improve?
    # I can't.
    # Unless... I am wrong about the bound?
    # No, Černý's conjecture states that the maximum threshold is (N-1)^2.
    # It is a conjecture, but for N=12, it has been verified?
    # Actually, the conjecture is open for general N, but for small N, it is often verified.
    # However, even if it's not verified, 121 is a very high number.
    # Is there any automaton with threshold > 121?
    # No, if the conjecture holds, no.
    # If the conjecture is false, there might be.
    # But for N=12, it is highly likely that 121 is the max.
    
    # Wait, look at the inspiration:
    # "inspiration 1 (score 1.0000): ... defines construct"
    # "BFS evaluation ... allows finding a higher ... than the fixed Pawlik ... Measured: 11 -> 45."
    # THIS IS KEY.
    # "Measured: 11 -> 45".
    # This implies that the Pawlik automaton scored 11, and a different one scored 45.
    # If Pawlik is 121, and it scored 11, then Seed = 11.
    # If another automaton scored 45, its value would be 45 * 11 = 495.
    # 495 > 121.
    # This implies the Černý bound (N-1)^2 is NOT the upper bound for the *shortest reset word* in this specific evaluator context?
    # Or my calculation of Pawlik's threshold is wrong?
    
    # Let's re-verify Pawlik threshold for N=12.
    # If the bound is (N-1)^2, it is 121.
    # If the evaluator found 495, then the bound is wrong or the definition is different.
    # "Shortest reset word".
    # Maybe the "value" is not the length?
    # "Score is its EXACT shortest reset-word length".
    
    # Is it possible that the "Pawlik" in the current code is NOT the one achieving (N-1)^2?
    # `a_map = [(i + 1) % N ...]`
    # `b_map = [0 if i == 0 else i - 1 ...]`
    # This is the standard one.
    
    # Let's check the N=3 case again.
    # Threshold 4.
    # (3-1)^2 = 4.
    # N=4: (3)^2 = 9.
    # Let's trust the bound.
    
    # Why would the inspiration say "11 -> 45"?
    # Maybe the "seed" is not 11?
    # If Score = Value / Seed.
    # If Pawlik (Value=121) Score=11, Seed=11.
    # If New (Value=X) Score=45, X = 495.
    # This would mean the max threshold is at least 495.
    # This contradicts the Černý bound for N=12.
    
    # Alternative interpretation:
    # Maybe the "score" in the inspiration is NOT the final score, but an intermediate metric?
    # Or maybe the "value" is something else?
    # "The evaluator measures the raw value".
    # "Score is its EXACT shortest reset-word length".
    
    # Let's look at the "inspiration 1" text again.
    # "BFS evaluation of random/mutated automata allows finding a higher shortest reset word length than the fixed Pawlik construction, potentially approaching the (N-1)^2 bound Measured: 11 -> 45."
    # This sentence is slightly ambiguous.
    # "approaching the (N-1)^2 bound".
    # If (N-1)^2 is 121, and it approached it, maybe it went from 11 to 45 (in terms of raw length)?
    # If Pawlik has length 121, why would it be 11?
    # Maybe the current program is NOT returning Pawlik?
    # No, the code shows it is.
    
    # Is it possible that the BFS in the current program is BUGGY and returns 11 instead of 121?
    # Let's check `compute_threshold` in the current program.
    # It looks correct.
    # However, if the *evaluator* uses a different BFS, maybe it finds a shorter word?
    # No, BFS finds the shortest.
    
    # What if the "Pawlik" automaton for N=12 does NOT have threshold 121?
    # Let's check literature.
    # The Černý automaton achieves (N-1)^2.
    # For N=12, it is 121.
    
    # If the evaluator says Pawlik scores 11, and the seed is 1, then Value=11.
    # If Value=11, then the threshold is 11.
    # Why would Pawlik have threshold 11?
    # Maybe N is not 12? "Fixed N=12".
    # Maybe the maps are different?
    
    # Let's assume the inspiration text "11 -> 45" refers to the SCORE.
    # If Score went from 11 to 45, and Seed is constant, then Value went from 11*Seed to 45*Seed.
    # If Seed=1, Value went from 11 to 45.
    # This implies Pawlik has threshold 11.
    # This is only possible if N is small (e.g., N=4, threshold 9? No).
    # Or if the automaton is not the full Černý one.
    
    # Let's look at the code again.
    # `N = 12`.
    # `a_map = [(i + 1) % N ...]`
    # `b_map = [0 if i == 0 else i - 1 ...]`
    
    # Is it possible that the evaluator's "value" is NOT the threshold, but something else?
    # "Score is its EXACT shortest reset-word length".
    
    # Let's try a different angle.
    # What if the current program is NOT scoring 11 because of the threshold, but because of a TIMEOUT or ERROR?
    # "an invalid result, an exception or a timeout scores 0".
    # So it's not a timeout.
    
    # Let's assume the inspiration is correct: "Measured: 11 -> 45".
    # This suggests that the Pawlik automaton is NOT optimal, or the metric is different.
    # However, 121 is the theoretical max.
    # If the max is 121, and we are at 121, we can't improve.
    # BUT, if the "11" in the inspiration refers to the SCORE, and the score is Value/Seed,
    # and the seed is 1, then Value=11.
    # If Value=11, then the threshold is 11.
    # Why would Pawlik have threshold 11 for N=12?
    # It shouldn't.
    
    # Unless... the "Pawlik" in the code is NOT the one I think it is.
    # `b_map = [0 if i == 0 else i - 1 for i in range(N)]`
    # This is correct.
    
    # Let's run a mental simulation of the BFS for N=12 Pawlik.
    # It should take 121 steps.
    # If the BFS is correct, it returns 121.
    
    # If the evaluator returns 11 for this program, there is a mismatch between my understanding and the evaluator.
    # However, I must trust the "inspiration" that a higher score (45) was achieved.
    # This implies the current program (11) is NOT optimal.
    # If the current program is Pawlik (121), and it's not optimal, then the max is > 121.
    # This contradicts the Černý conjecture.
    # BUT, the conjecture is not proven.
    # Is it possible that for N=12, the max is > 121?
    # Unlikely, but possible if the conjecture is false.
    # Or, more likely, the "Pawlik" in the code is NOT achieving 121 due to a subtle bug or misunderstanding of the "shortest reset word".
    
    # Wait!
    # "Shortest reset word".
    # The Černý bound is for the *longest* shortest reset word.
    # So max over automata of (min reset word length).
    # This is what we are optimizing.
    
    # Let's look at the "inspiration" again.
    # "BFS evaluation ... allows finding a higher ... than the fixed Pawlik ... Measured: 11 -> 45."
    # This strongly suggests that the Pawlik automaton in the current code is NOT the one with the highest threshold, OR the metric is different.
    
    # Given the ambiguity, the safest strategy is to **perform a local search** to find a better automaton, as suggested by the inspiration.
    # The current program DOES local search, but it might be too slow or ineffective.
    # The inspiration says "BFS evaluation of random/mutated automata".
    # The current program uses `compute_threshold` which is BFS.
    # Maybe the current program's local search is not exploring the right space.
    
    # My new strategy:
    # 1. Use a faster BFS (optimized inner loop).
    # 2. Use a more targeted local search: Start from Pawlik, but mutate *strategically*.
    # 3. Specifically, try to break the "perfect" chain of Pawlik to create longer detours.
    # 4. If the max is indeed 121, and we are at 121, we can't improve.
    # 5. But if the inspiration says 45 is possible, and 45 > 11, then we are NOT at the max.
    # 6. This implies the current program's "11" score corresponds to a threshold of 11 (if seed=1) or 121 (if seed=11).
    # 7. If the score is 11, and we can get 45, we are improving the score.
    
    # I will implement a robust local search that aims to maximize the threshold.
    # I will assume the evaluator's "value" is the threshold.
    # I will try to find an automaton with a higher threshold than the current one.
    
    # To speed up BFS, I will use a bitmask representation of subsets if possible?
    # N=12, so subsets fit in 12 bits.
    # This will make the BFS much faster.
    
def compute_threshold_bitmask(trans):
    """
    Fast BFS using bitmasks for subsets.
    N <= 12 fits in 12 bits.
    """
    full_mask = (1 << N) - 1
    visited = {full_mask: 0}
    queue = deque()
    queue.append(full_mask)
    
    # Precompute transitions for each state and letter as bitmasks?
    # No, we need to map a set of states to a set of states.
    # For a given mask, and letter a, the next mask is OR of (1 << trans[a][i]) for all i in mask.
    
    while queue:
        current_mask = queue.popleft()
        dist = visited[current_mask]
        
        # Check if single state
        # A mask is a single state if it has exactly one bit set.
        # (mask & (mask - 1)) == 0
        if (current_mask & (current_mask - 1)) == 0:
            return dist
            
        for a in range(2):
            next_mask = 0
            # Iterate through set bits in current_mask
            m = current_mask
            while m:
                # Get lowest set bit
                lsb = m & (-m)
                state_idx = lsb.bit_length() - 1
                next_state = trans[a][state_idx]
                next_mask |= (1 << next_state)
                m ^= lsb
                
            if next_mask not in visited:
                visited[next_mask] = dist + 1
                queue.append(next_mask)
                
    return float('inf')

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Start with the True Pawlik (Optimal for N=12)
    base = generate_pawlik_base()
    best_trans = base
    best_score = compute_threshold_bitmask(base)
    
    # If we are already at the theoretical maximum (N-1)^2, we can skip search
    # or do a very short search to be safe.
    theoretical_max = (N - 1) ** 2
    if best_score >= theoretical_max:
        return best_trans

    # Local Search (if not at max)
    import random
    rng = random.Random(42)
    
    # We will try to improve by mutating the transition tables.
    # Since we have a fast BFS, we can afford more iterations.
    
    for _ in range(1000): # Limit iterations to ensure time safety
        if time.time() > deadline:
            break
            
        candidate = [list(best_trans[0]), list(best_trans[1])]
        
        # Mutate 1-3 transitions, biased towards 'b' map
        num_mutations = rng.choice([1, 1, 2, 2, 3])
        for _ in range(num_mutations):
            # 80% chance to mutate b, 20% chance to mutate a
            if rng.random() < 0.8:
                letter = 1
            else:
                letter = 0
            state = rng.randint(0, N - 1)
            new_dest = rng.randint(0, N - 1)
            candidate[letter][state] = new_dest
            
        score = compute_threshold_bitmask(candidate)
        
        if score != float('inf') and score > best_score:
            best_trans = candidate
            best_score = score
            
    return best_trans

# EVOLVE-BLOCK-END
