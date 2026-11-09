import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    best_seq = []
    best_len = 0
    # We will use a backtracking search.
    # State: current sequence, set of current products.
    # To make it efficient, we need to:
    # 1. Iterate through candidates in increasing order (to maintain strictly increasing property).
    # 2. For each candidate, check if it can be appended.
    # 3. If yes, recurse.
    # 4. If no, skip.
    # Pruning: If the number of remaining candidates is small, we might still extend.
    # Heuristic: Prioritize candidates that don't create "large" products early?
    # Actually, since we must be strictly increasing, the order of selection is fixed by the values.
    # We can't choose 5 before 3 if we want 3,4,5.
    # Wait, the sequence must be strictly increasing. So if we pick 2, then 4, we cannot pick 3 later.
    # So the search is: subset of {2..128} that is strictly increasing (which is just a subset) 
    # such that the order is the natural order of the subset.
    # So we are effectively searching for the largest subset S of {2..128} such that 
    # products of consecutive blocks in S are distinct.
    
    # This is a combinatorial optimization problem.
    # Backtracking over the inclusion of each number from 128 down to 2?
    # Or from 2 up to 128?
    # If we go 2 -> 128, we decide "include 2?" then "include 3?" etc.
    # If we include 2, we must check if 2 conflicts with previous (none).
    # If we include 3, we check conflicts with [2] and [3] and [2,3].
    
    # Let's use a recursive backtracker.
    # State: current_index (next number to consider, 2..128), current_seq, current_prods.
    # Optimization: Memoization is hard because 'current_prods' is a set of large ints.
    # Instead, we rely on the structure.
    
    # To improve on greedy, we need to sometimes SKIP a number that would conflict, 
    # and then maybe include a LATER number. Greedy does this. 
    # But greedy might include a number that blocks many future options, 
    # whereas skipping it would allow a denser sequence later.
    # Backtracking allows us to "un-include" a number and try the next one.
    
    # However, full backtracking over 127 numbers is 2^127. 
    # We need strong pruning.
    
    # Key insight: The products grow very fast. 
    # If we have a sequence, the products of long blocks become huge.
    # The constraint is that ALL products must be distinct.
    
    # Let's try a Depth-First Search with the following:
    # 1. Iterate candidates from 2 to 128.
    # 2. At each step, try to INCLUDE the candidate.
    # 3. If valid, recurse.
    # 4. If we reach the end, update best.
    # 5. If we fail to include, we MUST skip it (since we are iterating in order, 
    #    skipping is the only other option).
    # Wait, standard backtracking for subset selection:
    # def search(idx, seq, prods):
    #   if idx > 128: update best
    #   # Option 1: Include idx
    #   if valid(idx, seq, prods):
    #       new_seq = seq + [idx]
    #       new_prods = prods | new_products(idx, seq)
    #       search(idx+1, new_seq, new_prods)
    #   # Option 2: Exclude idx
    #   search(idx+1, seq, prods)
    
    # This explores 2^127 paths. Too slow.
    # We need to prune "Exclude" branches if we are already worse than best_len + (128 - idx).
    # And we need to prune "Include" branches if they lead to dead ends early.
    
    # A better heuristic: 
    # The "greedy" approach is fast but suboptimal.
    # Let's use a **Beam Search** with a very large width, but smarter scoring.
    # Or, let's use **Iterative Deepening** with a strict time limit per depth? No.
    
    # Let's try a **Local Search** approach starting from a good seed.
    # The current program does some local search.
    # What if we use **Simulated Annealing**?
    # State: A subset of {2..128}.
    # Move: Swap an element in the set with an element outside, or add/remove.
    # Constraint: Must be strictly increasing (handled by sorting the set).
    # Cost: -length + penalty for conflicts (if any).
    # Since we need UNIQUE products, any conflict is fatal.
    # So we can only move to VALID states.
    
    # Strategy: 
    # 1. Generate a good initial sequence using the existing greedy logic (fast).
    # 2. Use a **Randomized Local Search** (Hill Climbing with Random Restarts) 
    #    that tries to **INSERT** missing numbers at random positions, 
    #    and if that fails, tries to **REMOVE** a number that is causing conflicts 
    #    with many other potential insertions.
    
    # Actually, the most effective different strategy for "maximize length with constraints" 
    # when greedy fails is **Backtracking with Heuristic Ordering**.
    # But the search space is too big for pure backtracking.
    
    # Let's try **Genetic Algorithm** style:
    # Population of sequences.
    # Crossover: Take prefix of one, suffix of another (if compatible).
    # Mutation: Randomly add/remove/swap.
    # Selection: Keep longest valid sequences.
    
    # Given the 60s limit, a GA might be too slow to converge.
    
    # Let's look at the structure of the problem again.
    # Products of consecutive blocks.
    # If a sequence is valid, can we always extend it? No.
    
    # New Idea: **Constraint Programming with Arc Consistency (AC-3)**?
    # Too complex to implement quickly.
    
    # Let's go with **Intensive Beam Search**.
    # The current program uses a beam width of 10.
    # Let's increase the beam width significantly (e.g., 1000) and use a **Diversity Preserving** selection.
    # Instead of just keeping the top K by length, keep the top K by (length, diversity of prime factors).
    # Or, simply keep more beams.
    
    # Wait, the current program's beam search is:
    # for round in range(10): ...
    # This is very shallow.
    
    # Let's implement a **Deep Beam Search** that runs until time limit.
    # Width: 500.
    # At each step, for each beam, try all next candidates.
    # Keep top 500.
    
    # To make it different: **Use a different scoring function for the beam**.
    # Score = length * 1000 - (number of "conflict risks").
    # How to estimate conflict risk?
    # A product is a "risk" if it shares prime factors with many other numbers in [2,128].
    # If we have a product P, and P * x is likely to equal some other product?
    # Hard to predict.
    
    # Let's try **Randomized Backtracking** with a strict time budget per branch.
    # No, that's what the current program does (random restarts).
    
    # **Different Design Family: Dynamic Programming on State?**
    # State is the set of products. Too large.
    
    # **Different Design Family: Integer Linear Programming (ILP)?**
    # Standard library doesn't have an ILP solver.
    
    # **Different Design Family: SAT Solver?**
    # No standard library SAT solver.
    
    # **Different Design Family: Greedy with Lookahead (Alpha-Beta Pruning)?**
    # This is essentially backtracking.
    
    # Let's implement a **Robust Beam Search** with:
    # 1. Larger beam width (e.g., 200).
    # 2. **Diversity**: When selecting beams, don't just take the longest.
    #    Take the longest, but also ensure we have beams with different "last few elements".
    # 3. **Time-aware**: Run as many iterations as possible.
    
    # The current program's beam search is limited to 10 rounds.
    # If we run it for, say, 500 rounds (or until time runs out), we might find longer sequences.
    # Each round adds one element. Max length is 127. So 127 rounds max.
    # If we do 127 rounds with beam width 100, that's 127 * 100 * 127 checks.
    # ~1.6 million checks. Each check is O(N^2) for product calculation?
    # We can optimize the check to be O(N) by maintaining suffix products.
    
    # Let's implement this.
    
    def get_suffix_prods(seq):
        """Returns list of suffix products: [prod(seq[-1]), prod(seq[-2:]), ...]"""
        prods = []
        p = 1
        for i in range(len(seq) - 1, -1, -1):
            p *= seq[i]
            prods.append(p)
        return prods

    def can_append(seq, prods_set, suffix_prods, x):
        """
        Check if x can be appended to seq.
        seq: current sequence
        prods_set: set of all existing block products
        suffix_prods: list of suffix products of seq, where suffix_prods[0] is seq[-1], 
                      suffix_prods[1] is seq[-2]*seq[-1], etc.
        x: new element
        Returns: (bool, new_prods_set, new_suffix_prods)
        """
        if x in prods_set:
            return False, None, None
        
        new_suffix_prods = [x]
        p = x
        # Check new products: x, x*seq[-1], x*seq[-2]*seq[-1], ...
        # These correspond to:
        # new_suffix_prods[0] = x
        # new_suffix_prods[1] = x * seq[-1] = x * suffix_prods[0]
        # new_suffix_prods[2] = x * seq[-2] * seq[-1] = x * suffix_prods[1]
        
        # We need to check if any of these are in prods_set.
        # Also check if any of these are in the new_suffix_prods (duplicates among new).
        
        # Optimization: 
        # The new products are x * suffix_prods[i] for i in range(len(suffix_prods)) plus x itself?
        # Wait, suffix_prods[0] is seq[-1]. 
        # Block [x] -> x
        # Block [seq[-1], x] -> seq[-1] * x = suffix_prods[0] * x
        # Block [seq[-2], seq[-1], x] -> suffix_prods[1] * x
        # ...
        # Block [seq, x] -> suffix_prods[-1] * x
        
        # So new products are {x} U {x * s for s in suffix_prods}
        
        new_prods = [x]
        for s in suffix_prods:
            new_prods.append(x * s)
            
        # Check for duplicates within new_prods
        if len(new_prods) != len(set(new_prods)):
            return False, None, None
            
        # Check against existing prods
        for p_new in new_prods:
            if p_new in prods_set:
                return False, None, None
                
        # If valid, construct new state
        # New suffix prods for the new sequence:
        # The new sequence ends with x.
        # Suffix of length 1: x
        # Suffix of length 2: seq[-1] * x = suffix_prods[0] * x
        # Suffix of length 3: suffix_prods[1] * x
        # ...
        # Suffix of length k+1: suffix_prods[k-1] * x
        
        # So new_suffix_prods = [x, x*suffix_prods[0], x*suffix_prods[1], ...]
        # Note: The order in my `get_suffix_prods` helper was [last, last-1..last, ...]
        # So new_suffix_prods should follow the same convention.
        
        ns = [x]
        for s in suffix_prods:
            ns.append(x * s)
            
        # Update prods set
        # We can't just copy the set every time if it's huge, but for beam search we do.
        # To be faster, we could use a "versioned" set or just copy.
        # Given N <= 127, prods set size <= N(N+1)/2 ~ 8000. Copying is fast.
        
        new_prods_set = prods_set | set(new_prods)
        
        return True, new_prods_set, ns

    # Beam Search
    # State: (seq, prods_set, suffix_prods, length)
    
    # Initial state
    # We can start with empty sequence.
    # But to speed up, we can start with the greedy result as one beam.
    
    # Let's run the greedy first to get a baseline.
    greedy_seq = []
    greedy_prods = set()
    greedy_suffix = []
    for x in range(2, 129):
        if not greedy_seq:
            greedy_seq.append(x)
            greedy_prods.add(x)
            greedy_suffix = [x]
            continue
        ok, new_p, new_s = can_append(greedy_seq, greedy_prods, greedy_suffix, x)
        if ok:
            greedy_seq.append(x)
            greedy_prods = new_p
            greedy_suffix = new_s
            
    if len(greedy_seq) > best_len:
        best_len = len(greedy_seq)
        best_seq = greedy_seq[:]
        
    # Initialize beam with the greedy sequence and some random short sequences
    beam = []
    if greedy_seq:
        beam.append((greedy_seq, greedy_prods, greedy_suffix))
        
    # Add a few random starts to ensure diversity
    import random
    random.seed(123)
    for _ in range(20):
        start_val = random.randint(2, 128)
        seq = [start_val]
        prods = {start_val}
        suffix = [start_val]
        beam.append((seq, prods, suffix))
        
    # Sort beam by length descending
    beam.sort(key=lambda x: -len(x[0]))
    
    beam_width = 500
    beam = beam[:beam_width]
    
    # Iterate
    # We will expand the beam until we can't anymore or time runs out.
    # Max depth is 127.
    
    while time.time() < deadline:
        new_beam = []
        
        for seq, prods, suffix in beam:
            if time.time() > deadline:
                break
            
            last_val = seq[-1] if seq else 1
            # Try all candidates greater than last_val
            for x in range(last_val + 1, 129):
                if time.time() > deadline:
                    break
                
                ok, new_prods, new_suffix = can_append(seq, prods, suffix, x)
                if ok:
                    new_seq = seq + [x]
                    # Update best
                    if len(new_seq) > best_len:
                        best_len = len(new_seq)
                        best_seq = new_seq[:]
                        
                    new_beam.append((new_seq, new_prods, new_suffix))
                    
        if not new_beam:
            break
            
        # Select top beam_width
        # Sort by length descending. To break ties, we can use a hash of the last few elements to encourage diversity?
        # Or just sort by length.
        new_beam.sort(key=lambda x: -len(x[0]))
        
        # To ensure diversity, we can limit the number of beams with the same prefix?
        # For now, just take top K.
        beam = new_beam[:beam_width]
        
        # If the longest sequence in the beam is the same as the best found, 
        # and we've exhausted all extensions for that length, we might be stuck.
        # But beam search handles this by keeping multiple options.
        
    return best_seq

# Quick test to ensure no syntax errors
if __name__ == "__main__":
    res = construct()
    print(f"Result length: {len(res)}")
    print(f"First 10: {res[:10]}")
