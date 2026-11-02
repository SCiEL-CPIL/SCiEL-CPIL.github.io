import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    # The task specifies q=3, n=8, radius=1.
    # We handle the general case but optimize for the specific task.
    if q != 3 or n != 8 or radius != 1:
        # Fallback to seed logic for other parameters if they ever appear
        # though the contract implies specific task parameters.
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    N = 3 ** n
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Precompute neighbors for all words in the space
    # word is an integer in [0, N-1]
    # neighbors[word] is a list of integers (the word itself + 16 neighbors)
    neighbors = [[] for _ in range(N)]
    
    # To speed up precomputation, we can use base-3 arithmetic properties
    # Or simply iterate. 6561 * 17 ops is tiny.
    for i in range(N):
        # Convert i to base 3
        digits = []
        tmp = i
        for _ in range(n):
            digits.append(tmp % 3)
            tmp //= 3
        # digits is reversed (LSB first)
        
        ball = [i]
        for pos in range(n):
            orig = digits[pos]
            for new_val in range(3):
                if new_val == orig:
                    continue
                # Calculate new integer
                diff = new_val - orig
                new_int = i + diff * (3 ** pos)
                ball.append(new_int)
        neighbors[i] = ball

    # Greedy Initialization
    # cover_count[i] = number of codewords covering i
    cover_count = np.zeros(N, dtype=np.int32)
    code = []
    uncovered = set(range(N))
    
    # Optimization: Instead of checking all N candidates every step (O(N^2)),
    # we can just pick the best candidate. With N=6561, N^2 is ~43M ops.
    # In Python, this might be slow. 
    # Better: Maintain a list of uncovered. For each candidate, count how many it covers.
    # We can vectorize the "how many does candidate c cover" check.
    
    # Let's try a simpler greedy:
    # 1. Start with empty code.
    # 2. While uncovered is not empty:
    #    a. Find candidate c that maximizes |ball(c) ∩ uncovered|
    #    b. Add c to code.
    #    c. Update uncovered.
    
    # To make this faster, we can use the fact that balls are small.
    # We can iterate through all N candidates and compute the count.
    # This is O(N * N * 17) in worst case if done naively? No.
    # Step 2a is O(N * 17) if we have a boolean array of uncovered.
    # Total steps ~ N/17. Total ops ~ (N/17) * N * 17 = N^2 = 43M.
    # This is feasible in NumPy.
    
    uncovered_mask = np.ones(N, dtype=bool)
    
    while True:
        if time.time() > deadline:
            break
        if not np.any(uncovered_mask):
            break
            
        # Calculate coverage for all candidates
        # We need to sum uncovered_mask[neighbors[i]] for all i.
        # This is hard to vectorize directly without a matrix.
        # Alternative: Iterate through uncovered words? No.
        
        # Let's use a different strategy:
        # Pick a random uncovered word, find its neighbors, pick the best neighbor.
        # This is a heuristic greedy (like Set Cover).
        
        # Pick a random uncovered index
        unc_idx = np.where(uncovered_mask)[0]
        if len(unc_idx) == 0:
            break
        target = unc_idx[random.randint(0, len(unc_idx)-1)]
        
        # Find the candidate in neighbors[target] that covers the most uncovered words.
        # Actually, the best candidate is usually the one that is "central" to the uncovered region.
        # Let's just check the 17 neighbors of `target`. One of them must cover `target`.
        # We pick the one that covers the most OTHER uncovered words.
        
        best_cand = -1
        max_cov = -1
        for cand in neighbors[target]:
            # Count how many uncovered words are in neighbors[cand]
            # This is a dot product or sum of boolean array
            cov = np.sum(uncovered_mask[neighbors[cand]])
            if cov > max_cov:
                max_cov = cov
                best_cand = cand
        
        if best_cand == -1:
            break
            
        # Add best_cand
        code.append(best_cand)
        
        # Update cover_count and uncovered_mask
        for w in neighbors[best_cand]:
            if uncovered_mask[w]:
                uncovered_mask[w] = False
                # We don't strictly need cover_count for greedy, but for local search we do.
                # Let's build cover_count from scratch after greedy? 
                # Or maintain it.
        
        # It is faster to rebuild cover_count after greedy phase.
        
    # Rebuild cover_count
    cover_count = np.zeros(N, dtype=np.int32)
    for c in code:
        for w in neighbors[c]:
            cover_count[w] += 1
            
    # Local Search
    # 1. Removal
    # 2. Swap
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Pass 1: Removal
        # Iterate backwards to allow removal of multiple items in one pass
        for i in range(len(code) - 1, -1, -1):
            if time.time() > deadline:
                break
            c = code[i]
            # Check if any word in neighbors[c] has cover_count == 1
            # If yes, removing c would uncover it.
            # If no, remove c.
            counts_in_ball = cover_count[neighbors[c]]
            if np.min(counts_in_ball) > 1:
                # Safe to remove
                for w in neighbors[c]:
                    cover_count[w] -= 1
                code.pop(i)
                improved = True
        
        # Pass 2: Swap
        # Try to swap each codeword c with some other word w
        # Condition: 
        # 1. For all v in neighbors[c] \ neighbors[w], cover_count[v] > 1 (so they remain covered by others)
        # 2. Ideally, this swap might allow future removals, but we just need to maintain coverage.
        #    Actually, if we swap c with w, and coverage is maintained, we might have "loosened" 
        #    the constraint on w's ball, allowing removals in the next pass.
        #    But a swap is only "beneficial" if it leads to a smaller code. 
        #    A simple swap that maintains coverage doesn't reduce size immediately.
        #    However, it might move a "redundant" codeword to a "more useful" position?
        #    No, if it's redundant, we should have removed it.
        #    Swaps are useful if we swap a codeword that is "barely" needed with one that is "very" needed?
        #    Actually, standard approach: 
        #    Swap c with w if:
        #    - The new set covers everything.
        #    - AND, we can then remove at least one other codeword?
        #    That's complex.
        
        # Alternative Swap Logic:
        # A codeword c is "critical" if removing it breaks coverage.
        # If c is not critical, we removed it in Pass 1.
        # So after Pass 1, all codewords are critical.
        # Can we swap a critical codeword c with w to make it non-critical?
        # If we swap c with w, c is gone. w is added.
        # If w covers all the words that c uniquely covered, then c's unique words are now covered by w.
        # Does this make w non-critical? Not necessarily.
        # Does it make some OTHER codeword non-critical? Possibly.
        
        # Let's try: For each c, find w such that:
        # 1. coverage is maintained (no word drops to 0).
        # 2. The number of words with count==1 in the new configuration is smaller?
        #    Or simply: try all w in the space? Too slow.
        #    Try w in neighbors of some uncovered-ish area?
        
        # Let's stick to a simpler heuristic often used:
        # Try to replace c with a word w that is far from c?
        # Actually, let's just do random swaps that maintain coverage, and hope removal finds a benefit.
        # This is stochastic local search.
        
        # To keep it deterministic and fast, we can try swaps with neighbors of neighbors.
        
        # Let's try a limited swap: 
        # For each c, look at its neighbors. If we move c to a neighbor w,
        # does it free up capacity?
        
        # Given time constraints, a robust Removal + Randomized Restart Greedy might be better than complex swaps.
        # But the prompt mentions "2-exchange" and "swap" in inspirations.
        
        # Let's implement a simple swap:
        # For each c in code:
        #   For each w in neighbors[c] (or a random subset):
        #     Check if swap maintains coverage.
        #     If yes, do it. (This doesn't reduce size, but changes state).
        #     Then run Removal again.
        
        # This loop might be slow if we check all w.
        # Let's limit w to the 17 neighbors of c.
        
        for i in range(len(code)):
            if time.time() > deadline:
                break
            c = code[i]
            # Try swapping with neighbors of c
            for w in neighbors[c]:
                if w == c:
                    continue
                # Check if swap is valid
                # Words that lose c: neighbors[c]
                # Words that gain w: neighbors[w]
                # A word v is at risk if v in neighbors[c] and v NOT in neighbors[w].
                # For such v, cover_count[v] must be > 1.
                
                # Optimization: 
                # We can just simulate the change on a copy or check conditions.
                
                # Condition: min(cover_count[v] for v in neighbors[c] if v not in neighbors[w]) > 1
                # Note: words in intersection are fine (count drops then rises, net 0, but min count during process? 
                # No, we just care about final state. Final count is same for intersection.
                # For v in N(c) \ N(w), count decreases by 1. Must be >= 1.
                # For v in N(w) \ N(c), count increases by 1. Fine.
                
                # We need to check v in N(c) \ N(w).
                # Since balls are small, we can just iterate.
                
                valid = True
                for v in neighbors[c]:
                    if v not in neighbors[w]: # This 'in' check is O(17)
                        if cover_count[v] == 1:
                            valid = False
                            break
                
                if valid:
                    # Perform swap
                    for v in neighbors[c]:
                        cover_count[v] -= 1
                    for v in neighbors[w]:
                        cover_count[v] += 1
                    code[i] = w
                    improved = True
                    break # Restart removal pass? Or just continue? 
                    # Breaking inner loop to re-evaluate removals is safer.
            
            if improved:
                # If we did a swap, we should probably re-run removal.
                # We can just break out to the while loop start.
                break

    # Convert indices back to lists of integers
    result = []
    for c in code:
        word = []
        tmp = c
        for _ in range(n):
            word.append(tmp % 3)
            tmp //= 3
        word.reverse()
        result.append(word)
        
    return result
