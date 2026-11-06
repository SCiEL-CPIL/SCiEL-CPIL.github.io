import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    N = 256
    
    # Precompute all possible consecutive block sums for a given sequence?
    # No, we build incrementally.
    
    # The key insight for speed: 
    # When we add a new element x to the end of a sequence seq, the new sums are:
    # x, seq[-1]+x, seq[-2]+seq[-1]+x, ..., sum(seq)+x
    # We need to check if any of these are in the existing set of sums.
    # To make this faster, we maintain the set of current sums.
    
    # However, the "randomized restarts" insight suggests that deterministic 
    # greedy or simple DFS might get stuck. 
    # We will use a DFS that picks the next element from a randomized candidate list.
    # We will try to extend the sequence as far as possible.
    # If we hit a dead end (no valid next element), we backtrack.
    
    # To handle the 60s limit, we will run multiple randomized DFS attempts.
    # Each attempt tries to build a long sequence.
    
    def check_and_add(sums_set, seq, x):
        """
        Check if x can be appended to seq.
        Returns True if valid, and updates sums_set if valid.
        Note: We pass sums_set by reference, so we must revert if invalid.
        To avoid complex reverting, we will compute the new sums and check against sums_set.
        If valid, we add them to sums_set.
        """
        n = len(seq)
        new_sums = []
        s = 0
        # Calculate new sums ending at x
        for i in range(n - 1, -1, -1):
            s += seq[i]
            new_sums.append(s)
        
        # Check for conflicts
        for ns in new_sums:
            if ns in sums_set:
                return False, []
        
        # No conflicts
        return True, new_sums

    def dfs(seq, sums_set, depth, max_depth, candidates, time_limit):
        """
        Depth-first search to extend the sequence.
        seq: current sequence (list)
        sums_set: set of all current block sums
        depth: current depth (len(seq))
        max_depth: target depth
        candidates: list of available numbers to try
        time_limit: absolute time to stop
        """
        if depth == max_depth:
            return seq[:]
        
        # Try to extend
        # We want to pick a candidate that allows further extension.
        # Randomization helps avoid deterministic traps.
        
        # Optimization: Instead of iterating all candidates, we can shuffle them 
        # and try a few. But backtracking requires trying all if we want to be complete.
        # Given the small N (256) and the constraint, the branching factor is high.
        
        # Strategy: 
        # 1. Sort candidates by some heuristic? 
        #    Larger numbers might create larger sums, reducing collisions with small sums?
        #    Or smaller numbers to keep sums small?
        #    Actually, distinct sums are required. 
        #    Let's just iterate through a shuffled list of candidates.
        
        # To speed up, we can prune candidates that are "obviously" bad?
        # E.g., if x is in sums_set, it's invalid immediately (sum of length 1).
        # Also if any suffix sum + x is in sums_set.
        
        # We will iterate through candidates in a random order.
        # If we find a valid extension, we recurse.
        # If recursion fails (dead end), we try the next candidate.
        
        # To prevent exponential blowup, we limit the branching or use a timeout.
        
        # Since we are doing randomized restarts, each DFS run can be "greedy" with 
        # limited backtracking depth or just a single path with random choices?
        # No, standard DFS backtracking is better for finding a valid long path.
        
        # However, full DFS on 256 nodes is too slow.
        # We will use a "Limited DFS" or "Greedy with Backtracking" hybrid.
        
        # Let's try a different approach: 
        # Randomized Greedy Construction with Local Repair?
        # No, the directive asks for a structural change.
        
        # Let's stick to Randomized DFS with a depth limit per branch?
        # Actually, the "Randomized restarts" that scored 1.436 likely used 
        # multiple independent greedy attempts with different random seeds/orders.
        
        # Let's implement a robust Randomized Greedy with Backtracking (Limited).
        # We will try to build a sequence of length L.
        # We pick the next element randomly from valid candidates.
        # If we get stuck, we backtrack one step and pick a different candidate.
        # We limit the number of backtracks to keep it fast.
        
        pass # This function is too complex to implement efficiently in a short time 
             # without knowing the exact performance characteristics.
    
    # Alternative Structural Change:
    # **Integer Linear Programming (ILP)** formulation?
    # No, SciPy's linprog is for LP, not ILP (unless using pulp, which is not stdlib).
    # But we can use a greedy approach that is smarter.
    
    # Let's look at the "Randomized restarts" insight again.
    # "Randomized restarts escape local optima of the deterministic greedy".
    # This suggests that running a greedy algorithm many times with different 
    # random priorities for candidate selection is effective.
    
    # Strategy:
    # 1. Initialize an empty sequence.
    # 2. Maintain a set of current sums.
    # 3. At each step, generate a list of valid candidates (numbers in [1,256] not used).
    # 4. Randomly shuffle this list.
    # 5. Pick the first candidate that is valid.
    # 6. Repeat until no valid candidates.
    # 7. Repeat this entire process many times (restarts) with different random seeds.
    # 8. Keep the longest sequence found.
    
    # This is essentially "Randomized Greedy". It is different from Simulated Annealing 
    # because it does not allow invalid states or swaps; it is a constructive heuristic.
    
    def randomized_greedy(seed):
        random.seed(seed)
        seq = []
        used = set()
        sums_set = set()
        
        # Precompute suffix sums? No, seq changes.
        
        for _ in range(N): # Max length N
            # Find all valid candidates
            valid_candidates = []
            for x in range(1, N + 1):
                if x in used:
                    continue
                
                # Check if x can be added
                # New sums are: x, seq[-1]+x, ..., sum(seq)+x
                # We need to check if any of these are in sums_set.
                # Also x itself must not be in sums_set (which is covered if seq is empty, 
                # but if seq is not empty, x is a new sum of length 1).
                
                # Optimization: 
                # If x is in sums_set, skip.
                if x in sums_set:
                    continue
                
                # Check suffix sums
                suffix_sum = 0
                valid = True
                # Iterate backwards through seq
                for i in range(len(seq) - 1, -1, -1):
                    suffix_sum += seq[i]
                    if suffix_sum in sums_set:
                        valid = False
                        break
                
                if valid:
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            # Randomly pick one
            # To introduce more variation, we can pick randomly from the top K candidates
            # or just randomly from all valid candidates.
            x = random.choice(valid_candidates)
            
            # Add x to seq
            seq.append(x)
            used.add(x)
            
            # Update sums_set
            suffix_sum = 0
            for i in range(len(seq) - 1, -1, -1):
                suffix_sum += seq[i]
                sums_set.add(suffix_sum)
        
        return seq

    best_seq = []
    best_len = 0
    
    # Run multiple randomized greedy attempts
    # How many? 
    # Each attempt is O(N^2) roughly (N steps, each checking up to N candidates, each O(N) check).
    # N=256. 256^3 ~ 16M ops. In Python, this might be slow.
    # Let's optimize the check.
    
    # Optimized Check:
    # We can precompute the suffix sums of the current sequence.
    # When we add x, the new sums are x + suffix_sum[i] for i in 0..len(seq).
    # We just need to check if any of these are in sums_set.
    
    def randomized_greedy_fast(seed):
        random.seed(seed)
        seq = []
        used = set()
        sums_set = set()
        
        # suffix_sums[i] = sum(seq[i:])
        # Initially empty
        
        for _ in range(N):
            valid_candidates = []
            
            # We can filter candidates faster if we maintain some structure, 
            # but for now, let's just loop.
            # To speed up, we can break early in the suffix check.
            
            for x in range(1, N + 1):
                if x in used:
                    continue
                if x in sums_set:
                    continue
                
                # Check suffix sums
                # We need sum(seq[i:]) + x for all i
                # Let's compute these on the fly
                s = 0
                valid = True
                for i in range(len(seq) - 1, -1, -1):
                    s += seq[i]
                    if s in sums_set:
                        valid = False
                        break
                
                if valid:
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            x = random.choice(valid_candidates)
            seq.append(x)
            used.add(x)
            
            # Update sums_set
            s = 0
            for i in range(len(seq) - 1, -1, -1):
                s += seq[i]
                sums_set.add(s)
        
        return seq

    # How many restarts can we do in 85s?
    # Let's test 1 restart timing.
    test_seq = randomized_greedy_fast(0)
    t1 = time.time()
    elapsed = t1 - start_time
    
    # Estimate ops per second
    # If 1 restart takes T seconds, we can do 85/T restarts.
    # Let's run as many as possible.
    
    # To ensure we find a good solution, we should also try different 
    # "biases" in the random selection.
    # E.g., prefer larger numbers, prefer smaller numbers.
    
    # Let's add a parameter to bias the choice.
    def randomized_greedy_biased(seed, bias='random'):
        random.seed(seed)
        seq = []
        used = set()
        sums_set = set()
        
        for _ in range(N):
            valid_candidates = []
            
            for x in range(1, N + 1):
                if x in used:
                    continue
                if x in sums_set:
                    continue
                
                s = 0
                valid = True
                for i in range(len(seq) - 1, -1, -1):
                    s += seq[i]
                    if s in sums_set:
                        valid = False
                        break
                
                if valid:
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            if bias == 'random':
                x = random.choice(valid_candidates)
            elif bias == 'min':
                x = min(valid_candidates)
            elif bias == 'max':
                x = max(valid_candidates)
            elif bias == 'median':
                valid_candidates.sort()
                x = valid_candidates[len(valid_candidates)//2]
            else:
                x = random.choice(valid_candidates)
            
            seq.append(x)
            used.add(x)
            
            s = 0
            for i in range(len(seq) - 1, -1, -1):
                s += seq[i]
                sums_set.add(s)
        
        return seq

    # Run a mix of strategies
    # 1. Random restarts with random choice
    # 2. Deterministic greedy (min, max)
    # 3. Random restarts with min/max bias?
    
    # First, get a baseline from deterministic greedy
    seq_min = randomized_greedy_biased(0, 'min')
    seq_max = randomized_greedy_biased(0, 'max')
    
    best_seq = max([seq_min, seq_max], key=len)
    best_len = len(best_seq)
    
    # Now run randomized restarts
    # We need to be careful with time.
    # Let's run 100 restarts or until time is up.
    
    for i in range(1, 1000):
        if time.time() > deadline - 5:
            break
        
        # Mix of random and biased
        if i % 3 == 0:
            seq = randomized_greedy_biased(i, 'min')
        elif i % 3 == 1:
            seq = randomized_greedy_biased(i, 'max')
        else:
            seq = randomized_greedy_biased(i, 'random')
        
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)
    
    # Final verification
    def check_valid(seq):
        seen = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in seen:
                    return False
                seen.add(s)
        return True

    if not check_valid(best_seq):
        # Fallback to a known valid sequence (e.g., powers of 2? No, must be in [1,256])
        # Powers of 2: 1, 2, 4, 8, 16, 32, 64, 128 (8 elements)
        # This is a valid sequence.
        best_seq = [1, 2, 4, 8, 16, 32, 64, 128]
        
    return best_seq

# Keep the original function name and signature
def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    N = 256

    def randomized_greedy_biased(seed, bias='random'):
        random.seed(seed)
        seq = []
        used = set()
        sums_set = set()
        
        for _ in range(N):
            valid_candidates = []
            
            for x in range(1, N + 1):
                if x in used:
                    continue
                if x in sums_set:
                    continue
                
                s = 0
                valid = True
                for i in range(len(seq) - 1, -1, -1):
                    s += seq[i]
                    if s in sums_set:
                        valid = False
                        break
                
                if valid:
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            if bias == 'random':
                x = random.choice(valid_candidates)
            elif bias == 'min':
                x = min(valid_candidates)
            elif bias == 'max':
                x = max(valid_candidates)
            elif bias == 'median':
                valid_candidates.sort()
                x = valid_candidates[len(valid_candidates)//2]
            else:
                x = random.choice(valid_candidates)
            
            seq.append(x)
            used.add(x)
            
            s = 0
            for i in range(len(seq) - 1, -1, -1):
                s += seq[i]
                sums_set.add(s)
        
        return seq

    def check_valid(seq):
        seen = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in seen:
                    return False
                seen.add(s)
        return True

    # 1. Deterministic Greedy Baselines
    seq_min = randomized_greedy_biased(0, 'min')
    seq_max = randomized_greedy_biased(0, 'max')
    
    best_seq = max([seq_min, seq_max], key=len)
    best_len = len(best_seq)
    
    # 2. Randomized Restarts
    # We will run as many as possible within the time limit.
    # Each run is O(N^2) to O(N^3). 
    # N=256. 
    # Let's monitor time.
    
    for i in range(1, 500):
        if time.time() > deadline - 2:
            break
        
        # Cycle through biases to explore different parts of the solution space
        # 'random' is the most likely to find diverse solutions
        # 'min' and 'max' might find specific structural solutions
        if i % 4 == 0:
            bias = 'min'
        elif i % 4 == 1:
            bias = 'max'
        elif i % 4 == 2:
            bias = 'median'
        else:
            bias = 'random'
            
        seq = randomized_greedy_biased(i, bias)
        
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)
            
    # Final verification
    if not check_valid(best_seq):
        # Fallback
        best_seq = [1, 2, 4, 8, 16, 32, 64, 128]
        
    return best_seq
