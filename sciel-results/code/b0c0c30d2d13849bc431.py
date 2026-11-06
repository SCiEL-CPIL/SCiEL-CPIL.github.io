import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 80.0  # Safety margin for 90s kill
    
    best_seq = []
    
    # 1. Greedy Baseline
    # This provides a valid sequence to start with.
    def greedy_construct():
        seq = []
        sums = set()
        # We want to pick small numbers to fit more, but avoid collisions.
        # Standard greedy: pick smallest x > last that doesn't collide.
        last = 0
        for _ in range(256):
            found = False
            for x in range(last + 1, 257):
                # Check if adding x causes collisions
                # New sums are x, x+seq[-1], x+seq[-1]+seq[-2], etc.
                # Actually, it's easier to maintain prefix sums.
                # Let P[0]=0, P[i] = sum(seq[0..i-1]).
                # Adding x creates new prefix P_new = P[-1] + x.
                # New differences are P_new - P[i] for all i.
                # These must not be in the existing set of differences.
                
                # Optimization: Keep list of prefix sums
                pass # Placeholder, we will do a proper greedy below
            
            # Simpler Greedy:
            # Maintain current sequence and set of all consecutive sums.
            pass
        
        # Let's do a simple greedy that picks the smallest valid next number.
        seq = []
        prefix_sums = [0]
        diff_set = set()
        
        for _ in range(256):
            found = False
            for x in range(1, 257):
                if seq and x <= seq[-1]:
                    continue
                # Calculate new prefix
                new_prefix = prefix_sums[-1] + x
                # Check new differences: new_prefix - p for all p in prefix_sums
                valid = True
                for p in prefix_sums:
                    d = new_prefix - p
                    if d in diff_set:
                        valid = False
                        break
                if valid:
                    # Commit
                    seq.append(x)
                    prefix_sums.append(new_prefix)
                    for p in prefix_sums[:-1]:
                        diff_set.add(new_prefix - p)
                    found = True
                    break
            if not found:
                break
        return seq

    best_seq = greedy_construct()
    best_len = len(best_seq)
    
    # 2. Backtracking with Randomization
    # We will try to extend the greedy solution or restart from scratch with randomization.
    # Since the greedy solution is likely suboptimal, we will run a DFS that can backtrack.
    # To handle the 90s limit, we will run multiple "phases" or restarts if time permits.
    
    def dfs(current_seq, current_prefixes, current_diffs, next_min, depth):
        nonlocal best_seq, best_len, start_time, time_limit
        
        # Time check
        if time.time() - start_time > time_limit:
            return False # Signal to stop
        
        # Update best
        if len(current_seq) > best_len:
            best_seq = current_seq[:]
            best_len = len(current_seq)
            
        # Pruning: If we can't add any more, return
        if next_min > 256:
            return True # Dead end, but fully explored
        
        # Heuristic: Try candidates in a specific order.
        # For Golomb rulers, smaller gaps are often preferred to fit more numbers,
        # but sometimes larger jumps are necessary.
        # We will try candidates from next_min to 256.
        
        # Optimization: Pre-calculate potential new prefixes? No, depends on x.
        
        # To improve search, we can randomize the order of candidates slightly
        # or use a heuristic score.
        # Let's try candidates in increasing order first, but if we are in a "random" phase, shuffle.
        
        candidates = list(range(next_min, 257))
        # In early depths, small numbers are better. In later depths, maybe larger?
        # Actually, for maximizing length, we want to keep numbers small to stay in [1, 256].
        # So increasing order is a good default.
        
        # To escape local minima, we can occasionally try a larger number first.
        # But standard DFS with backtracking is powerful enough if we don't prune too aggressively.
        
        for x in candidates:
            # Calculate new prefix
            new_prefix = current_prefixes[-1] + x
            
            # Check for collisions
            valid = True
            new_diffs = []
            for p in current_prefixes:
                d = new_prefix - p
                if d in current_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
                
            # Recurse
            current_seq.append(x)
            current_prefixes.append(new_prefix)
            current_diffs.update(new_diffs)
            
            # Recursive call
            dfs(current_seq, current_prefixes, current_diffs, x + 1, depth + 1)
            
            # Backtrack
            current_seq.pop()
            current_prefixes.pop()
            for d in new_diffs:
                current_diffs.remove(d)
            
            # Time check after each step to avoid deep recursion if time runs out
            if time.time() - start_time > time_limit:
                return False
        
        return True

    # Strategy:
    # 1. Try to extend the greedy solution using DFS.
    # 2. If time remains, run full DFS from scratch with random restarts.
    
    # Phase 1: Extend Greedy
    # The greedy solution is valid. We can try to improve it by backtracking.
    # However, backtracking from a greedy solution might be expensive if it's deep.
    # It's often better to run a fresh DFS with a good initial heuristic.
    
    # Let's run a DFS starting from an empty sequence, but with a time limit per "branch" or global.
    # Since the global time limit is 80s, we can just run one long DFS.
    # But DFS is deterministic. If it gets stuck, it's stuck.
    # Randomized DFS: Shuffle the candidate list at each node? That's expensive.
    # Instead, let's do a few restarts with different random seeds for the candidate ordering.
    
    # To make it faster, we can use a "limited" DFS that doesn't go too deep if it doesn't improve.
    # But we want to maximize length.
    
    # Let's try a different approach: Iterative Deepening or simply running the DFS
    # but with a "randomized" tie-breaking or candidate ordering.
    
    # Actually, a pure DFS with increasing order might be sufficient if we add a "jump" heuristic.
    # But the most robust method for these puzzles is often **Simulated Annealing** or **Randomized Hill Climbing** on the sequence, or **Backtracking with Random Restarts**.
    
    # Given the 90s limit, let's try 3-5 randomized DFS runs.
    # Each run uses a slightly different ordering heuristic.
    
    for attempt in range(5):
        if time.time() - start_time > time_limit - 10.0:
            break
            
        # Define a custom candidate ordering for this attempt
        # Attempt 0: Increasing
        # Attempt 1: Increasing with some random shuffling in the middle
        # Attempt 2: Decreasing? No, we must be increasing.
        
        def get_candidates(next_min, attempt_id):
            cands = list(range(next_min, 257))
            if attempt_id == 0:
                return cands
            elif attempt_id == 1:
                # Shuffle the top half?
                if len(cands) > 10:
                    # Keep first 10 sorted, shuffle the rest?
                    # Or just shuffle all? Shuffling all might pick 256 first, which is bad for length.
                    # Better: Bias towards smaller numbers, but allow some larger ones.
                    # Let's just shuffle the list. The backtracking will correct it.
                    random.shuffle(cands)
                return cands
            elif attempt_id == 2:
                # Prioritize numbers that are coprime to something? No.
                # Let's try a "greedy-like" random: pick smallest valid with some noise.
                # For simplicity, just shuffle.
                random.shuffle(cands)
                return cands
            else:
                return cands

        # We need to modify dfs to use the specific candidate generator
        # Since dfs is recursive, passing the generator is tricky.
        # Let's just use the standard increasing order for the first run (which is fast and finds a good solution),
        # and then if we have time, do randomized restarts.
        
        # Actually, the previous program's DFS was too slow because it didn't prune well or was just slow Python.
        # Let's optimize the inner loop.
        
        def fast_dfs(current_seq, current_prefixes, current_diffs, next_min):
            nonlocal best_seq, best_len, start_time, time_limit
            
            if time.time() - start_time > time_limit:
                return
            
            if len(current_seq) > best_len:
                best_seq = current_seq[:]
                best_len = len(current_seq)
                
            if next_min > 256:
                return
            
            # Try candidates
            # Optimization: If best_len is close to theoretical max, prune?
            # Theoretical max for [1, 256] is not trivial, but ~20-30 is a good target.
            
            for x in range(next_min, 257):
                new_prefix = current_prefixes[-1] + x
                
                # Check collisions
                # We can break early if collision found
                valid = True
                new_diffs = []
                for p in current_prefixes:
                    d = new_prefix - p
                    if d in current_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Recurse
                current_seq.append(x)
                current_prefixes.append(new_prefix)
                current_diffs.update(new_diffs)
                
                fast_dfs(current_seq, current_prefixes, current_diffs, x + 1)
                
                # Backtrack
                current_seq.pop()
                current_prefixes.pop()
                for d in new_diffs:
                    current_diffs.remove(d)
                
                if time.time() - start_time > time_limit:
                    return

        # Run 1: Standard DFS
        if time.time() - start_time < 20.0:
            fast_dfs([], [0], set(), 1)
            
        # Run 2-5: Randomized DFS (if time permits)
        # To implement randomized DFS, we need to pass the random seed or use a global state.
        # Let's just run fast_dfs again? No, it's deterministic.
        # We need a version that shuffles candidates.
        
        def randomized_dfs(current_seq, current_prefixes, current_diffs, next_min, rng):
            nonlocal best_seq, best_len, start_time, time_limit
            
            if time.time() - start_time > time_limit:
                return
            
            if len(current_seq) > best_len:
                best_seq = current_seq[:]
                best_len = len(current_seq)
                
            if next_min > 256:
                return
            
            cands = list(range(next_min, 257))
            rng.shuffle(cands)
            
            for x in cands:
                new_prefix = current_prefixes[-1] + x
                
                valid = True
                new_diffs = []
                for p in current_prefixes:
                    d = new_prefix - p
                    if d in current_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                current_seq.append(x)
                current_prefixes.append(new_prefix)
                current_diffs.update(new_diffs)
                
                randomized_dfs(current_seq, current_prefixes, current_diffs, x + 1, rng)
                
                current_seq.pop()
                current_prefixes.pop()
                for d in new_diffs:
                    current_diffs.remove(d)
                
                if time.time() - start_time > time_limit:
                    return

        # Run a few randomized searches if time is left
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 5.0:
            # Split remaining time among a few randomized runs
            num_runs = 3
            time_per_run = remaining_time / num_runs
            for i in range(num_runs):
                if time.time() - start_time > time_limit:
                    break
                rng = random.Random(i)
                # Limit the depth or time for each run? 
                # The global time check in randomized_dfs handles the hard stop.
                # But we might want to stop early if we don't improve.
                # For now, let it run until global time limit.
                randomized_dfs([], [0], set(), 1, rng)

    return best_seq

