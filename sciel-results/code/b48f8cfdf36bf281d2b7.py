import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit

    # Helper to check if adding a value x to the end of a sequence is valid
    # current_sums is a set of all existing consecutive block sums
    # seq is the current sequence
    # Returns the new sums to add if valid, else None
    def get_new_sums(seq, x, current_sums):
        if seq and x <= seq[-1]:
            return None
        
        new_sums = set()
        total = x
        if total in current_sums:
            return None
        new_sums.add(total)
        
        for a in reversed(seq):
            total += a
            if total in current_sums:
                return None
            new_sums.add(total)
            
        return new_sums

    # Greedy baseline: pick the smallest valid number from [1, 256]
    # This ensures we have a valid, non-trivial sequence immediately.
    # Usually powers of 2 or similar sparse sequences work well here.
    def greedy_baseline():
        seq = []
        sums = set()
        for x in range(1, 257):
            new_sums = get_new_sums(seq, x, sums)
            if new_sums is not None:
                seq.append(x)
                sums.update(new_sums)
        return seq, sums

    # Initialize with a known valid seed to ensure non-zero score
    # The greedy baseline is fast and deterministic.
    best_seq, best_sums = greedy_baseline()
    best_len = len(best_seq)
    
    # If greedy didn't find anything (unlikely, 1 should always work), fallback
    if not best_seq:
        best_seq = [1]
        best_sums = {1}
        best_len = 1

    # Backtracking search to improve upon the greedy baseline
    # We will use a randomized DFS with pruning.
    # State: seq, sums, remaining candidates (as a set or list)
    # To speed up, we can pre-filter candidates that are obviously bad?
    # Or just shuffle and try.
    
    # Optimization: Use a list for remaining candidates for faster iteration?
    # Set is better for "remove" but slower to iterate/shuffle.
    # Since N=256 is small, list might be faster if we manage indices carefully.
    # But set is simpler. Let's stick to set for correctness first.
    
    def backtrack(seq, sums, available):
        nonlocal best_seq, best_sums, best_len
        
        if time.time() > deadline:
            return
        
        # Pruning: If we can't possibly beat the best length, stop.
        if len(seq) + len(available) <= best_len:
            return
            
        # If current seq is already better, update best
        if len(seq) > best_len:
            best_seq = seq[:]
            best_sums = sums.copy() # Copy is expensive, but we only do it on improvement
            best_len = len(seq)
            # Optional: if we hit a very high length, we might tighten search?
            # For now, keep searching for even longer.
        
        if not available:
            return
            
        # Heuristic: Try candidates in a specific order.
        # Randomized order helps escape local minima of the greedy approach.
        # We can also bias towards numbers that create "larger" gaps or "smaller" numbers.
        # For distinct sums, smaller numbers pack more but collide more.
        # Let's just shuffle.
        
        candidates = list(available)
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                return
                
            new_sums = get_new_sums(seq, x, sums)
            if new_sums is None:
                continue
                
            # Recurse
            seq.append(x)
            sums.update(new_sums)
            available.remove(x)
            
            backtrack(seq, sums, available)
            
            # Backtrack
            seq.pop()
            for s in new_sums:
                sums.discard(s)
            available.add(x)
            
            # If we found a very long sequence, we can stop early?
            # Theoretical max length for N=256 is not known precisely, but likely < 20.
            # If we reach 15-16, we are doing well.
            # Let's not hardcode, but rely on time.

    # Run multiple restarts of the backtracking process
    # Starting from the greedy prefix might be a good heuristic.
    # Or start from scratch with randomized greedy?
    
    # Strategy 1: Start with greedy prefix, then backtrack to extend it.
    # Strategy 2: Randomized greedy from scratch, then backtrack.
    
    # Let's mix them.
    # First, try to extend the greedy sequence further.
    # The greedy sequence is maximal in the sense that no single next number from [1,256] works.
    # But backtracking can find a different path.
    
    # We will run the backtracking search starting from an empty set,
    # but guided by a preference for the greedy choices?
    # Actually, pure random DFS on 256 items might be too slow if branching factor is high.
    # The constraint "distinct sums" is very strong.
    
    # Let's try a different approach: Local Search / Simulated Annealing?
    # Or just more aggressive random restarts of the greedy with different tie-breaking?
    
    # Let's stick to DFS but limit the depth or branching.
    # To avoid exponential explosion, we can use a "beam search" or just limit the number of branches explored per node?
    # No, DFS with pruning is usually effective for constraint satisfaction.
    
    # Key issue: The `get_new_sums` function iterates over `reversed(seq)`.
    # If seq length is ~15, this is fast.
    # The main cost is the number of nodes visited.
    
    # Let's run a few restarts.
    # In each restart, we can randomize the candidate order more strongly.
    
    # To make the search more effective, we can use a "conflict-based" heuristic.
    # But simple shuffle is a good baseline.
    
    # Let's try running the backtracker starting from the empty set,
    # but we will use a time-sliced approach: run for a few seconds, then restart with different random seed.
    
    # However, since we already have a greedy baseline, we can use it to prune.
    # If we start from the greedy prefix, we save time.
    # But the greedy prefix might be a dead end for extending, whereas a different prefix might be better.
    
    # Let's do this:
    # 1. Start with the greedy sequence.
    # 2. Try to extend it using backtracking.
    # 3. If that fails or time runs out, try starting from scratch with randomization.
    
    # Actually, the greedy sequence is *maximal* (no single extension).
    # So backtracking from the full greedy sequence won't extend it directly.
    # We need to backtrack *before* the end of the greedy sequence to find alternative branches.
    
    # So, let's run the backtracker from scratch, but with a bias towards the greedy choices.
    # Or simply run random DFS.
    
    # Let's run random DFS with a time limit.
    # To improve success rate, we can randomize the initial candidate list.
    
    # One optimization: Sort candidates by "potential" (e.g., distance to existing sums)?
    # Too complex.
    
    # Let's just do random DFS.
    
    # We will run the backtracking search in a loop, with random restarts.
    # Each restart starts with an empty sequence and a shuffled set of candidates.
    # To make it faster, we can limit the depth of the search or use a heuristic to stop early.
    
    # Let's try running the backtracker.
    # We need to be careful about recursion depth and stack overflow.
    # Max depth 256 is fine for Python default recursion limit (1000).
    
    # To speed up, we can use a set for `available` and `sums`.
    
    # Let's run a few iterations of the backtracker.
    # Since it's random, we can just run it until the deadline.
    
    # But we should probably restart it if it gets stuck in a deep, unproductive branch.
    # A simple way to do this is to use a timeout for each "root" of the search tree?
    # Or just let it run.
    
    # Let's try running the backtracker starting from the empty set.
    # We will shuffle the candidates at each step (already done in backtrack).
    
    # To increase diversity, we can randomize the *initial* choice more strongly.
    # The `random.shuffle` in `backtrack` does this.
    
    # Let's run the backtracker.
    
    # One potential issue: If the greedy baseline is length L, and the backtracker
    # explores a branch that reaches length L but cannot extend, it might waste time.
    # The pruning `len(seq) + len(available) <= best_len` helps here.
    
    # Let's execute the backtracking search.
    # We will do this in a loop, allowing multiple "re-rootings" if we want,
    # but a single DFS from root should explore the whole space (modulo pruning).
    # So one call to backtrack([], set(), set(range(1, 257))) should suffice,
    # provided it doesn't get stuck in a local minimum due to the ordering.
    # But since it's DFS with backtracking, it *will* find the optimal solution
    # if the time limit allows.
    
    # The risk is that it takes too long to find a good solution.
    # So, we can run multiple *independent* random DFS searches in parallel?
    # No, Python is single-threaded (GIL).
    # We can interleave them: run DFS for 1 second, stop, restart with new random seed.
    
    # Let's implement a time-sliced DFS.
    
    def time_sliced_dfs(timeout_seconds):
        nonlocal best_seq, best_sums, best_len
        
        # We will run the DFS, but check time more frequently.
        # The current `backtrack` function already checks time.
        # So we can just call it.
        
        # To make it "restartable", we can just call it again with a new random state?
        # The `random.shuffle` inside will use the global random state.
        # So just calling it again will continue with a different random sequence.
        
        # But DFS is deterministic given the random sequence.
        # So restarting just means continuing the search from the root with a new random order.
        
        # Let's just call the backtrack function.
        # It will update best_seq/best_len if it finds something better.
        
        # However, if the first DFS explores a huge part of the tree and finds a length L,
        # the second DFS (restart) might not find anything better because of the pruning.
        # This is actually good: it focuses on the unexplored parts or proves optimality?
        # No, it just prunes branches that can't beat L.
        
        # So, let's just run the DFS until the deadline.
        
        # One improvement: Use a "beam" of the top K candidates at each step?
        # This would be a heuristic search, not exact.
        # But it might find good solutions faster.
        
        # Let's stick to exact DFS for now, as N=256 is small enough that the constraint
        # "distinct sums" is very restrictive, leading to low branching factors.
        
        backtrack([], set(), set(range(1, 257)))
        
    # Run the DFS
    # We will run it in a loop, but the DFS itself will run until it finishes or time runs out.
    # Since the DFS is recursive and checks time at every node, it will stop when time is up.
    # But if it finishes early (exploring the whole space), we are done.
    # If it doesn't finish, it will stop at the deadline.
    
    # To be safe, we can wrap it in a try-except for RecursionError?
    # Max depth 256 is safe.
    
    # Let's just call it.
    backtrack([], set(), set(range(1, 257)))
    
    # Verify the best sequence found
    # (Optional, but good for debugging)
    
    return best_seq

# EVOLVE-BLOCK-END
