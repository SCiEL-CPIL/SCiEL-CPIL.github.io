import time
import random
import sys

sys.setrecursionlimit(10000)

def _check_extension(seq, sums, x):
    """
    Check if adding x to seq is valid.
    Returns the new sums to add if valid, or None if invalid.
    """
    tails = []
    total = x
    tails.append(total)
    for a in reversed(seq):
        total += a
        tails.append(total)
        
    # Check for duplicates within the new tails
    # Since seq is strictly increasing and positive, tails are strictly increasing?
    # tails = [x, x+a_n, x+a_n+a_{n-1}, ...]
    # a_i > 0, so tails are strictly increasing. So no duplicates within tails.
    # We only need to check against existing sums.
    
    # Optimization: Check intersection
    # Convert tails to a set for fast lookup? No, tails is small (len(seq)+1).
    # sums is a set.
    # We can iterate through tails and check if any is in sums.
    for s in tails:
        if s in sums:
            return None
            
    return tails

def _dfs_backtrack(seq, sums, last, best_seq, best_len, deadline, heuristic='smallest'):
    """
    Depth-first search with backtracking.
    seq: current sequence
    sums: set of all consecutive block sums
    last: last element in seq
    best_seq: reference to the list holding the best sequence found
    best_len: current best length
    deadline: time limit
    heuristic: 'smallest' or 'random'
    """
    current_len = len(seq)
    if current_len > best_len:
        best_len = current_len
        best_seq[:] = seq[:]
        
    if time.time() > deadline:
        return best_len
        
    # Generate candidates
    # Candidates must be > last and <= 256
    # To avoid generating huge lists, we can iterate or generate a small set.
    # Since we want to maximize length, smaller numbers are generally preferred 
    # as they leave more room for future numbers.
    
    candidates = list(range(last + 1, 257))
    
    if heuristic == 'random':
        random.shuffle(candidates)
    # else: already sorted smallest to largest
        
    for x in candidates:
        if time.time() > deadline:
            return best_len
            
        new_tails = _check_extension(seq, sums, x)
        if new_tails is not None:
            # Valid move
            seq.append(x)
            sums.update(new_tails)
            
            # Recurse
            best_len = _dfs_backtrack(seq, sums, x, best_seq, best_len, deadline, heuristic)
            
            # Backtrack
            seq.pop()
            for s in new_tails:
                sums.discard(s)
                
            # Pruning: If we found a sequence longer than possible with remaining numbers?
            # Max possible length is bounded by 256 - last.
            # But the sum constraint is tighter.
            
            # Early termination if we hit the theoretical max?
            # Theoretical max length L is such that sum of 1..L <= 256*L? No.
            # Just let it run until deadline.
            
    return best_len

def _greedy_smallest():
    """Baseline greedy strategy."""
    values = []
    sums = set()
    for x in range(1, 257):
        tails = _check_extension(values, sums, x)
        if tails is not None:
            values.append(x)
            sums.update(tails)
    return values

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85 seconds to be safe within 90s limit
    
    best_seq = []
    best_len = 0
    
    # Strategy 1: Greedy Smallest
    # This is fast and provides a good initial lower bound
    g_seq = _greedy_smallest()
    if len(g_seq) > best_len:
        best_len = len(g_seq)
        best_seq = g_seq[:]
        
    if time.time() > deadline:
        return best_seq
        
    # Strategy 2: DFS with Backtracking (Smallest First)
    # This explores the space more deeply, backtracking when greedy fails.
    # We limit the search by the deadline.
    # We start with the greedy sequence? No, start from scratch to allow different branches.
    # Or, we can start DFS from the greedy sequence and try to extend/modify?
    # Modification is hard. Better to run independent DFS.
    
    # To make DFS efficient, we can use the greedy result as a guide.
    # But a fresh DFS with backtracking is more robust.
    
    # We will run a few independent DFS searches with different random seeds/orders.
    
    for i in range(10): # Try up to 10 randomized DFS searches
        if time.time() > deadline:
            break
            
        # Randomize the heuristic for candidate ordering
        # Using 'random' allows escaping local minima
        # We also randomize the starting point? No, must start from 1..256.
        
        # To avoid re-doing the same search, we can slightly perturb the candidate order.
        # Or just use 'random' which shuffles the entire candidate list at each node.
        # This is expensive if the branching factor is high, but here it's likely low.
        
        # Let's try a hybrid: Smallest first, but with a random "jump" occasionally?
        # Simpler: Just run DFS with 'smallest' first, then 'random'.
        
        if i == 0:
            # Try deterministic smallest-first backtracking
            # This might get stuck if the first few choices lead to a dead end.
            # But it's a good start.
            # We'll just let the random ones handle the exploration.
            pass
            
        # Run DFS with random candidate ordering
        # We need a fresh state
        seq = []
        sums = set()
        last = 0
        
        # We can use the best_len found so far as a pruning heuristic?
        # If current_len + (256 - last) <= best_len, prune?
        # Not necessarily, because we might pick larger numbers and skip many.
        # But if we are picking numbers > last, the max possible additional elements is 256 - last.
        # So if len(seq) + (256 - last) <= best_len, we can prune.
        
        # Let's add this pruning to the DFS logic inside _dfs_backtrack?
        # It's easier to just let it run and rely on the time limit.
        
        # Actually, _dfs_backtrack doesn't have access to the pruning logic easily.
        # Let's modify _dfs_backtrack to accept a prune function or just keep it simple.
        
        # For now, run the DFS.
        # To speed up, we can limit the depth of backtracking?
        # No, let it run.
        
        # Use a different random seed for each iteration to get different paths
        random.seed(i)
        
        # We call _dfs_backtrack with heuristic 'random'
        # This will shuffle candidates at each node.
        _dfs_backtrack(seq, sums, last, best_seq, best_len, deadline, heuristic='random')
        
    # One more try with 'smallest' to see if backtracking helps there
    if time.time() < deadline:
        seq = []
        sums = set()
        last = 0
        _dfs_backtrack(seq, sums, last, best_seq, best_len, deadline, heuristic='smallest')
        
    return best_seq

# To make the DFS more effective, we should add the pruning check.
# Let's rewrite _dfs_backtrack to include the simple pruning:
# If len(seq) + (256 - last) <= best_len, return.

def _dfs_backtrack_optimized(seq, sums, last, best_seq, best_len, deadline, heuristic='smallest'):
    current_len = len(seq)
    if current_len > best_len:
        best_len = current_len
        best_seq[:] = seq[:]
        
    if time.time() > deadline:
        return best_len
        
    # Pruning: Even if we pick the smallest possible next numbers, 
    # can we beat the best_len?
    # Max possible length from here is current_len + (256 - last)
    if current_len + (256 - last) <= best_len:
        return best_len
        
    candidates = list(range(last + 1, 257))
    
    if heuristic == 'random':
        random.shuffle(candidates)
        
    for x in candidates:
        if time.time() > deadline:
            return best_len
            
        new_tails = _check_extension(seq, sums, x)
        if new_tails is not None:
            seq.append(x)
            sums.update(new_tails)
            
            best_len = _dfs_backtrack_optimized(seq, sums, x, best_seq, best_len, deadline, heuristic)
            
            seq.pop()
            for s in new_tails:
                sums.discard(s)
                
    return best_len

# Update construct to use the optimized version
def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    best_seq = []
    best_len = 0
    
    # Strategy 1: Greedy Smallest
    g_seq = _greedy_smallest()
    if len(g_seq) > best_len:
        best_len = len(g_seq)
        best_seq = g_seq[:]
        
    if time.time() > deadline:
        return best_seq
        
    # Strategy 2: DFS with Backtracking
    # We run multiple iterations with randomization to explore the space.
    for i in range(20):
        if time.time() > deadline:
            break
            
        random.seed(i)
        seq = []
        sums = set()
        last = 0
        
        # Alternate between random and smallest?
        # Random is better for escaping local minima.
        # Smallest is better for finding long sequences quickly if the path is clear.
        # Let's use 'random' for exploration.
        
        _dfs_backtrack_optimized(seq, sums, last, best_seq, best_len, deadline, heuristic='random')
        
    # Final check with smallest-first backtracking
    if time.time() < deadline:
        seq = []
        sums = set()
        last = 0
        _dfs_backtrack_optimized(seq, sums, last, best_seq, best_len, deadline, heuristic='smallest')
        
    return best_seq
