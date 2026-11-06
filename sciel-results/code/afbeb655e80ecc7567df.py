import time
import random

# Global variables for the backtracking solver
best_set = []
best_size = 0
sums = set()
start_time = 0
time_limit = 110.0 # Leave 10s margin for 120s limit

def can_add(x, current_set, current_sums):
    # Check if x can be added to current_set
    # Sums to check: x + y for y in current_set, and 2*x
    for y in current_set:
        s = x + y
        if s in current_sums:
            return False
    if 2 * x in current_sums:
        return False
    return True

def add_element(x, current_set, current_sums):
    # Add x to set and update sums
    for y in current_set:
        current_sums.add(x + y)
    current_sums.add(2 * x)
    current_set.append(x)

def remove_element(x, current_set, current_sums):
    # Remove x from set and update sums
    for y in current_set:
        if y != x: # x is the last element, so y < x usually, but set is unsorted in this list? 
            # We maintain sorted order in the list? 
            # In DFS, we append x, so x is the largest.
            # So we just remove sums involving x.
            current_sums.discard(x + y)
    current_sums.discard(2 * x)
    current_set.pop()

def dfs(current_set, current_sums, start_val, target_size):
    global best_set, best_size, sums
    
    if time.time() - start_time > time_limit:
        return False # Time exceeded
    
    current_size = len(current_set)
    if current_size > best_size:
        best_size = current_size
        best_set = current_set[:]
        sums = current_sums.copy()
        
    if current_size == target_size:
        return True
        
    # Heuristic: Try candidates from start_val to 284
    # To improve search, we can skip values that are "too close" to existing ones?
    # No, just iterate.
    
    # Optimization: If remaining slots > remaining candidates, prune?
    # Not easy to calculate remaining candidates dynamically without lookahead.
    
    for x in range(start_val, 285):
        if time.time() - start_time > time_limit:
            return False
            
        if can_add(x, current_set, current_sums):
            add_element(x, current_set, current_sums)
            if dfs(current_set, current_sums, x + 1, target_size):
                return True
            remove_element(x, current_set, current_sums)
            
    return False

def construct():
    global start_time, best_set, best_size, sums, time_limit
    
    start_time = time.time()
    
    # Initialize with prescribed elements
    initial_set = [1, 2, 5, 11]
    initial_sums = set()
    for i in range(len(initial_set)):
        for j in range(i, len(initial_set)):
            initial_sums.add(initial_set[i] + initial_set[j])
            
    best_set = initial_set[:]
    best_size = len(initial_set)
    sums = initial_sums.copy()
    
    # Target size 20
    target_size = 20
    
    # First, try a deterministic DFS
    # To avoid getting stuck in a specific branch, we might want to randomize slightly or use multiple restarts
    # However, the prompt suggests backtracking is good.
    
    # We can try to improve the search by shuffling the candidate order if the first attempt is slow.
    # But simple iteration is a good start.
    
    # To increase the chance of finding a solution, we can try different "strategies" for candidate selection
    # Strategy 1: Increasing order
    # Strategy 2: Randomized order (with restarts)
    
    # Let's run the deterministic DFS first with a short time limit to see if it finds 20 quickly
    # If not, we might need a more robust search.
    
    # Actually, a pure DFS might be too slow if it explores deep dead ends.
    # Let's add a "randomized restart" mechanism.
    
    # We'll run the DFS. If it returns False (didn't find 20), we might try again with different heuristics.
    # But for now, the deterministic DFS with time limit is the baseline.
    
    # One improvement: Use a "greedy" pre-fill to get a larger initial set, then backtrack?
    # The initial set is small.
    
    # Let's try to run the DFS.
    dfs(initial_set, initial_sums, 12, target_size)
    
    # If we didn't reach 20, we might have a smaller best_set.
    # We can try a local search or randomized restarts if time remains.
    
    if best_size < 20 and time.time() - start_time < time_limit - 5:
        # Try randomized restarts with different candidate orderings
        # We'll shuffle the range of candidates and try to build the set
        # This is a heuristic to escape local minima in the search tree
        
        attempts = 0
        while time.time() - start_time < time_limit - 5 and attempts < 10:
            attempts += 1
            # Reset for a new search attempt
            # We can't just reset global time, but we can run a limited DFS
            
            # Create a shuffled list of candidates
            candidates = list(range(12, 285))
            random.shuffle(candidates)
            
            # Define a local DFS that uses the shuffled candidates
            # This is complex to implement cleanly with the global state.
            # Instead, let's just rely on the deterministic DFS which is systematic.
            # If the deterministic DFS is fast enough, it's better.
            
            pass # Skipping complex restarts for simplicity and reliability within time constraints
            
    return best_set
