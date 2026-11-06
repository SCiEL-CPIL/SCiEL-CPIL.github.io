import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    best_set = []
    best_size = 0
    
    # Helper: check if adding x to current_set keeps all sums <= 2
    def can_add_x(x, current_set, current_counts):
        for y in current_set:
            s = x + y
            if current_counts.get(s, 0) >= 2:
                return False
        return True
    
    # Helper: add x and update counts
    def add_x(x, current_set, current_counts):
        current_set.append(x)
        for y in current_set:
            s = x + y
            current_counts[s] = current_counts.get(s, 0) + 1
    
    # Helper: remove x and update counts
    def remove_x(x, current_set, current_counts):
        current_set.remove(x)
        for y in current_set:
            s = x + y
            current_counts[s] -= 1
            if current_counts[s] == 0:
                del current_counts[s]
    
    # Initial greedy pass to get a baseline
    current_set, current_counts = [], {}
    for x in range(1, 1001):
        if can_add_x(x, current_set, current_counts):
            add_x(x, current_set, current_counts)
    
    best_set = list(current_set)
    best_size = len(current_set)
    
    # Spectral gap-closure heuristic: analyze sum-count distribution
    # to find "narrow" gaps where inserting elements maximizes future slots
    
    def analyze_sum_counts(current_counts):
        """Analyze which sums are at capacity (count=2) vs have slack."""
        full_sums = set()
        slack_sums = defaultdict(int)
        for s, c in current_counts.items():
            if c >= 2:
                full_sums.add(s)
            else:
                slack_sums[s] = 2 - c
        return full_sums, slack_sums
    
    def score_candidate(x, current_set, current_counts, full_sums, slack_sums):
        """Score a candidate by how many future insertion slots it preserves.
        
        A good candidate:
        1. Can be added now (all sums x+y have count < 2)
        2. Minimizes the number of sums that become full (count becomes 2)
        3. Maximizes the "slack" remaining in the sum-count distribution
        """
        if not can_add_x(x, current_set, current_counts):
            return -1
        
        # Count how many sums would become full
        newly_full = 0
        total_slack_after = 0
        
        for y in current_set:
            s = x + y
            current_count = current_counts.get(s, 0)
            new_count = current_count + 1
            if new_count == 2:
                newly_full += 1
            total_slack_after += (2 - new_count)
        
        # Also consider x+x
        s_2x = 2 * x
        current_count_2x = current_counts.get(s_2x, 0)
        new_count_2x = current_count_2x + 1
        if new_count_2x == 2:
            newly_full += 1
        total_slack_after += (2 - new_count_2x)
        
        # Score: prefer candidates that create fewer full sums and preserve more slack
        # Use a weighted score: minimize newly_full, maximize total_slack_after
        score = total_slack_after - 10 * newly_full
        return score
    
    # Randomized restarts with spectral gap-closure
    while time.time() < deadline:
        # Try different strategies for ordering
        strategy = random.choice(['random', 'spectral', 'hybrid'])
        
        if strategy == 'random':
            order = list(range(1, 1001))
            random.shuffle(order)
        elif strategy == 'spectral':
            # Start with greedy, then try to improve via gap-closure
            order = list(range(1, 1001))
        else:  # hybrid
            order = list(range(1, 1001))
            random.shuffle(order)
            # But prioritize candidates with higher spectral scores
            # We'll do this in the main loop
        
        current_set, current_counts = [], {}
        
        if strategy == 'spectral':
            # First do a greedy pass
            for x in range(1, 1001):
                if can_add_x(x, current_set, current_counts):
                    add_x(x, current_set, current_counts)
            
            # Then try to improve by swapping elements
            # Find elements that are "blocking" many potential additions
            full_sums, slack_sums = analyze_sum_counts(current_counts)
            
            # Try removing elements and adding better ones
            for _ in range(50):  # Limited number of swap attempts
                if time.time() > deadline:
                    break
                
                if not current_set:
                    break
                
                # Pick a random element to potentially remove
                idx = random.randint(0, len(current_set) - 1)
                x_remove = current_set[idx]
                
                # Try removing it and see if we can add more
                remove_x(x_remove, current_set, current_counts)
                
                # Try to add elements that were previously blocked
                added_any = False
                for x_add in range(1, 1001):
                    if x_add == x_remove:
                        continue
                    if x_add in current_set:
                        continue
                    if can_add_x(x_add, current_set, current_counts):
                        # Score this addition
                        full_sums_temp, slack_sums_temp = analyze_sum_counts(current_counts)
                        score = score_candidate(x_add, current_set, current_counts, full_sums_temp, slack_sums_temp)
                        if score > 0:
                            add_x(x_add, current_set, current_counts)
                            added_any = True
                            break
                
                if not added_any:
                    # Put the removed element back
                    add_x(x_remove, current_set, current_counts)
        
        else:
            # Standard greedy or hybrid
            for x in order:
                if can_add_x(x, current_set, current_counts):
                    add_x(x, current_set, current_counts)
            
            # For hybrid, try some swaps after initial construction
            if strategy == 'hybrid' and time.time() < deadline:
                for _ in range(20):
                    if time.time() > deadline:
                        break
                    
                    if not current_set:
                        break
                    
                    idx = random.randint(0, len(current_set) - 1)
                    x_remove = current_set[idx]
                    
                    remove_x(x_remove, current_set, current_counts)
                    
                    added_any = False
                    for x_add in range(1, 1001):
                        if x_add == x_remove:
                            continue
                        if x_add in current_set:
                            continue
                        if can_add_x(x_add, current_set, current_counts):
                            add_x(x_add, current_set, current_counts)
                            added_any = True
                            break
                    
                    if not added_any:
                        add_x(x_remove, current_set, current_counts)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
    
    return best_set
