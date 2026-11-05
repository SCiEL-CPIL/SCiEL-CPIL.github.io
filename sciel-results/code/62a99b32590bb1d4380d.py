import random
import time
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Total number of differences that must be distinct
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    best_rows = None
    best_max = float('inf')
    
    # Helper to compute max entry
    def get_max_entry(rows):
        return max(max(row) for row in rows)
    
    # Fast validity check using a set
    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    # Compute the number of conflicting differences
    def count_conflicts(rows):
        used = {}
        conflicts = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        used[d] += 1
                        conflicts += 1
                    else:
                        used[d] = 1
        return conflicts
    
    # Pure greedy construction
    def pure_greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    # Initial baseline
    baseline_rows = pure_greedy()
    best_rows = [row[:] for row in baseline_rows]
    best_max = get_max_entry(best_rows)
    
    # UCB1-based gap selection for local search
    # Each "arm" corresponds to a candidate gap value (distance between consecutive elements)
    # We maintain counts and cumulative rewards for each gap value
    
    # Define the range of gap values to consider
    # Gaps are differences between consecutive elements in a row
    # We'll consider gaps from 1 to best_max
    
    # For UCB1, we need to track:
    # - count[gap]: number of times gap has been tried
    # - reward[gap]: cumulative reward for gap
    
    # The reward for trying a gap is based on how well it reduces the maximum entry
    # while maintaining validity
    
    # We'll use a simpler approach: for each position in each row, we try to reduce
    # the element by selecting a new gap that leads to a smaller value
    
    # Phase 1: Greedy tightening with conflict resolution
    # Try to reduce each element as much as possible while maintaining validity
    
    def greedy_tighten(rows):
        """Try to reduce each element in the rows while maintaining validity."""
        improved = True
        while improved:
            improved = False
            for r in range(n_rows):
                for c in range(k, 0, -1):  # Process from right to left
                    old_val = rows[r][c]
                    # Try to reduce this value
                    min_val = rows[r][c - 1] + 1
                    # Try values from min_val to old_val - 1
                    for new_val in range(min_val, old_val):
                        rows[r][c] = new_val
                        if is_valid(rows):
                            old_val = new_val
                            improved = True
                            break
                    rows[r][c] = old_val
        return rows
    
    # Apply greedy tightening
    best_rows = greedy_tighten(best_rows)
    best_max = get_max_entry(best_rows)
    
    # Phase 2: UCB1-based local search
    # We maintain a pool of candidate moves, each associated with a gap value
    # The UCB1 score balances exploitation (high reward) and exploration (low count)
    
    # We'll track statistics for each (row, col, new_val) combination indirectly
    # through the gap values
    
    # For efficiency, we'll use a simpler approach:
    # - Maintain a set of "promising" gap values based on past success
    # - Use UCB1 to select which gap to try next
    
    # Track gap statistics
    gap_counts = {}
    gap_rewards = {}
    
    def get_ucb_score(gap, exploration_const=1.414):
        """Compute UCB1 score for a gap value."""
        if gap not in gap_counts or gap_counts[gap] == 0:
            return float('inf')
        avg_reward = gap_rewards[gap] / gap_counts[gap]
        exploitation = avg_reward
        exploration = exploration_const * math.sqrt(math.log(100) / gap_counts[gap])
        return exploitation + exploration
    
    def update_gap_stats(gap, reward):
        """Update statistics for a gap value."""
        if gap not in gap_counts:
            gap_counts[gap] = 0
            gap_rewards[gap] = 0
        gap_counts[gap] += 1
        gap_rewards[gap] += reward
    
    # Local search with UCB1-guided move selection
    # We'll try to reduce the maximum entry by modifying elements
    
    # First, identify the rows with the maximum entry
    # Then try to reduce those entries
    
    # For each element that can be reduced, we try different gap values
    # The UCB1 score guides which gap to try next
    
    # We'll maintain a priority queue of candidate moves
    # Each move is (row, col, new_val, gap)
    
    # For simplicity, we'll use a round-robin approach with UCB1 bias
    
    # Track which gaps have been tried for each position
    position_gap_counts = {}
    position_gap_rewards = {}
    
    def get_position_ucb_score(r, c, gap, exploration_const=1.414):
        """Compute UCB1 score for a specific position and gap."""
        key = (r, c, gap)
        if key not in position_gap_counts or position_gap_counts[key] == 0:
            return float('inf')
        avg_reward = position_gap_rewards[key] / position_gap_counts[key]
        exploitation = avg_reward
        exploration = exploration_const * math.sqrt(math.log(100) / position_gap_counts[key])
        return exploitation + exploration
    
    def update_position_stats(r, c, gap, reward):
        """Update statistics for a specific position and gap."""
        key = (r, c, gap)
        if key not in position_gap_counts:
            position_gap_counts[key] = 0
            position_gap_rewards[key] = 0
        position_gap_counts[key] += 1
        position_gap_rewards[key] += reward
    
    # Local search phase
    # We'll iterate through all positions and try to reduce them
    # Using UCB1 to guide which new values to try
    
    # For each position, we maintain a list of candidate new values
    # The UCB1 score determines which candidate to try next
    
    # We'll use a temperature-based acceptance criterion
    # to allow some exploration
    
    T = 50.0
    cooling_rate = 0.9995
    T_min = 0.5
    
    # Current state
    current_rows = [row[:] for row in best_rows]
    current_max = best_max
    
    # Compute current cost (conflicts + max_entry)
    def compute_cost(rows):
        conflicts = count_conflicts(rows)
        max_entry = get_max_entry(rows)
        return max_entry + 100 * conflicts
    
    current_cost = compute_cost(current_rows)
    
    # Main search loop
    steps = 0
    max_steps = 100000
    
    while time.time() < deadline and steps < max_steps and T > T_min:
        steps += 1
        
        # Select a random row and column to modify
        r = random.randint(0, n_rows - 1)
        c = random.randint(1, k)
        
        # Determine the range of possible new values
        min_val = current_rows[r][c - 1] + 1
        max_val = best_max - 1
        if c < k:
            max_val = min(max_val, current_rows[r][c + 1] - 1)
        
        if min_val > max_val:
            T *= cooling_rate
            continue
        
        # Get the current value
        old_val = current_rows[r][c]
        
        # Generate candidate new values
        # We'll try a few candidates based on UCB1 scores
        candidates = []
        
        # Always include the current value
        candidates.append(old_val)
        
        # Add some random candidates
        for _ in range(3):
            if min_val <= max_val:
                candidates.append(random.randint(min_val, max_val))
        
        # Add candidates based on UCB1 for different gaps
        # The gap is the difference between consecutive elements
        if c > 1:
            prev_gap = current_rows[r][c] - current_rows[r][c - 1]
            # Try gaps near the previous gap
            for delta in [-2, -1, 1, 2]:
                new_gap = prev_gap + delta
                if new_gap > 0:
                    new_val = current_rows[r][c - 1] + new_gap
                    if min_val <= new_val <= max_val:
                        candidates.append(new_val)
        
        # Remove duplicates
        candidates = list(set(candidates))
        
        # Score each candidate using UCB1
        scored_candidates = []
        for new_val in candidates:
            if new_val == old_val:
                continue
            # Compute the gap for this candidate
            gap = new_val - current_rows[r][c - 1]
            ucb_score = get_position_ucb_score(r, c, gap)
            scored_candidates.append((ucb_score, new_val, gap))
        
        if not scored_candidates:
            T *= cooling_rate
            continue
        
        # Sort by UCB score (descending) and pick the best
        scored_candidates.sort(key=lambda x: -x[0])
        
        # With some probability, pick a random candidate instead of the best
        if random.random() < 0.2:
            idx = random.randint(0, len(scored_candidates) - 1)
        else:
            idx = 0
        
        _, new_val, gap = scored_candidates[idx]
        
        # Apply the move
        current_rows[r][c] = new_val
        new_cost = compute_cost(current_rows)
        delta_cost = new_cost - current_cost
        
        # Acceptance criterion
        accept = False
        if delta_cost < 0:
            accept = True
        elif T > 0:
            if random.random() < math.exp(-delta_cost / T):
                accept = True
        
        if accept:
            current_cost = new_cost
            # Update UCB1 statistics with a reward based on cost improvement
            reward = -delta_cost  # Positive reward for improvement
            update_position_stats(r, c, gap, reward)
            update_gap_stats(gap, reward)
            
            # Check if we improved the best solution
            if is_valid(current_rows):
                curr_max = get_max_entry(current_rows)
                if curr_max < best_max:
                    best_max = curr_max
                    best_rows = [row[:] for row in current_rows]
        else:
            # Reject the move
            current_rows[r][c] = old_val
            # Update statistics with a penalty
            reward = -1.0
            update_position_stats(r, c, gap, reward)
            update_gap_stats(gap, reward)
        
        T *= cooling_rate
    
    # Final greedy tightening
    best_rows = greedy_tighten(best_rows)
    best_max = get_max_entry(best_rows)
    
    return best_rows
# EVOLVE-BLOCK-END
