import time
import random
import math
import sys
import numpy as np

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k):
    """
    Standard greedy initialization: each row takes the smallest next mark with no difference collision.
    """
    rows = []
    used = set()
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _row_diffs_set(row):
    """Compute the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _compute_all_used(rows):
    """Compute the union of all differences across all rows."""
    all_used = set()
    for row in rows:
        all_used |= _row_diffs_set(row)
    return all_used

def _global_max(rows):
    """Compute the maximum entry across all rows."""
    return max(max(row) for row in rows)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Initialization
    rows = _greedy_init(n_rows, k)
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    
    # Q-Learning setup
    # State: (current_max, num_diffs_used, row_index, position)
    # Action: next value to place
    # We'll use a simple Q-table with state discretization
    
    # For Q-learning, we need to define states and actions
    # State: (current_max_entry, number_of_diffs_used, row_idx, pos_idx)
    # Action: next value (discretized)
    
    # Q-table: state -> action -> Q-value
    # We'll use a dictionary for sparse Q-table
    Q = {}
    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount factor
    epsilon = 0.3  # Exploration rate
    
    def get_state(current_rows, ri, pi, current_max, all_used):
        # Discretize state to make Q-table manageable
        # Use: (current_max // 5, len(all_used) // 10, ri, pi)
        return (current_max // 5, len(all_used) // 10, ri, pi)
    
    def get_actions(current_rows, ri, pi, current_max, all_used):
        """Get valid actions for placing next value at position pi in row ri."""
        actions = []
        min_val = current_rows[ri][pi - 1] + 1 if pi > 0 else 1
        # Upper bound: current_max + 20 to allow some exploration
        max_val = current_max + 20
        for v in range(min_val, max_val + 1):
            # Check if this value creates valid differences
            new_diffs = set()
            valid = True
            for i in range(pi):
                d = v - current_rows[ri][i]
                if d in all_used:
                    valid = False
                    break
                new_diffs.add(d)
            if valid:
                # Also check internal consistency
                for i in range(pi):
                    for j in range(i + 1, pi):
                        d = current_rows[ri][j] - current_rows[ri][i]
                        if d in new_diffs:
                            valid = False
                            break
                    if not valid:
                        break
            if valid:
                actions.append(v)
        return actions
    
    def select_action(state, actions, epsilon):
        """Epsilon-greedy action selection."""
        if random.random() < epsilon:
            return random.choice(actions)
        else:
            # Choose action with highest Q-value
            best_action = actions[0]
            best_q = -float('inf')
            for a in actions:
                q_val = Q.get((state, a), 0.0)
                if q_val > best_q:
                    best_q = q_val
                    best_action = a
            return best_action
    
    def update_q(state, action, reward, next_state, next_actions, epsilon):
        """Update Q-value using Bellman equation."""
        if next_actions:
            next_action = select_action(next_state, next_actions, epsilon)
            next_q = Q.get((next_state, next_action), 0.0)
            target = reward + gamma * next_q
        else:
            target = reward
        
        old_q = Q.get((state, action), 0.0)
        Q[(state, action)] = old_q + alpha * (target - old_q)
    
    # Q-Learning loop
    current_rows = [r[:] for r in rows]
    all_used = _compute_all_used(current_rows)
    current_max = _global_max(current_rows)
    
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a row to modify
        ri = random.randrange(n_rows)
        
        # Pick a position to modify (start from the end to minimize max)
        pi = k  # Start from the last position
        
        # Get current state
        state = get_state(current_rows, ri, pi, current_max, all_used)
        
        # Get valid actions
        actions = get_actions(current_rows, ri, pi, current_max, all_used)
        
        if not actions:
            no_improve += 1
            continue
        
        # Select action using Q-learning
        action = select_action(state, actions, epsilon)
        
        # Apply the action
        old_val = current_rows[ri][pi]
        current_rows[ri][pi] = action
        
        # Compute new state and reward
        new_all_used = _compute_all_used(current_rows)
        new_max = _global_max(current_rows)
        
        # Reward: negative of the max (we want to minimize)
        # Bonus for reducing max
        reward = -new_max
        if new_max < current_max:
            reward += 10  # Bonus for improvement
        
        # Compute next state (for Q-update)
        next_pi = pi - 1
        if next_pi >= 1:
            next_state = get_state(current_rows, ri, next_pi, new_max, new_all_used)
            next_actions = get_actions(current_rows, ri, next_pi, new_max, new_all_used)
        else:
            next_state = None
            next_actions = []
        
        # Update Q-value
        if next_state is not None:
            update_q(state, action, reward, next_state, next_actions, epsilon)
        
        # Update state
        current_max = new_max
        all_used = new_all_used
        
        # Check for improvement
        if current_max < best_val:
            best_val = current_max
            best_rows = [r[:] for r in current_rows]
            no_improve = 0
            # Decay epsilon
            epsilon = max(0.05, epsilon * 0.95)
        else:
            no_improve += 1
        
        # Occasionally restart from best to avoid getting stuck
        if no_improve > 2000:
            current_rows = [r[:] for r in best_rows]
            all_used = _compute_all_used(current_rows)
            current_max = _global_max(current_rows)
            no_improve = 0
            epsilon = max(0.1, epsilon * 0.8)
    
    return best_rows

# EVOLVE-BLOCK-END
