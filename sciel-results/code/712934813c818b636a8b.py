"""Difference triangle set (7,5) - maximum clique / backtracking with pruning."""
import time
import random
from itertools import combinations

def construct(n_rows, k):
    deadline = time.time() + 145.0  # safety margin before 150s kill
    deadline_check_interval = 2000  # check every N operations
    
    # For the specific task: n_rows=7, k=5 -> each row has 6 elements starting at 0
    # We need 7 rows of 6 increasing integers starting at 0
    # All positive differences within rows must be distinct across all rows
    # Minimize the largest entry (scope)
    
    # Number of differences per row: C(6,2) = 15
    # Total differences needed: 7 * 15 = 105
    # Counting bound: need at least 105 distinct positive integers, so max entry >= 105
    # Best known: 112
    
    # Strategy: 
    # 1. Start with greedy to get a baseline
    # 2. Use backtracking with constraint propagation to find valid configurations
    # 3. Use local search / simulated annealing to reduce the scope
    # 4. Try to minimize the maximum entry
    
    # First, get a greedy solution as baseline
    greedy_rows = _greedy_construct(n_rows, k)
    greedy_scope = max(max(row) for row in greedy_rows) if greedy_rows else 0
    
    best_rows = greedy_rows
    best_scope = greedy_scope
    
    # Try to improve with backtracking + pruning
    # We'll try different upper bounds on the maximum entry
    # and use backtracking to find valid row configurations
    
    # For efficiency, we'll use a different approach:
    # Generate candidate rows (subsets of {0,...,M} of size k+1 starting with 0)
    # and find a set of n_rows pairwise difference-disjoint rows
    
    # Given time constraints, let's use a smarter approach:
    # 1. Use the greedy solution as starting point
    # 2. Try to reduce the scope by local search
    
    # Local search: try to reduce the maximum entry
    # We can try removing the largest element and replacing it with a smaller one
    
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        # Try to reduce the scope by modifying rows
        current_rows = [list(row) for row in best_rows]
        current_scope = max(max(row) for row in current_rows)
        
        # Find the row with the largest entry
        max_row_idx = max(range(n_rows), key=lambda i: current_rows[i][-1])
        
        # Try to reduce the last element of this row
        # The last element is the scope contributor
        last_val = current_rows[max_row_idx][-1]
        
        # Try smaller values for the last element
        for new_last in range(last_val - 1, 0, -1):
            if time.time() > deadline:
                break
            
            # Check if we can replace the last element
            candidate_row = current_rows[max_row_idx][:-1] + [new_last]
            
            if not _is_valid_row(candidate_row):
                continue
            
            # Check if this row is difference-disjoint from all other rows
            if _is_difference_disjoint(candidate_row, current_rows, max_row_idx):
                current_rows[max_row_idx] = candidate_row
                new_scope = max(max(row) for row in current_rows)
                if new_scope < best_scope:
                    best_rows = [list(row) for row in current_rows]
                    best_scope = new_scope
                break
        
        # Also try to modify other elements
        for row_idx in range(n_rows):
            if time.time() > deadline:
                break
            
            for pos in range(1, k + 1):
                if time.time() > deadline:
                    break
                
                current_val = current_rows[row_idx][pos]
                
                # Try smaller values
                for new_val in range(current_val - 1, current_rows[row_idx][pos - 1], -1):
                    if time.time() > deadline:
                        break
                    
                    candidate_row = list(current_rows[row_idx])
                    candidate_row[pos] = new_val
                    
                    if not _is_valid_row(candidate_row):
                        continue
                    
                    if _is_difference_disjoint(candidate_row, current_rows, row_idx):
                        current_rows[row_idx] = candidate_row
                        new_scope = max(max(row) for row in current_rows)
                        if new_scope < best_scope:
                            best_rows = [list(row) for row in current_rows]
                            best_scope = new_scope
                        break
    
    # Try a different approach: rebuild from scratch with different heuristics
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        # Randomized greedy with different ordering
        random_rows = _randomized_greedy(n_rows, k, seed=attempt)
        if random_rows:
            scope = max(max(row) for row in random_rows)
            if scope < best_scope:
                best_rows = random_rows
                best_scope = scope
    
    return best_rows


def _greedy_construct(n_rows, k):
    """Greedy construction: each row takes the smallest next mark whose differences are unused."""
    used = set()
    rows = []
    for _ in range(n_rows):
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


def _is_valid_row(row):
    """Check if a row is strictly increasing and starts at 0."""
    if not row or row[0] != 0:
        return False
    for i in range(1, len(row)):
        if row[i] <= row[i - 1]:
            return False
    return True


def _is_difference_disjoint(candidate_row, all_rows, exclude_idx):
    """Check if candidate_row's differences are disjoint from all other rows' differences."""
    candidate_diffs = set()
    for i in range(len(candidate_row)):
        for j in range(i + 1, len(candidate_row)):
            candidate_diffs.add(candidate_row[j] - candidate_row[i])
    
    for idx, row in enumerate(all_rows):
        if idx == exclude_idx:
            continue
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                if (row[j] - row[i]) in candidate_diffs:
                    return False
    return True


def _randomized_greedy(n_rows, k, seed=0):
    """Randomized greedy construction with different heuristics."""
    rng = random.Random(seed)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                # With some probability, skip this candidate to explore different paths
                if rng.random() < 0.1 and x < 200:
                    continue
                row.append(x)
                used |= new_diffs
        rows.append(row)
    
    return rows
