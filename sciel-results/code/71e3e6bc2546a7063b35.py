import time
import random
import math
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    row_size = k + 1
    diffs_per_row = row_size * (row_size - 1) // 2
    total_diffs = n_rows * diffs_per_row
    lb = total_diffs  # Lower bound on M

    def get_metrics(rows):
        used = set()
        conflicts = 0
        max_val = 0
        for r in rows:
            for i in range(1, len(r)):
                if r[i] <= r[i-1]:
                    return float('inf'), float('inf')
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        conflicts += 1
                    else:
                        used.add(d)
            max_val = max(max_val, r[-1])
        return max_val, conflicts

    def generate_all_rows(M):
        """Generate all valid Sidon rows of size row_size starting at 0 with max entry <= M."""
        rows = []
        
        def backtrack(current, x):
            if len(current) == row_size:
                rows.append(current[:])
                return
            # Pruning: need enough room for remaining elements
            remaining = row_size - len(current)
            if x + remaining > M:
                return
            for next_x in range(x + 1, M + 1):
                # Check if adding next_x creates duplicate differences within this row
                new_diffs = set()
                valid = True
                for a in current:
                    d = next_x - a
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if not valid:
                    continue
                # Also check against existing differences in current
                existing_diffs = set()
                for i in range(len(current)):
                    for j in range(i + 1, len(current)):
                        existing_diffs.add(current[j] - current[i])
                if new_diffs & existing_diffs:
                    continue
                current.append(next_x)
                backtrack(current, next_x)
                current.pop()
        
        backtrack([0], 0)
        return rows

    def row_diff_set(row):
        """Get the set of all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def find_clique(all_rows, diff_sets, n_needed, time_limit):
        """Find a clique of size n_needed in the compatibility graph using branch and bound with bitsets."""
        n = len(all_rows)
        if n < n_needed:
            return None
        
        # Build adjacency bitsets: adj[i] is a bitset of nodes compatible with i
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if not (diff_sets[i] & diff_sets[j]):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        found = [None]
        end_time = time.time() + time_limit
        
        # Maximum clique / clique finding via branch and bound
        # We want a clique of exactly n_needed (or at least n_needed)
        # Use a greedy coloring bound for maximum clique, then extract n_needed subset
        
        # Simpler: backtracking to find any clique of size n_needed
        def backtrack_clique(clique, candidates, depth):
            if time.time() > end_time:
                return
            if len(clique) == n_needed:
                found[0] = clique[:]
                return
            if not candidates:
                return
            if len(clique) + len(candidates) < n_needed:
                return
            
            # Try each candidate in candidates
            for idx in range(len(candidates) - 1, -1, -1):
                node = candidates[idx]
                if len(clique) + (len(candidates) - idx) <= n_needed:
                    break  # Not enough remaining to fill
                # Add node to clique
                clique.append(node)
                # New candidates: intersection of current candidates (before idx) with adj[node]
                new_cands = []
                for j in range(idx):
                    c = candidates[j]
                    if (adj[node] >> c) & 1:
                        new_cands.append(c)
                backtrack_clique(clique, new_cands, depth + 1)
                if found[0] is not None:
                    return
                clique.pop()
        
        backtrack_clique([], list(range(n)), 0)
        return found[0]

    def greedy_construct():
        """Greedy construction as fallback."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_size:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # Main strategy: binary search on M using clique finding
    best_rows = None
    best_max = float('inf')

    # First, get a greedy solution as baseline
    greedy_result = greedy_construct()
    best_max, _ = get_metrics(greedy_result)
    best_rows = greedy_result

    # Try to improve using the clique approach
    # Start from lb and go up
    search_start = lb
    search_end = best_max

    if search_start >= search_end:
        # Already at lower bound
        return best_rows

    # Binary search or linear search from lb upward
    # Use linear search from lb since the range might be small
    M = search_start
    max_M_to_try = min(search_end - 1, search_start + 50)  # Don't go above greedy
    bt_deadline = start_time + 100.0  # Leave time for final checks

    while M <= max_M_to_try and time.time() < bt_deadline:
        remaining_time = bt_deadline - time.time()
        if remaining_time < 5:
            break

        # Generate all valid rows for this M
        all_rows = generate_all_rows(M)
        
        if len(all_rows) >= n_rows:
            # Compute difference sets
            diff_sets = [row_diff_set(r) for r in all_rows]
            
            # Find clique
            clique = find_clique(all_rows, diff_sets, n_rows, remaining_time - 5)
            
            if clique is not None:
                best_rows = [all_rows[i][:] for i in clique]
                best_max = M
                break
        M += 1

    # If we didn't find improvement, try simulated annealing on the greedy solution
    if best_rows == greedy_result and time.time() < start_time + 120:
        current_rows = [r[:] for r in best_rows]
        current_max = best_max
        
        # Build used_diffs and conflict tracking
        used_diffs = {}
        for r in current_rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    used_diffs[d] = used_diffs.get(d, 0) + 1
        
        conflict_count = sum(1 for v in used_diffs.values() if v > 1)
        
        T = 30.0
        cooling = 0.997
        sa_deadline = start_time + 120.0
        
        while time.time() < sa_deadline and T > 0.3:
            r_idx = random.randint(0, n_rows - 1)
            j_idx = random.randint(1, row_size - 1)
            
            row = current_rows[r_idx]
            old_val = row[j_idx]
            
            # Compute old diffs involving position j_idx
            old_diffs = []
            for i in range(j_idx):
                old_diffs.append(old_val - row[i])
            for i in range(j_idx + 1, len(row)):
                old_diffs.append(row[i] - old_val)
            
            # Try random new value
            max_bound = max(current_max, lb)
            new_val = random.randint(1, max_bound)
            
            if j_idx > 0 and new_val <= row[j_idx - 1]:
                T *= cooling
                continue
            if j_idx < row_size - 1 and new_val >= row[j_idx + 1]:
                T *= cooling
                continue
            
            # Compute new diffs
            new_diffs = []
            for i in range(j_idx):
                new_diffs.append(new_val - row[i])
            for i in range(j_idx + 1, len(row)):
                new_diffs.append(row[i] - new_val)
            
            # Apply move
            row[j_idx] = new_val
            
            # Update used_diffs
            for d in old_diffs:
                used_diffs[d] -= 1
                if used_diffs[d] == 0:
                    del used_diffs[d]
            for d in new_diffs:
                used_diffs[d] = used_diffs.get(d, 0) + 1
            
            new_conflict_count = sum(1 for v in used_diffs.values() if v > 1)
            new_max = max(r[-1] for r in current_rows)
            
            old_score = conflict_count * 10000 + current_max
            new_score = new_conflict_count * 10000 + new_max
            delta = new_score - old_score
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                conflict_count = new_conflict_count
                if new_conflict_count == 0 and new_max < best_max:
                    best_max = new_max
                    best_rows = [r[:] for r in current_rows]
                current_max = new_max
            else:
                # Revert
                row[j_idx] = old_val
                for d in old_diffs:
                    used_diffs[d] = used_diffs.get(d, 0) + 1
                for d in new_diffs:
                    used_diffs[d] -= 1
                    if used_diffs[d] == 0:
                        del used_diffs[d]
            
            T *= cooling
            
            if best_max == lb:
                break

    return best_rows
