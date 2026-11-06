import time
import heapq
from itertools import combinations

def construct():
    """
    Find a Sidon set containing 0 with all differences 1..30, minimizing the largest element.
    Uses Dijkstra's algorithm on the state space of (Sidon set, covered differences),
    where the cost is the maximum element in the set.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Leave 5s margin before 150s kill

    TARGET = set(range(1, 31))
    TARGET_MASK = (1 << 30) - 1

    def is_sidon_fast(S):
        """Check if S is a Sidon set using a set of differences."""
        S = sorted(S)
        diffs = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d = S[j] - S[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def get_diffs(S):
        S = sorted(S)
        diffs = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                diffs.add(S[j] - S[i])
        return diffs

    def covers_all(S):
        diffs = get_diffs(S)
        return TARGET.issubset(diffs)

    def score_set(S):
        diffs = get_diffs(S)
        missing = len(TARGET - diffs)
        return (missing, max(S) if S else 0)

    def dijkstra_search(max_val, deadline):
        """
        Dijkstra's algorithm on state space of (frozenset(S), diff_mask).
        Cost = max element in S. We want to minimize this.
        
        State: (max_element, frozenset(S), diff_mask)
        We use a priority queue ordered by max_element.
        """
        # State: (max_elem, S_tuple, diff_mask)
        # S_tuple is a sorted tuple for hashability
        # diff_mask is a 30-bit integer
        
        # Initial state: S = (0,), diff_mask = 0
        init_state = (0, (0,), 0)
        
        # Priority queue: (max_elem, S_tuple, diff_mask)
        # We want to minimize max_elem
        pq = [(0, (0,), 0)]
        
        # visited: set of (S_tuple, diff_mask) we've already processed
        # But we can prune: if we've seen the same S_tuple with a superset of diff_mask
        # at the same or lower max_elem, we don't need to process again.
        # For simplicity, just track visited states.
        visited = set()
        
        best_solution = None
        best_max_elem = float('inf')
        
        while pq:
            if time.time() > deadline:
                break
                
            max_elem, S_tuple, diff_mask = heapq.heappop(pq)
            
            # If we've found a complete solution, we can stop (Dijkstra guarantees optimality)
            if diff_mask == TARGET_MASK:
                if max_elem < best_max_elem:
                    best_max_elem = max_elem
                    best_solution = list(S_tuple)
                break
            
            # Skip if already visited
            state_key = (S_tuple, diff_mask)
            if state_key in visited:
                continue
            visited.add(state_key)
            
            # If set is too large, skip
            if len(S_tuple) >= 32:
                continue
                
            # Generate next states: add a new element x
            S_list = list(S_tuple)
            current_max = max(S_list) if S_list else 0
            
            # Candidate x values: from 1 to max_val
            # But we can be smarter: only consider x that creates at least one new target diff
            # or helps cover uncovered diffs.
            
            # For each uncovered target difference d, x must be y+d or y-d for some y in S
            uncovered = [d for d in range(1, 31) if not (diff_mask & (1 << (d-1)))]
            
            candidate_x = set()
            for d in uncovered:
                for y in S_list:
                    x1 = y + d
                    if 1 <= x1 <= max_val:
                        candidate_x.add(x1)
                    x2 = y - d
                    if 1 <= x2 <= max_val:
                        candidate_x.add(x2)
            
            # Also try small values that might create multiple diffs
            for x in range(1, min(max_val, current_max + 10) + 1):
                if x not in S_tuple:
                    candidate_x.add(x)
            
            for x in sorted(candidate_x):
                if x in S_tuple:
                    continue
                if x > max_val:
                    continue
                
                # Check if adding x keeps the set Sidon
                new_diffs = []
                valid = True
                for y in S_list:
                    d = abs(x - y)
                    if d == 0:
                        valid = False
                        break
                    # Check against existing diffs
                    if diff_mask & (1 << (d-1)):
                        valid = False
                        break
                    # Check against new diffs
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Update diff_mask
                new_diff_mask = diff_mask
                for d in new_diffs:
                    new_diff_mask |= (1 << (d-1))
                
                # New state
                new_S_tuple = tuple(sorted(S_list + [x]))
                new_max = max(new_S_tuple)
                
                # Only push if this is a new state or improves on previous
                new_key = (new_S_tuple, new_diff_mask)
                if new_key not in visited:
                    heapq.heappush(pq, (new_max, new_S_tuple, new_diff_mask))
        
        return best_solution

    def greedy_fallback(deadline):
        """Simple greedy fallback if Dijkstra doesn't find a solution in time."""
        S = [0]
        covered = set()
        
        for d in range(1, 31):
            if d in covered:
                continue
            found = False
            for x in range(1, 1000000):
                if time.time() > deadline:
                    break
                if x in S:
                    continue
                trial = sorted(S + [x])
                if is_sidon_fast(trial):
                    trial_diffs = get_diffs(trial)
                    if d in trial_diffs:
                        S = trial
                        covered = trial_diffs & TARGET
                        found = True
                        break
            if not found and time.time() > deadline:
                break
        
        return S

    # Try Dijkstra with increasing bounds
    bounds = [30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 300, 500]
    
    best_S = None
    best_score = (30, 10**9)
    
    for bound in bounds:
        if time.time() > DEADLINE - 10:
            break
        
        sol = dijkstra_search(bound, DEADLINE - 10)
        
        if sol is not None:
            if is_sidon_fast(sol) and covers_all(sol):
                s = score_set(sol)
                if s < best_score:
                    best_score = s
                    best_S = sol
                    break  # Dijkstra finds optimal for this bound
    
    # Fallback
    if best_S is None:
        best_S = greedy_fallback(DEADLINE - 2)
    
    return best_S
