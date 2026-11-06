import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin (evaluator kills at 150s)
    max_n = 1000
    best_set = []
    best_size = 0

    # Helper: compute representation counts for a set
    # Returns a dict or array where counts[s] = number of pairs (a,b) in S with a<=b and a+b=s
    def compute_counts(S):
        S_list = list(S)
        n = len(S_list)
        counts = np.zeros(2 * max_n + 1, dtype=np.int32)
        for i in range(n):
            for j in range(i, n):
                s = S_list[i] + S_list[j]
                counts[s] += 1
        return counts

    # Fast validity check: can we add x to S?
    # Returns True if adding x doesn't violate B2[2]
    def can_add(x, S_list, counts):
        # Check 2*x
        if counts[2 * x] >= 2:
            return False
        # Check x + y for all y in S_list
        # We need to check if any count[x+y] >= 2
        sums = x + np.array(S_list, dtype=np.int32)
        # Filter sums that are in valid range
        valid_mask = (sums >= 1) & (sums <= 2 * max_n)
        if not np.all(valid_mask):
            sums = sums[valid_mask]
        if len(sums) > 0:
            if np.any(counts[sums] >= 2):
                return False
        return True

    # Add x to S, update counts
    def do_add(x, S_list, counts):
        S_list.append(x)
        s_2x = 2 * x
        counts[s_2x] += 1
        # For each y in S_list[:-1], update count[x+y]
        for y in S_list[:-1]:
            s = x + y
            counts[s] += 1

    # Remove x from S, update counts
    def do_remove(x, S_list, counts):
        idx = S_list.index(x)
        S_list.pop(idx)
        s_2x = 2 * x
        counts[s_2x] -= 1
        for y in S_list:
            s = x + y
            counts[s] -= 1

    # Greedy construction with a given order
    def run_greedy(order):
        S_list = []
        counts = np.zeros(2 * max_n + 1, dtype=np.int32)
        for c in order:
            if can_add(c, S_list, counts):
                do_add(c, S_list, counts)
        return S_list, counts

    # Local search with insert and remove moves
    def local_search(initial_list, initial_counts, max_iterations=5000):
        S_list = list(initial_list)
        S_set = set(S_list)
        counts = initial_counts.copy()
        C_set = set(range(1, max_n + 1)) - S_set
        
        # Track conflict involvement: for each element in S, how many sums s have counts[s]==2 and involve this element
        # We'll recompute when needed or maintain incrementally
        
        def get_conflict_counts():
            """Compute conflict_count for each element in S_list"""
            conflict_count = {y: 0 for y in S_list}
            for s in range(2, 2 * max_n + 1):
                if counts[s] == 2:
                    # Find pairs in S_list that sum to s
                    for i, a in enumerate(S_list):
                        b = s - a
                        if b in S_set and a <= b:
                            conflict_count[a] += 1
                            if a != b:
                                conflict_count[b] += 1
            return conflict_count

        best_local = len(S_list)
        best_local_list = list(S_list)
        best_local_counts = counts.copy()
        
        conflict_count = get_conflict_counts()
        
        for it in range(max_iterations):
            if time.time() - start_time > deadline:
                break
                
            # Try insert moves: pick random candidates from C_set
            if C_set:
                candidates = random.sample(list(C_set), min(20, len(C_set)))
                random.shuffle(candidates)
                inserted = False
                for x in candidates:
                    if can_add(x, S_list, counts):
                        do_add(x, S_list, counts)
                        S_set.add(x)
                        C_set.remove(x)
                        # Update conflict_count
                        conflict_count = get_conflict_counts()
                        inserted = True
                        if len(S_list) > best_local:
                            best_local = len(S_list)
                            best_local_list = list(S_list)
                            best_local_counts = counts.copy()
                        break
                
                if inserted:
                    continue
            
            # No insert possible. Try remove moves to free up capacity
            if S_list:
                # Pick element with max conflict_count
                if all(v == 0 for v in conflict_count.values()):
                    # No conflicts, but we're stuck. Pick random to shuffle
                    y = random.choice(S_list)
                else:
                    max_conf = max(conflict_count.values())
                    candidates_remove = [y for y in S_list if conflict_count[y] == max_conf]
                    y = random.choice(candidates_remove)
                
                do_remove(y, S_list, counts)
                S_set.remove(y)
                C_set.add(y)
                # Update conflict_count
                conflict_count = get_conflict_counts()
        
        return best_local_list, best_local_counts

    # 1. Deterministic Ascending Greedy
    asc_order = list(range(1, max_n + 1))
    current_list, current_counts = run_greedy(asc_order)
    best_size = len(current_list)
    best_set = list(current_list)
    
    # Apply local search
    if time.time() - start_time < deadline - 5.0:
        improved_list, improved_counts = local_search(best_set, current_counts)
        if len(improved_list) > best_size:
            best_size = len(improved_list)
            best_set = improved_list

    # 2. Random Restarts
    while time.time() - start_time < deadline - 10.0:
        order = list(range(1, max_n + 1))
        random.shuffle(order)
        current_list, current_counts = run_greedy(order)
        
        if len(current_list) > best_size:
            best_size = len(current_list)
            best_set = list(current_list)
        
        # Try to improve with local search
        if time.time() - start_time < deadline - 5.0:
            improved_list, improved_counts = local_search(current_list, current_counts)
            if len(improved_list) > best_size:
                best_size = len(improved_list)
                best_set = improved_list

    return best_set
